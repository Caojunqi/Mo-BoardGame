package mo.boardgame.game;

import ai.djl.Model;
import ai.djl.modality.rl.ActionSpace;
import ai.djl.modality.rl.ReplayBuffer;
import ai.djl.modality.rl.agent.RlAgent;
import ai.djl.modality.rl.env.RlEnv;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Adam;
import ai.djl.training.tracker.Tracker;
import ai.djl.util.RandomUtils;
import algorithm.CommonParameter;
import algorithm.ppo2.FixedBuffer;
import algorithm.ppo2.PPO;
import common.Tuple;
import mo.boardgame.common.ConstantParameter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Random;

/**
 * “左右互搏”环境
 *
 * @author Caojunqi
 * @date 2021-11-25 11:21
 */
public class SelfPlayEnv implements RlEnv {

    private static final Logger logger = LoggerFactory.getLogger(SelfPlayEnv.class);

    private NDManager manager;
    private Random random;
    /**
     * 游戏环境
     */
    private BaseBoardGameEnv gameEnv;
    private int batchSize;
    private int replayBufferSize;
    /**
     * 样本容器
     */
    private ReplayBuffer replayBuffer;
    /**
     * 对手类型
     */
    private OpponentType opponentType;
    private int curBufferSize;
    /**
     * 当前AI主角，使用的是正在优化的模型，其对手使用的是上一次优化完成的模型
     */
    private int agentPlayerId;
    /**
     * 当前要训练的AI智能体
     */
    private RlAgent aiAgent;
    /**
     * 所有参与游戏的AI主体，处于{@link SelfPlayEnv#agentPlayerId}位置的Agent为null。
     */
    private RlAgent[] agents;
    /**
     * 对手所用的模型信息 <模型名称, 模型参数信息>
     */
    private Tuple<String, Model> opponentModelInfo;

    public SelfPlayEnv(NDManager manager,
                       Random random,
                       BaseBoardGameEnv gameEnv,
                       int batchSize,
                       int replayBufferSize,
                       OpponentType opponentType) {
        this.manager = manager;
        this.random = random;
        this.gameEnv = gameEnv;
        this.batchSize = batchSize;
        this.replayBufferSize = replayBufferSize;
        this.opponentType = opponentType;
        resetBuffer();

        Model model = gameEnv.buildBaseModel();
        TrainingConfig config = buildDynamicTrainingConfig();
        Trainer trainer = model.newTrainer(config);
        trainer.initialize(gameEnv.getObservationShape(CommonParameter.INNER_BATCH_SIZE));
        trainer.notifyListeners(listener -> listener.onTrainingBegin(trainer));
        this.aiAgent = new PPO(manager.newSubManager(), random, trainer);

        loadOpponentModelInfo();
    }

    @Override
    public void reset() {
        gameEnv.reset();
        this.agents = new RlAgent[gameEnv.getPlayerNum()];
        this.agentPlayerId = RandomUtils.nextInt(gameEnv.getPlayerNum());
        setupOpponents();
        if (this.agentPlayerId != gameEnv.getCurPlayerId()) {
            continueGame();
        }
    }

    @Override
    public NDList getObservation() {
        return this.gameEnv.getObservation();
    }

    @Override
    public ActionSpace getActionSpace() {
        return this.gameEnv.getActionSpace();
    }

    @Override
    public RlEnv.Step step(NDList action, boolean training) {
        RlEnv.Step step = this.gameEnv.step(action, training);
        NDList preObservation = step.getPreObservation();
        if (!step.isDone()) {
            RlEnv.Step opponentStep = continueGame();
            if (opponentStep != null) {
                step = opponentStep;
            }
        }

        float[] allAgentsRewards = step.getReward().flatten().toFloatArray();
        float reward = allAgentsRewards[agentPlayerId];
        RlEnv.Step selfPlayStep =
                new SelfPlayEnvStep(manager.newSubManager(),
                        preObservation,
                        action,
                        step.getPostObservation(),
                        step.getPostActionSpace(),
                        reward,
                        step.isDone());
        step.close();
        if (training) {
            replayBuffer.addStep(selfPlayStep);
            this.curBufferSize++;
        }
        return selfPlayStep;
    }

    @Override
    public Step[] getBatch() {
        return this.replayBuffer.getBatch();
    }

    @Override
    public void close() {
        manager.close();
    }

    public void train() {
        int episode = 0;
        float episodeReward = 0;
        resetBuffer();
        // 收集样本数据
        while (this.curBufferSize < replayBufferSize) {
            episode++;
            float result = runEnvironment(this.aiAgent, true);
            episodeReward += result;
        }
        System.out.println("train: episode[" + episode + "], episodeReward[" + episodeReward + "], avgReward[" + episodeReward / episode + "]");
        // 训练模型
        this.aiAgent.trainBatch(getBatch());
    }

    public float eval() {
        return runEnvironment(aiAgent, false);
    }

    /**
     * 加载对手所用模型信息
     */
    private void loadOpponentModelInfo() {
        Tuple<String, Model> newOpponentModelInfo = this.opponentType.buildModel(this.gameEnv, this.opponentModelInfo);
        if (newOpponentModelInfo == null) {
            Model baseModel = gameEnv.buildBaseModel();
            newOpponentModelInfo = new Tuple<>(ConstantParameter.BASE_MODEL_NAME, baseModel);
        }
        this.opponentModelInfo = newOpponentModelInfo;
    }

    /**
     * 构建对手
     */
    private void setupOpponents() {
        loadOpponentModelInfo();
        for (int i = 0; i < this.gameEnv.getPlayerNum(); i++) {
            if (i == agentPlayerId) {
                continue;
            }
            Model model = this.opponentModelInfo.second;
            TrainingConfig config = buildStaticTrainingConfig();
            Trainer trainer = model.newTrainer(config);
            trainer.initialize(gameEnv.getObservationShape(CommonParameter.INNER_BATCH_SIZE));
            trainer.notifyListeners(listener -> listener.onTrainingBegin(trainer));
            PPO agent = new PPO(manager.newSubManager(), random, trainer);
            this.agents[i] = agent;
        }
    }

    private RlEnv.Step continueGame() {
        RlEnv.Step step = null;
        while (this.gameEnv.getCurPlayerId() != this.agentPlayerId) {
            this.gameEnv.render();
            NDList action = getCurAgent().chooseAction(this.gameEnv, true);
            step = this.gameEnv.step(action, true);
            logger.info("Rewards: " + step.getReward());
            logger.info("Done: " + step.isDone());
            if (step.isDone()) {
                break;
            }
        }
        return step;
    }

    private RlAgent getCurAgent() {
        int curPlayerId = this.gameEnv.getCurPlayerId();
        return this.agents[curPlayerId];
    }

    /**
     * @return 构建并返回静态训练配置，用于构建对手Agent的Trainer
     */
    private DefaultTrainingConfig buildStaticTrainingConfig() {
        return new DefaultTrainingConfig(Loss.l2Loss());
    }

    /**
     * @return 构建并返回动态训练配置，用于构建AI主角的Trainer
     */
    private DefaultTrainingConfig buildDynamicTrainingConfig() {
        //TODO 参数统一存放
        return new DefaultTrainingConfig(Loss.l2Loss())
                .addTrainingListeners(new ModelTrainingListener(this, 10, 50, 0.6f))
                .optOptimizer(
                        Adam.builder().optLearningRateTracker(Tracker.fixed(CommonParameter.LEARNING_RATE)).build());
    }

    private void resetBuffer() {
        if (this.replayBuffer != null) {
            for (Step step : this.replayBuffer.getBatch()) {
                if (step != null) {
                    step.close();
                }
            }
        }
        this.curBufferSize = 0;
        this.replayBuffer = new FixedBuffer(batchSize, replayBufferSize);
    }

    public BaseBoardGameEnv getGameEnv() {
        return gameEnv;
    }

    static final class SelfPlayEnvStep implements RlEnv.Step {
        private NDManager manager;
        private NDList preObservation;
        /**
         * 当前AI主角所采取的行为
         */
        private NDList agentAction;
        private NDList postObservation;
        private ActionSpace actionSpace;
        private float reward;
        private boolean done;

        private SelfPlayEnvStep(NDManager manager, NDList preObservation, NDList agentAction, NDList postObservation, ActionSpace actionSpace, float reward, boolean done) {
            this.manager = manager;
            this.preObservation = preObservation;
            this.preObservation.attach(this.manager);
            this.agentAction = agentAction;
            this.agentAction.attach(this.manager);
            this.postObservation = postObservation;
            this.postObservation.attach(this.manager);
            this.actionSpace = actionSpace;
            this.reward = reward;
            this.done = done;
        }

        @Override
        public NDList getPreObservation() {
            return preObservation;
        }

        @Override
        public NDList getAction() {
            return this.agentAction;
        }

        @Override
        public NDList getPostObservation() {
            return postObservation;
        }

        @Override
        public ActionSpace getPostActionSpace() {
            return actionSpace;
        }

        @Override
        public NDArray getReward() {
            return manager.create(reward);
        }

        @Override
        public boolean isDone() {
            return done;
        }

        @Override
        public void close() {
            this.manager.close();
        }
    }

}
