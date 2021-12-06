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
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.*;

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
    private ReplayBuffer replayBuffer;
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
     * 当前训练好的最佳模型 <模型名称, 模型参数信息>
     */
    private Tuple<String, Model> bestModelInfo;

    public SelfPlayEnv(NDManager manager, Random random, BaseBoardGameEnv gameEnv, int batchSize, int replayBufferSize) {
        this.manager = manager;
        this.random = random;
        this.gameEnv = gameEnv;
        this.batchSize = batchSize;
        this.replayBufferSize = replayBufferSize;
        resetBuffer();

        Model model = gameEnv.buildBaseModel();
        TrainingConfig config = buildDynamicTrainingConfig();
        Trainer trainer = model.newTrainer(config);
        trainer.initialize(gameEnv.getObservationShape(CommonParameter.INNER_BATCH_SIZE));
        trainer.notifyListeners(listener -> listener.onTrainingBegin(trainer));
        this.aiAgent = new PPO(manager, random, trainer);

        loadBestModel();
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
        if (!step.isDone()) {
            RlEnv.Step opponentStep = continueGame();
            if (opponentStep != null) {
                step = opponentStep;
            }
        }

        RlEnv.Step selfPlayStep = new SelfPlayEnvStep(manager.newSubManager(), step, agentPlayerId, action);
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
        // 检验模型训练结果
        float result = runEnvironment(this.aiAgent, false);
        System.out.println("test:" + result);
    }

    public float eval() {
        return runEnvironment(aiAgent, false);
    }

    private void loadBestModel() {
        File modelDir = new File(ConstantParameter.MODEL_DIR + this.gameEnv.getName() + ConstantParameter.DIR_SEPARATOR);
        Collection<File> modelFiles = FileUtils.listFiles(modelDir, null, true);
        if (modelFiles.isEmpty()) {
            if (this.bestModelInfo == null) {
                Model baseModel = gameEnv.buildBaseModel();
                this.bestModelInfo = new Tuple<>(ConstantParameter.BASE_MODEL_NAME, baseModel);
            }
            return;
        }
        List<File> sortedFiles = new ArrayList<>(modelFiles);
        sortedFiles.sort(Comparator.comparing(File::getName));
        File bestModelFile = sortedFiles.get(sortedFiles.size() - 1);
        String bestModelName = FilenameUtils.removeExtension(bestModelFile.getName());
        if (this.bestModelInfo != null && this.bestModelInfo.first.equals(bestModelName)) {
            return;
        }
        try {
            File bestModelFileDir = new File(ConstantParameter.MODEL_DIR + this.gameEnv.getName() + ConstantParameter.DIR_SEPARATOR);
            Model bestModel = gameEnv.buildBaseModel();
            bestModel.load(bestModelFileDir.toPath(), ConstantParameter.BEST_MODEL_PREFIX, null);
            this.bestModelInfo = new Tuple<>(bestModelName, bestModel);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * 构建对手
     */
    private void setupOpponents() {
        loadBestModel();
        for (int i = 0; i < this.gameEnv.getPlayerNum(); i++) {
            if (i == agentPlayerId) {
                continue;
            }
            Model model = this.bestModelInfo.second;
            TrainingConfig config = buildStaticTrainingConfig();
            Trainer trainer = model.newTrainer(config);
            trainer.initialize(gameEnv.getObservationShape(CommonParameter.INNER_BATCH_SIZE));
            trainer.notifyListeners(listener -> listener.onTrainingBegin(trainer));
            PPO agent = new PPO(manager, random, trainer);
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
                .addTrainingListeners(new ModelTrainingListener(this, 10, 10, 0.2f))
                .optOptimizer(
                        Adam.builder().optLearningRateTracker(Tracker.fixed(CommonParameter.LEARNING_RATE)).build());
    }

    private void resetBuffer() {
        this.curBufferSize = 0;
        this.replayBuffer = new FixedBuffer(batchSize, replayBufferSize);
    }

    public BaseBoardGameEnv getGameEnv() {
        return gameEnv;
    }

    static final class SelfPlayEnvStep implements RlEnv.Step {
        private NDManager manager;
        /**
         * 当前AI主角走子后，对手会执行相应的对策，在本回合最后一位对手走子结束后，返回的游戏环境Step
         */
        private RlEnv.Step gameEnvStep;
        /**
         * 当前AI主角的索引
         */
        private int agentId;
        /**
         * 当前AI主角所采取的行为
         */
        private NDList agentAction;

        private SelfPlayEnvStep(NDManager manager, RlEnv.Step gameEnvStep, int agentId, NDList agentAction) {
            this.manager = manager;
            this.gameEnvStep = gameEnvStep;
            this.agentId = agentId;
            this.agentAction = agentAction;
        }

        @Override
        public NDList getPreObservation() {
            return this.gameEnvStep.getPreObservation();
        }

        @Override
        public NDList getAction() {
            return this.agentAction;
        }

        @Override
        public NDList getPostObservation() {
            return this.gameEnvStep.getPostObservation();
        }

        @Override
        public ActionSpace getPostActionSpace() {
            return this.gameEnvStep.getPostActionSpace();
        }

        @Override
        public NDArray getReward() {
            float[] allAgentsRewards = this.gameEnvStep.getReward().flatten().toFloatArray();
            return manager.create(allAgentsRewards[agentId]);
        }

        @Override
        public boolean isDone() {
            return this.gameEnvStep.isDone();
        }

        @Override
        public void close() {
            this.manager.close();
        }
    }

}
