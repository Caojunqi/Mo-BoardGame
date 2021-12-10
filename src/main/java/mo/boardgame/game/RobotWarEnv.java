package mo.boardgame.game;

import ai.djl.Model;
import ai.djl.modality.rl.agent.RlAgent;
import ai.djl.modality.rl.env.RlEnv;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.loss.Loss;
import algorithm.ppo2.PPO;
import mo.boardgame.common.ConstantParameter;

import java.io.File;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

/**
 * 机器人混战：用于测试不同版本的机器人之间的性能
 *
 * @author Caojunqi
 * @date 2021-12-08 22:04
 */
public class RobotWarEnv {
    /**
     * 总战斗次数
     */
    private final static int WAR_TIMES = 100;
    /**
     * NDArray主管理器
     */
    private NDManager manager;
    /**
     * 随机数生成器
     */
    private Random random;
    /**
     * 游戏环境
     */
    private BaseBoardGameEnv gameEnv;
    /**
     * 机器人智能体
     */
    private RlAgent[] agents;
    private String[] modelEpochs;

    public RobotWarEnv(NDManager manager, Random random, BaseBoardGameEnv gameEnv, String[] modelEpochs) {
        this.manager = manager;
        this.random = random;
        this.gameEnv = gameEnv;
        this.agents = new RlAgent[this.gameEnv.getPlayerNum()];
        this.modelEpochs = modelEpochs;

        buildAgents();
    }

    public void run() {
        float[] warResult = new float[this.gameEnv.getPlayerNum()];
        for (int i = 0; i < WAR_TIMES; i++) {
            float[] result = onceWar();
            for (int j = 0; j < this.gameEnv.getPlayerNum(); j++) {
                warResult[j] += result[j];
            }
        }
        System.out.println("战斗结束，比分：" + output(warResult));
    }

    private float[] onceWar() {
        gameEnv.reset();
        boolean done = false;
        RlEnv.Step step = null;
        while (!done) {
            NDList action;
            int curPlayerId = this.gameEnv.getCurPlayerId();
            RlAgent robot = this.agents[curPlayerId];
            action = robot.chooseAction(this.gameEnv, false);
            step = this.gameEnv.step(action, false);
            done = step.isDone();
        }
        return step.getReward().toFloatArray();
    }

    /**
     * 构建机器人智能体
     */
    private void buildAgents() {
        if (this.modelEpochs == null || this.modelEpochs.length != this.gameEnv.getPlayerNum()) {
            throw new IllegalArgumentException("智能体模型数量不对，需要[" + this.gameEnv.getPlayerNum() + "]个模型！！");
        }

        for (int i = 0; i < this.modelEpochs.length; i++) {
            String fullName = ConstantParameter.MODEL_DIR +
                    this.gameEnv.getName() +
                    ConstantParameter.DIR_SEPARATOR;
            File file = new File(fullName);
            Model model = gameEnv.buildBaseModel();
            try {
                Map<String, String> options = new HashMap<>(1);
                options.put("epoch", this.modelEpochs[i]);
                model.load(file.toPath(), ConstantParameter.BEST_MODEL_PREFIX, options);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }

            TrainingConfig config = new DefaultTrainingConfig(Loss.l2Loss());
            Trainer trainer = model.newTrainer(config);
            trainer.initialize(gameEnv.getObservationShape());
            trainer.notifyListeners(listener -> listener.onTrainingBegin(trainer));
            PPO agent = new PPO(manager.newSubManager(), random, trainer);
            this.agents[i] = agent;
        }
    }

    private String output(float[] warResult) {
        StringBuilder stringBuilder = new StringBuilder();
        for (int i = 0; i < warResult.length; i++) {
            stringBuilder.append(this.modelEpochs[i]);
            stringBuilder.append("[");
            stringBuilder.append(warResult[i]);
            stringBuilder.append("]");
            stringBuilder.append(",");
        }
        return stringBuilder.toString();
    }
}
