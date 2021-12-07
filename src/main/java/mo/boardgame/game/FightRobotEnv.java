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
import algorithm.CommonParameter;
import algorithm.ppo2.PPO;
import mo.boardgame.common.ConstantParameter;
import org.apache.commons.io.FileUtils;

import java.io.File;
import java.util.*;

/**
 * 人机对抗环境
 *
 * @author Caojunqi
 * @date 2021-12-07 15:43
 */
public class FightRobotEnv {

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
     * 玩家索引
     */
    private int playerId;
    /**
     * 对手智能体，索引为{@link this#playerId}的项为null
     */
    private RlAgent[] agents;

    public FightRobotEnv(NDManager manager, Random random, BaseBoardGameEnv gameEnv) {
        this(manager, random, gameEnv, random.nextInt(gameEnv.getPlayerNum()));
    }

    public FightRobotEnv(NDManager manager, Random random, BaseBoardGameEnv gameEnv, int playerId) {
        this.manager = manager;
        this.random = random;
        this.gameEnv = gameEnv;
        this.playerId = playerId;

        setupOpponents();
    }

    public void run() {
        gameEnv.reset();
        boolean done = false;
        while (!done) {
            NDList action;
            int curPlayerId = this.gameEnv.getCurPlayerId();
            if (curPlayerId != this.playerId) {
                // 机器人行动
                RlAgent robot = this.agents[curPlayerId];
                action = robot.chooseAction(this.gameEnv, false);
            } else {
                // 玩家行动
                System.out.println("Enter your action: ");
                Scanner scanner = new Scanner(System.in);
                String actionStr = scanner.nextLine();
                action = this.gameEnv.parsePlayerAction(actionStr);
                if (action == null) {
                    // 输入错误
                    continue;
                }
            }

            RlEnv.Step step = this.gameEnv.step(action, false);
            done = step.isDone();
            gameEnv.render();
        }
    }

    /**
     * 构建对手
     */
    private void setupOpponents() {
        this.agents = new RlAgent[this.gameEnv.getPlayerNum()];
        Model opponentModel = loadOpponentModel();
        for (int i = 0; i < this.gameEnv.getPlayerNum(); i++) {
            if (i == playerId) {
                continue;
            }
            TrainingConfig config = new DefaultTrainingConfig(Loss.l2Loss());
            Trainer trainer = opponentModel.newTrainer(config);
            trainer.initialize(gameEnv.getObservationShape(CommonParameter.INNER_BATCH_SIZE));
            trainer.notifyListeners(listener -> listener.onTrainingBegin(trainer));
            PPO agent = new PPO(manager.newSubManager(), random, trainer);
            this.agents[i] = agent;
        }
    }

    /**
     * 加载训练好的模型
     */
    private Model loadOpponentModel() {
        File modelDir = new File(ConstantParameter.MODEL_DIR + this.gameEnv.getName() + ConstantParameter.DIR_SEPARATOR);
        Collection<File> modelFiles = FileUtils.listFiles(modelDir, null, true);
        if (modelFiles.isEmpty()) {
            return gameEnv.buildBaseModel();
        }
        List<File> sortedFiles = new ArrayList<>(modelFiles);
        sortedFiles.sort(Comparator.comparing(File::getName));
        File bestModelFile = sortedFiles.get(sortedFiles.size() - 1);
        try {
            File bestModelFileDir = new File(ConstantParameter.MODEL_DIR + this.gameEnv.getName() + ConstantParameter.DIR_SEPARATOR);
            Model bestModel = gameEnv.buildBaseModel();
            bestModel.load(bestModelFileDir.toPath(), ConstantParameter.BEST_MODEL_PREFIX, null);
            return bestModel;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

}
