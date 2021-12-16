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
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;

import java.io.File;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * 机器人混战：用于测试不同版本的机器人之间的性能
 *
 * @author Caojunqi
 * @date 2021-12-08 22:04
 */
public class RobotWarEnv {
    /**
     * 用于从模型文件名称中获取到epoch信息的正则表达式
     */
    private static final Pattern EPOCH_PATTERN = Pattern.compile(Pattern.quote(ConstantParameter.BEST_MODEL_PREFIX) + "-(\\d{4})");

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

    public RobotWarEnv(NDManager manager, Random random, BaseBoardGameEnv gameEnv) {
        this.manager = manager;
        this.random = random;
        this.gameEnv = gameEnv;
        this.agents = new RlAgent[this.gameEnv.getPlayerNum()];
    }

    public void run() {
        File modelDir = new File(ConstantParameter.MODEL_DIR + gameEnv.getName() + ConstantParameter.DIR_SEPARATOR);
        Collection<File> modelFiles = FileUtils.listFiles(modelDir, null, true);
        if (modelFiles.isEmpty() || modelFiles.size() < 2) {
            return;
        }
        List<File> sortedFiles = new ArrayList<>(modelFiles);
        sortedFiles.sort(Comparator.comparing(File::getName));
        File bestModelFile = sortedFiles.get(sortedFiles.size() - 1);
        String bestModelName = FilenameUtils.removeExtension(bestModelFile.getName());
        Matcher bestM = EPOCH_PATTERN.matcher(bestModelName);
        if (!bestM.matches()) {
            throw new IllegalStateException("训练出来的模型名称不规范，不能获取到epoch！！ 模型名称：" + bestModelName);
        }
        String bestModelEpoch = bestM.group(1);
        for (int i = 0; i < sortedFiles.size() - 1; i++) {
            File modelFile = sortedFiles.get(i);
            String modelName = FilenameUtils.removeExtension(modelFile.getName());
            Matcher m = EPOCH_PATTERN.matcher(modelName);
            if (!m.matches()) {
                throw new IllegalStateException("训练出来的模型名称不规范，不能获取到epoch！！ 模型名称：" + bestModelName);
            }
            String modelEpoch = m.group(1);

            String[] modelEpochs = new String[]{bestModelEpoch, modelEpoch};
            runOnePairEpochs(modelEpochs);
        }
    }

    /**
     * 测试一对模型
     */
    public void runOnePairEpochs(String[] modelEpochs) {
        buildAgents(modelEpochs);
        float[] warResult = new float[this.gameEnv.getPlayerNum()];
        for (int i = 0; i < WAR_TIMES; i++) {
            float[] result = onceWar();
            for (int j = 0; j < this.gameEnv.getPlayerNum(); j++) {
                warResult[j] += result[j];
            }
        }
        System.out.println("战斗结束，比分：" + output(modelEpochs, warResult));
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
    private void buildAgents(String[] modelEpochs) {
        if (modelEpochs == null || modelEpochs.length != this.gameEnv.getPlayerNum()) {
            throw new IllegalArgumentException("智能体模型数量不对，需要[" + this.gameEnv.getPlayerNum() + "]个模型！！");
        }

        for (int i = 0; i < modelEpochs.length; i++) {
            String fullName = ConstantParameter.MODEL_DIR +
                    this.gameEnv.getName() +
                    ConstantParameter.DIR_SEPARATOR;
            File file = new File(fullName);
            Model model = gameEnv.buildBaseModel();
            try {
                Map<String, String> options = new HashMap<>(1);
                options.put("epoch", modelEpochs[i]);
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

    private String output(String[] modelEpochs, float[] warResult) {
        StringBuilder stringBuilder = new StringBuilder();
        for (int i = 0; i < warResult.length; i++) {
            stringBuilder.append(modelEpochs[i]);
            stringBuilder.append("[");
            stringBuilder.append(warResult[i]);
            stringBuilder.append("]");
            stringBuilder.append(",");
        }
        return stringBuilder.toString();
    }
}
