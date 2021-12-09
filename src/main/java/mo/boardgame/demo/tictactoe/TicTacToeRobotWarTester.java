package mo.boardgame.demo.tictactoe;

import ai.djl.ndarray.NDManager;
import mo.boardgame.common.ConstantParameter;
import mo.boardgame.game.RobotWarEnv;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;

import java.io.File;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * 井字棋机器人混战
 *
 * @author Caojunqi
 * @date 2021-12-08 22:25
 */
public class TicTacToeRobotWarTester {
    /**
     * 用于从模型文件名称中获取到epoch信息的正则表达式
     */
    private static final Pattern EPOCH_PATTERN = java.util.regex.Pattern.compile(Pattern.quote(ConstantParameter.BEST_MODEL_PREFIX) + "-(\\d{4})");

    public static void main(String[] args) {
        Random random = new Random(0);
        NDManager mainManager = NDManager.newBaseManager();
        TicTacToeEnv gameEnv = new TicTacToeEnv(mainManager.newSubManager(), random, true);
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
            RobotWarEnv robotWarEnv = new RobotWarEnv(mainManager.newSubManager(), random, gameEnv, modelEpochs);
            robotWarEnv.run();
        }
    }
}
