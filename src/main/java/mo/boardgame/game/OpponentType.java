package mo.boardgame.game;

import ai.djl.Model;
import common.Tuple;
import mo.boardgame.common.ConstantParameter;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;

import java.io.File;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * 左右互搏时的对手类型
 *
 * @author Caojunqi
 * @date 2021-12-09 10:36
 */
public enum OpponentType {

    /**
     * 当前训练好的最佳模型
     */
    BEST {
        @Override
        public Tuple<String, Model> buildModel(BaseBoardGameEnv gameEnv, Tuple<String, Model> oldModelInfo) {
            File modelDir = new File(ConstantParameter.MODEL_DIR + gameEnv.getName() + ConstantParameter.DIR_SEPARATOR);
            Collection<File> modelFiles = FileUtils.listFiles(modelDir, null, true);
            if (modelFiles.isEmpty()) {
                return null;
            }
            List<File> sortedFiles = new ArrayList<>(modelFiles);
            sortedFiles.sort(Comparator.comparing(File::getName));
            File bestModelFile = sortedFiles.get(sortedFiles.size() - 1);
            String bestModelName = FilenameUtils.removeExtension(bestModelFile.getName());
            if (oldModelInfo != null && oldModelInfo.first.equals(bestModelName)) {
                // 最佳模型没有变
                return oldModelInfo;
            }
            try {
                File bestModelFileDir = new File(ConstantParameter.MODEL_DIR + gameEnv.getName() + ConstantParameter.DIR_SEPARATOR);
                Model bestModel = gameEnv.buildBaseModel();
                bestModel.load(bestModelFileDir.toPath(), ConstantParameter.BEST_MODEL_PREFIX, null);
                System.out.println("对手模型切换为：" + bestModelName);
                return new Tuple<>(bestModelName, bestModel);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }
    },
    /**
     * 尽可能选择最佳模型，偶尔随机选择模型
     */
    MOSTLY_BEST {
        @Override
        public Tuple<String, Model> buildModel(BaseBoardGameEnv gameEnv, Tuple<String, Model> oldModelInfo) {
            File modelDir = new File(ConstantParameter.MODEL_DIR + gameEnv.getName() + ConstantParameter.DIR_SEPARATOR);
            Collection<File> modelFiles = FileUtils.listFiles(modelDir, null, true);
            if (modelFiles.isEmpty()) {
                return null;
            }
            List<File> sortedFiles = new ArrayList<>(modelFiles);
            sortedFiles.sort(Comparator.comparing(File::getName));
            File bestModelFile = sortedFiles.get(sortedFiles.size() - 1);
            String bestModelName = FilenameUtils.removeExtension(bestModelFile.getName());

            Random random = gameEnv.getRandom();
            String modelName;
            if (random.nextFloat() < 0.8f) {
                // 选择最佳模型
                modelName = bestModelName;
            } else {
                // 随机选择模型
                File randomModelFile = sortedFiles.get(random.nextInt(sortedFiles.size()));
                modelName = FilenameUtils.removeExtension(randomModelFile.getName());
            }
            if (oldModelInfo != null && oldModelInfo.first.equals(modelName)) {
                // 模型没有变
                return oldModelInfo;
            }
            try {
                File modelFileDir = new File(ConstantParameter.MODEL_DIR + gameEnv.getName() + ConstantParameter.DIR_SEPARATOR);
                Model model = gameEnv.buildBaseModel();
                Map<String, String> options = new HashMap<>(1);
                final Pattern pattern = java.util.regex.Pattern.compile(Pattern.quote(ConstantParameter.BEST_MODEL_PREFIX) + "-(\\d{4})");
                Matcher m = pattern.matcher(modelName);
                if (!m.matches()) {
                    return null;
                }
                options.put("epoch", m.group(1));
                model.load(modelFileDir.toPath(), ConstantParameter.BEST_MODEL_PREFIX, options);
//                System.out.println("对手模型切换为：" + modelName);
                return new Tuple<>(modelName, model);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }
    },
    /**
     * 按照排名，来概率性地选择对手，模型越优秀，被选中为对手的概率就越高
     */
    RANK {
        @Override
        public Tuple<String, Model> buildModel(BaseBoardGameEnv gameEnv, Tuple<String, Model> oldModelInfo) {
            File modelDir = new File(ConstantParameter.MODEL_DIR + gameEnv.getName() + ConstantParameter.DIR_SEPARATOR);
            Collection<File> modelFiles = FileUtils.listFiles(modelDir, null, true);
            if (modelFiles.isEmpty()) {
                return null;
            }
            List<File> sortedFiles = new ArrayList<>(modelFiles);
            sortedFiles.sort(Comparator.comparing(File::getName));
            int rankRandomIndex = MathUtils.rankRandomIndex(gameEnv.getRandom(), sortedFiles.size());

            File modelFile = sortedFiles.get(rankRandomIndex);
            String modelName = FilenameUtils.removeExtension(modelFile.getName());
            if (oldModelInfo != null && oldModelInfo.first.equals(modelName)) {
                // 模型没有变
                return oldModelInfo;
            }
            try {
                File modelFileDir = new File(ConstantParameter.MODEL_DIR + gameEnv.getName() + ConstantParameter.DIR_SEPARATOR);
                Model model = gameEnv.buildBaseModel();
                Map<String, String> options = new HashMap<>(1);
                final Pattern pattern = java.util.regex.Pattern.compile(Pattern.quote(ConstantParameter.BEST_MODEL_PREFIX) + "-(\\d{4})");
                Matcher m = pattern.matcher(modelName);
                if (!m.matches()) {
                    return null;
                }
                options.put("epoch", m.group(1));
                model.load(modelFileDir.toPath(), ConstantParameter.BEST_MODEL_PREFIX, options);
//                System.out.println("对手模型切换为：" + randomModelName);
                return new Tuple<>(modelName, model);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }
    },
    /**
     * 从当前训练好的模型中随机选择
     */
    RANDOM {
        @Override
        public Tuple<String, Model> buildModel(BaseBoardGameEnv gameEnv, Tuple<String, Model> oldModelInfo) {
            File modelDir = new File(ConstantParameter.MODEL_DIR + gameEnv.getName() + ConstantParameter.DIR_SEPARATOR);
            Collection<File> modelFiles = FileUtils.listFiles(modelDir, null, true);
            if (modelFiles.isEmpty()) {
                return null;
            }
            List<File> fileArr = new ArrayList<>(modelFiles);
            File randomModelFile = fileArr.get(gameEnv.getRandom().nextInt(fileArr.size()));
            String randomModelName = FilenameUtils.removeExtension(randomModelFile.getName());
            if (oldModelInfo != null && oldModelInfo.first.equals(randomModelName)) {
                // 模型没有变
                return oldModelInfo;
            }
            try {
                File modelFileDir = new File(ConstantParameter.MODEL_DIR + gameEnv.getName() + ConstantParameter.DIR_SEPARATOR);
                Model model = gameEnv.buildBaseModel();
                Map<String, String> options = new HashMap<>(1);
                final Pattern pattern = java.util.regex.Pattern.compile(Pattern.quote(ConstantParameter.BEST_MODEL_PREFIX) + "-(\\d{4})");
                Matcher m = pattern.matcher(randomModelName);
                if (!m.matches()) {
                    return null;
                }
                options.put("epoch", m.group(1));
                model.load(modelFileDir.toPath(), ConstantParameter.BEST_MODEL_PREFIX, options);
//                System.out.println("对手模型切换为：" + randomModelName);
                return new Tuple<>(randomModelName, model);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }
    },
    ;

    /**
     * 构建对手模型
     *
     * @param gameEnv      游戏
     * @param oldModelInfo 最近一次使用的模型信息
     * @return 接下来要使用的模型信息
     */
    public abstract Tuple<String, Model> buildModel(BaseBoardGameEnv gameEnv, Tuple<String, Model> oldModelInfo);
}
