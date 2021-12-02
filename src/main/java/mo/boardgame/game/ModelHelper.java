package mo.boardgame.game;

import ai.djl.Model;
import ai.djl.ndarray.NDList;
import ai.djl.repository.zoo.Criteria;
import ai.djl.translate.NoopTranslator;
import mo.boardgame.common.ConstantParameter;

import java.io.File;
import java.nio.file.Paths;

/**
 * 策略模型工具类
 *
 * @author Caojunqi
 * @date 2021-11-26 17:04
 */
public final class ModelHelper {

    /**
     * 文件夹分隔符
     */
    private final static String DIR_SEPARATOR = "/";

    /**
     * 加载神经网络模型
     *
     * @param gameName  游戏名称，用来定位模型所处文件夹
     * @param modelName 模型名称
     * @return 神经网络模型
     */
    public static Model load(String gameName, String modelName) {
        try {
            String modelFullDir = ConstantParameter.MODEL_DIR + gameName + DIR_SEPARATOR + modelName;
            File modelFile = new File(modelFullDir);
            if (modelFile.exists()) {
                Criteria<NDList, NDList> criteria = Criteria.builder()
                        .setTypes(NDList.class, NDList.class)
                        .optTranslator(new NoopTranslator())
                        .optModelPath(Paths.get(modelFullDir))
                        .optModelName(modelName)
                        .build();
                return criteria.loadModel();
            } else {
                return null;
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
