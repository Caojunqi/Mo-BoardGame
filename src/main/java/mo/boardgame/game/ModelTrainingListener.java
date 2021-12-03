package mo.boardgame.game;

import ai.djl.training.Trainer;
import ai.djl.training.listener.TrainingListenerAdapter;
import mo.boardgame.common.ConstantParameter;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;

/**
 * 模型数据处理监听器
 *
 * @author Caojunqi
 * @date 2021-12-03 11:58
 */
public class ModelTrainingListener extends TrainingListenerAdapter {
    /**
     * 模型训练步数
     */
    private int step;
    /**
     * 模型评估频率
     */
    private int evalFreq;
    /**
     * 每次评估模型时需要循环的次数
     */
    private int evalEpisodes;
    /**
     * 环境
     */
    private SelfPlayEnv selfPlayEnv;
    /**
     * 阈值，在self-play环境中，对手的能力是不断提升的，所以只要新模型的reward超过阈值，就表明，新模型优于对手。
     */
    private float threshold;
    /**
     * 模型代数
     */
    private int generation;

    public ModelTrainingListener(SelfPlayEnv selfPlayEnv, int evalFreq, int evalEpisodes, float threshold) {
        this.evalFreq = evalFreq;
        this.evalEpisodes = evalEpisodes;
        this.selfPlayEnv = selfPlayEnv;
        this.threshold = threshold;
    }

    @Override
    public void onTrainingBatch(Trainer trainer, BatchData batchData) {
        this.step++;
        if (this.evalFreq <= 0 || this.step % this.evalFreq != 0) {
            return;
        }

        super.onTrainingBatch(trainer, batchData);

        float totalEvalRewards = 0;
        for (int i = 0; i < evalEpisodes; i++) {
            float reward = selfPlayEnv.eval();
            totalEvalRewards += reward;
        }
        float meanEvalReward = totalEvalRewards / evalEpisodes;
        if (meanEvalReward <= this.threshold) {
            return;
        }
        this.generation++;
        try {
            trainer.getModel().save(buildBestModelPath(), null);
        } catch (IOException e) {
            throw new IllegalStateException("Best Model Save Error!!");
        }
    }

    private Path buildBestModelPath() {
        String gameName = this.selfPlayEnv.getGameEnv().getName();
        String modelFullDir = ConstantParameter.MODEL_DIR +
                gameName +
                ConstantParameter.DIR_SEPARATOR +
                ConstantParameter.BEST_MODEL_PREFIX +
                this.generation +
                ConstantParameter.MODEL_FILE_EXTENSION_SPLIT +
                ConstantParameter.MODEL_SUFFIX;
        File modelFile = new File(modelFullDir);
        return modelFile.toPath();
    }
}
