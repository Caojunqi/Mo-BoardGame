package mo.boardgame.demo.gomoku;

import ai.djl.ndarray.NDManager;
import mo.boardgame.game.SelfPlayEnv;

import java.util.Random;

/**
 * @author Caojunqi
 * @date 2021-12-07 22:29
 */
public class GomokuTrainer {

    public static void main(String[] args) {
        int epoch = 500;
        int replayBufferSize = 2048;

        Random random = new Random(0);
        NDManager mainManager = NDManager.newBaseManager();
        GomokuEnv gameEnv = new GomokuEnv(mainManager.newSubManager(), random, false);
        SelfPlayEnv selfPlayEnv = new SelfPlayEnv(mainManager.newSubManager(), random, gameEnv, replayBufferSize, replayBufferSize);
        for (int i = 0; i < epoch; i++) {
            selfPlayEnv.train();
        }
    }

}
