package mo.boardgame.demo.tictactoe;

import ai.djl.ndarray.NDManager;
import mo.boardgame.game.SelfPlayEnv;

import java.util.Random;

/**
 * 井字棋训练类
 *
 * @author Caojunqi
 * @date 2021-12-02 15:18
 */
public class TicTacToeTrainer {

    public static void main(String[] args) {
        int epoch = 500;
        int replayBufferSize = 2048;

        Random random = new Random(0);
        NDManager mainManager = NDManager.newBaseManager();
        TicTacToeEnv gameEnv = new TicTacToeEnv(mainManager, random, replayBufferSize, replayBufferSize);
        SelfPlayEnv selfPlayEnv = new SelfPlayEnv(mainManager, random, gameEnv, replayBufferSize, replayBufferSize);
        for (int i = 0; i < epoch; i++) {
            selfPlayEnv.train();
        }
    }
}
