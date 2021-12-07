package mo.boardgame.demo.tictactoe;

import ai.djl.ndarray.NDManager;
import mo.boardgame.game.FightRobotEnv;

import java.util.Random;

/**
 * 井字棋测试类
 *
 * @author Caojunqi
 * @date 2021-12-07 15:36
 */
public class TicTacToeTester {

    public static void main(String[] args) {
        Random random = new Random(0);
        NDManager mainManager = NDManager.newBaseManager();
        TicTacToeEnv gameEnv = new TicTacToeEnv(mainManager.newSubManager(), random, true);
        FightRobotEnv fightRobotEnv = new FightRobotEnv(mainManager.newSubManager(), random, gameEnv);
        fightRobotEnv.run();
    }
}
