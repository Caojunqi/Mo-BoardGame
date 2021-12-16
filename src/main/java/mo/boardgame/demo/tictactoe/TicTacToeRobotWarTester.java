package mo.boardgame.demo.tictactoe;

import ai.djl.ndarray.NDManager;
import mo.boardgame.game.RobotWarEnv;

import java.util.Random;

/**
 * 井字棋机器人混战
 *
 * @author Caojunqi
 * @date 2021-12-08 22:25
 */
public class TicTacToeRobotWarTester {

    public static void main(String[] args) {
        NDManager mainManager = NDManager.newBaseManager();
        Random random = new Random(0);
        TicTacToeEnv gameEnv = new TicTacToeEnv(mainManager.newSubManager(), random, true);
        RobotWarEnv robotWarEnv = new RobotWarEnv(mainManager.newSubManager(), random, gameEnv);
        robotWarEnv.run();
    }
}
