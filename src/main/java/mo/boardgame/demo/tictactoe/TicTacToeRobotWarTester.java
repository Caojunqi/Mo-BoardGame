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
        Random random = new Random(0);
        NDManager mainManager = NDManager.newBaseManager();
        TicTacToeEnv gameEnv = new TicTacToeEnv(mainManager.newSubManager(), random, true);
        String[] modelFiles = new String[]{"0117", "0115"};
        RobotWarEnv robotWarEnv = new RobotWarEnv(mainManager.newSubManager(), random, gameEnv, modelFiles);
        robotWarEnv.run();
    }
}
