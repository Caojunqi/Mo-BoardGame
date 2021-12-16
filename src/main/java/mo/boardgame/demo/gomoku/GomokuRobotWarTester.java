package mo.boardgame.demo.gomoku;

import ai.djl.ndarray.NDManager;
import mo.boardgame.game.RobotWarEnv;

import java.util.Random;

/**
 * 五子棋机器人混战
 *
 * @author Caojunqi
 * @date 2021-12-08 22:25
 */
public class GomokuRobotWarTester {

    public static void main(String[] args) {
        NDManager mainManager = NDManager.newBaseManager();
        Random random = new Random(0);
        GomokuEnv gameEnv = new GomokuEnv(mainManager.newSubManager(), random, true);
        RobotWarEnv robotWarEnv = new RobotWarEnv(mainManager.newSubManager(), random, gameEnv);
        robotWarEnv.run();
    }
}
