package mo.boardgame.demo.gomoku;

import ai.djl.ndarray.NDManager;
import mo.boardgame.game.FightRobotEnv;

import java.util.Random;

/**
 * @author Caojunqi
 * @date 2021-12-07 22:30
 */
public class GomokuTester {

    public static void main(String[] args) {
        Random random = new Random(0);
        NDManager mainManager = NDManager.newBaseManager();
        GomokuEnv gameEnv = new GomokuEnv(mainManager.newSubManager(), random, true);
        FightRobotEnv fightRobotEnv = new FightRobotEnv(mainManager.newSubManager(), random, gameEnv);
        fightRobotEnv.run();
    }

}
