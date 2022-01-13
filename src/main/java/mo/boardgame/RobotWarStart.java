package mo.boardgame;

import ai.djl.ndarray.NDManager;
import mo.boardgame.game.BaseBoardGameEnv;
import mo.boardgame.game.RobotWarEnv;

import java.util.Random;

/**
 * 机器人混战启动
 *
 * @author Caojunqi
 * @date 2021-12-23 11:07
 */
public class RobotWarStart {

	public static void main(String[] args) {
		BoardGameType gameType = BoardGameType.GOMOKU2;

		NDManager mainManager = NDManager.newBaseManager();
		Random random = new Random(0);
		BaseBoardGameEnv gameEnv = gameType.buildBoardGameEnv(mainManager.newSubManager(), random, true);
		RobotWarEnv robotWarEnv = new RobotWarEnv(mainManager.newSubManager(), random, gameEnv);
		robotWarEnv.run();
	}
}
