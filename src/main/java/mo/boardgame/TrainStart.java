package mo.boardgame;

import ai.djl.engine.Engine;
import ai.djl.ndarray.NDManager;
import mo.boardgame.game.BaseBoardGameEnv;
import mo.boardgame.game.OpponentType;
import mo.boardgame.game.SelfPlayEnv;

import java.util.Random;

/**
 * 训练启动
 *
 * @author Caojunqi
 * @date 2021-11-23 11:31
 */
public class TrainStart {

	public static void main(String[] args) {
		int epoch = 500;
		int replayBufferSize = 2048;
		BoardGameType gameType = BoardGameType.GOMOKU2;

		Engine.getInstance().setRandomSeed(0);
		Random random = new Random(0);
		NDManager mainManager = NDManager.newBaseManager();
		BaseBoardGameEnv gameEnv = gameType.buildBoardGameEnv(mainManager.newSubManager(), random, false);
		SelfPlayEnv selfPlayEnv = new SelfPlayEnv(mainManager.newSubManager(), random, gameEnv, replayBufferSize, replayBufferSize, OpponentType.MOSTLY_BEST);
		for (int i = 0; i < epoch; i++) {
			selfPlayEnv.train();
		}

	}

}
