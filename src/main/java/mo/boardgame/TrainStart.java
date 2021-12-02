package mo.boardgame;

import env.action.core.impl.DiscreteAction;
import mo.boardgame.demo.tictactoe.env.TicTacToeEnv;
import mo.boardgame.demo.tictactoe.env.TicTacToeState;
import mo.boardgame.game.BaseBoardGameEnv;

/**
 * 训练启动
 *
 * @author Caojunqi
 * @date 2021-11-23 11:31
 */
public class TrainStart {

    public static void main(String[] args) {
        BaseBoardGameEnv<TicTacToeState, DiscreteAction> tic = new TicTacToeEnv();

    }

}
