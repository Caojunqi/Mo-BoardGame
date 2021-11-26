package mo.boardgame.demo.tictactoe.env;

import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import env.state.core.IState;
import utils.DjlUtils;

/**
 * 井字棋状态数据
 *
 * @author Caojunqi
 * @date 2021-11-25 17:37
 */
public class TicTacToeState implements IState<TicTacToeState> {

    /**
     * 两张棋盘数据，
     * 第一张是玩家的落子信息，该数据是以当前玩家为主视角的，当前玩家的落子点用1表示，对手玩家的落子点用-1表示，未落子点用0表示，
     * 第二张是当前棋盘的合法落子位置信息。
     */
    private float[][][] stateData;

    public TicTacToeState(float[][][] stateData) {
        this.stateData = stateData;
    }

    @Override
    public TicTacToeState clone() {
        return new TicTacToeState(this.stateData.clone());
    }

    @Override
    public Class<TicTacToeStateCollector> getCollectorClz() {
        return TicTacToeStateCollector.class;
    }

    @Override
    public NDList singleStateList(NDManager manager) {
        return new NDList(DjlUtils.create(manager, this.stateData));
    }

    public float[][][] getStateData() {
        return stateData;
    }
}
