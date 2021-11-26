package mo.boardgame.demo.tictactoe.env;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import env.state.collector.IStateCollector;
import env.state.core.IState;
import utils.DjlUtils;

/**
 * 井字棋状态数据收集器
 *
 * @author Caojunqi
 * @date 2021-11-25 18:17
 */
public class TicTacToeStateCollector implements IStateCollector {

    private float[][][][] stateDatas;

    public TicTacToeStateCollector(int batchSize) {
        this.stateDatas = new float[batchSize][][][];
    }

    @Override
    public void addState(int index, IState state) {
        this.stateDatas[index] = ((TicTacToeState) state).getStateData();
    }

    @Override
    public NDArray createNDArray(NDManager manager) {
        return DjlUtils.create(manager, stateDatas);
    }
}
