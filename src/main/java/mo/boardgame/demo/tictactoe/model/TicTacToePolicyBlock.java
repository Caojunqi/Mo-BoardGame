package mo.boardgame.demo.tictactoe.model;

import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import algorithm.BaseModelBlock;

/**
 * 井字棋策略神经网络
 *
 * @author Caojunqi
 * @date 2021-11-26 15:40
 */
public class TicTacToePolicyBlock extends BaseModelBlock {


    public TicTacToePolicyBlock(int actionDim) {
    }

    @Override
    protected NDList forwardInternal(ParameterStore parameterStore, NDList inputs, boolean training, PairList<String, Object> params) {
        return null;
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        return new Shape[0];
    }

    @Override
    public NDList forward(ParameterStore parameterStore, NDList inputs, boolean training) {
        return null;
    }
}
