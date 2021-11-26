package mo.boardgame.demo.tictactoe.env;

import env.state.space.IStateSpace;

/**
 * 井字棋状态空间
 *
 * @author Caojunqi
 * @date 2021-11-25 14:22
 */
public class TicTacToeStateSpace implements IStateSpace<TicTacToeState> {

    /**
     * 状态空间，3*3*2，棋盘大小为3*3，每个位置的取值范围为[0,2]，其中0表示未落子，1表示玩家1落子，2表示玩家2落子。
     */
    private float[][][] spaces;

    public TicTacToeStateSpace(int boardWidth, int boardHeight) {
        float[][][] spaces = new float[boardWidth][boardHeight][2];
        for (int w = 0; w < boardWidth; w++) {
            for (int h = 0; h < boardHeight; h++) {
                spaces[w][h][0] = 0;
                spaces[w][h][1] = 2;
            }
        }
        this.spaces = spaces;
    }

    @Override
    public int getDim() {
        return 3;
    }
}
