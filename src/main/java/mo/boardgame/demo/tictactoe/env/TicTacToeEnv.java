package mo.boardgame.demo.tictactoe.env;

import common.Tuple;
import env.action.core.impl.DiscreteAction;
import env.action.space.impl.DiscreteActionSpace;
import env.common.Environment;
import org.apache.commons.lang3.Validate;
import utils.datatype.Snapshot;

import java.util.ArrayList;
import java.util.List;

/**
 * 井字棋
 *
 * @author Caojunqi
 * @date 2021-11-25 11:23
 */
public class TicTacToeEnv extends Environment<TicTacToeState, DiscreteAction> {
    /**
     * 棋盘长宽
     */
    private static final int GRID_LENGTH = 3;
    /**
     * 玩家数量
     */
    private static final int N_PLAYERS = 2;
    /**
     * 棋盘上格子数量
     */
    private static final int NUM_SQUARES = GRID_LENGTH * GRID_LENGTH;

    /**
     * 当前棋盘现状
     */
    private List<Token> board;
    /**
     * 当前行动玩家索引
     */
    private int curPlayerId;
    /**
     * 当前已走步数
     */
    private int turns;
    /**
     * 当前合法的落子位置
     */
    private int[] legalPositions;

    public TicTacToeEnv() {
        super(new TicTacToeStateSpace(GRID_LENGTH, GRID_LENGTH), new DiscreteActionSpace(NUM_SQUARES));
    }

    @Override
    protected Snapshot<TicTacToeState> doStep(DiscreteAction action) {
        int actionData = action.getActionData();
        Validate.isTrue(actionData < this.board.size());
        boolean done;
        float reward;
        if (this.board.get(actionData).playerId != 0) {
            // not empty
            done = true;
            reward = -1;
        } else {
            this.board.set(actionData, Token.getPlayerToken(this.curPlayerId));
            this.turns++;
            Tuple<Integer, Boolean> result = checkGameOver();
            done = result.second;
            reward = result.first;
        }
        if (!done) {
            this.curPlayerId = (this.curPlayerId + 1) % N_PLAYERS;
        }
        return new Snapshot<>(getCurState(), reward, done);
    }

    @Override
    public TicTacToeState reset() {
        List<Token> board = new ArrayList<>(NUM_SQUARES);
        for (int i = 0; i < NUM_SQUARES; i++) {
            board.add(Token.NONE);
        }
        this.board = board;
        this.curPlayerId = 0;
        this.turns = 0;
        return getCurState();
    }

    @Override
    public void render() {

    }

    @Override
    public void close() {

    }

    /**
     * @return 获取当前棋盘状态
     */
    public TicTacToeState getCurState() {
        float[][][] data = new float[2][][];
        float[][] curPositions = new float[GRID_LENGTH][GRID_LENGTH];
        for (int i = 0; i < this.board.size(); i++) {
            Token token = this.board.get(i);
            int h = i / GRID_LENGTH;
            int w = i % GRID_LENGTH;
            int num = this.curPlayerId == 0 ? 0 : (this.curPlayerId == token.getPlayerId() ? 1 : -1);
            curPositions[h][w] = num;
        }
        float[][] legalPositions = new float[GRID_LENGTH][GRID_LENGTH];
        for (int i = 0; i < this.legalPositions.length; i++) {
            int h = i / GRID_LENGTH;
            int w = i % GRID_LENGTH;
            legalPositions[h][w] = this.legalPositions[i];
        }
        data[0] = curPositions;
        data[1] = legalPositions;
        return new TicTacToeState(data);
    }

    /**
     * 判断棋局是否终了
     *
     * @return 二元组first-收益；二元组second-是否结束
     */
    public Tuple<Integer, Boolean> checkGameOver() {
        for (int i = 0; i < GRID_LENGTH; i++) {
            // horizontals and verticals
            if ((squareIsPlayer(i * GRID_LENGTH, curPlayerId)
                    && squareIsPlayer(i * GRID_LENGTH + 1, curPlayerId)
                    && squareIsPlayer(i * GRID_LENGTH + 2, curPlayerId))
                    || (squareIsPlayer(i, curPlayerId)
                    && squareIsPlayer(i + GRID_LENGTH, curPlayerId)
                    && squareIsPlayer(i + GRID_LENGTH * 2, curPlayerId))) {
                return new Tuple<>(1, true);
            }
        }

        // diagonals
        if ((squareIsPlayer(0, curPlayerId)
                && squareIsPlayer(4, curPlayerId)
                && squareIsPlayer(8, curPlayerId))
                || (squareIsPlayer(6, curPlayerId)
                && squareIsPlayer(4, curPlayerId)
                && squareIsPlayer(2, curPlayerId))) {
            return new Tuple<>(1, true);
        }

        if (this.turns == NUM_SQUARES) {
            return new Tuple<>(0, true);
        }
        return new Tuple<>(0, false);
    }

    /**
     * 判断棋盘上指定位置是否由指定玩家落子
     *
     * @param square   棋盘位置索引
     * @param playerId 玩家索引
     * @return true-square上是playerId的落子；false-square上不是playerId的落子。
     */
    private boolean squareIsPlayer(int square, int playerId) {
        return this.board.get(square).playerId == playerId;
    }


    /**
     * 井字棋棋盘记号
     */
    public enum Token {
        /**
         * 无人落子
         */
        NONE(".", 0),
        /**
         * 玩家1落子
         */
        X("X", 1),
        /**
         * 玩家2落子
         */
        O("O", 2),
        ;

        public static final Token[] VALUES = Token.values();
        /**
         * 字符串表示
         */
        private String symbol;
        /**
         * 对应的玩家索引
         */
        private int playerId;

        Token(String symbol, int playerId) {
            this.symbol = symbol;
            this.playerId = playerId;
        }

        /**
         * 获取指定玩家对应的棋盘记号
         *
         * @param playerId 玩家索引
         * @return 该玩家使用的棋盘记号
         */
        public static Token getPlayerToken(int playerId) {
            for (Token token : VALUES) {
                if (token.playerId == playerId) {
                    return token;
                }
            }
            throw new IllegalArgumentException("不存在指定玩家索引对应的井字棋记号！！playerId:" + playerId);
        }

        public String getSymbol() {
            return symbol;
        }

        public int getPlayerId() {
            return playerId;
        }
    }
}
