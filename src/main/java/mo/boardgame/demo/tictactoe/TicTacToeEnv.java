package mo.boardgame.demo.tictactoe;

import ai.djl.Model;
import ai.djl.modality.rl.ActionSpace;
import ai.djl.modality.rl.ReplayBuffer;
import ai.djl.modality.rl.env.RlEnv;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import algorithm.ppo2.FixedBuffer;
import common.Tuple;
import mo.boardgame.game.BaseBoardGameEnv;
import org.apache.commons.lang3.Validate;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * 井字棋
 *
 * @author Caojunqi
 * @date 2021-11-25 11:23
 */
public class TicTacToeEnv extends BaseBoardGameEnv {

    private final static Logger logger = LoggerFactory.getLogger("chapters.introduction.HelloWorld1");

    private NDManager manager;
    private Random random;
    /**
     * 游戏名称
     */
    private static final String GAME_NAME = "tictactoe";
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
    private ActionSpace actionSpace;
    /**
     * 当前棋盘现状
     */
    private List<Token> board;
    /**
     * 当前已走步数
     */
    private int turns;
    /**
     * 当前合法的落子位置
     */
    private int[] legalPositions;
    private ReplayBuffer replayBuffer;
    /**
     * 游戏是否结束
     */
    private boolean done;

    public TicTacToeEnv(NDManager manager, Random random, int batchSize, int replayBufferSize) {
        super(GAME_NAME, N_PLAYERS);
        this.manager = manager;
        this.random = random;
        this.replayBuffer = new FixedBuffer(batchSize, replayBufferSize);
        this.actionSpace = buildActionSpace();
        reset();
    }

    @Override
    public void reset() {
        List<Token> board = new ArrayList<>(NUM_SQUARES);
        for (int i = 0; i < NUM_SQUARES; i++) {
            board.add(Token.NONE);
        }
        this.board = board;
        setCurPlayerId(0);
        this.turns = 0;
        this.done = false;
    }

    @Override
    public NDList getObservation() {
        return buildObservation();
    }

    @Override
    public ActionSpace getActionSpace() {
        return this.actionSpace;
    }

    @Override
    public Step step(NDList action, boolean training) {
        NDList preState = buildObservation();
        int actionData = action.singletonOrThrow().getInt();
        Validate.isTrue(actionData < this.board.size());
        boolean done;
        float[] reward;
        if (this.board.get(actionData) != Token.NONE) {
            // not empty
            done = true;
            reward = new float[]{1, 1};
            reward[getCurPlayerId()] = -1;
        } else {
            this.board.set(actionData, Token.getPlayerToken(getCurPlayerId()));
            this.turns++;
            Tuple<Integer, Boolean> result = checkGameOver();
            done = result.second;
            reward = new float[]{-result.first, -result.first};
            reward[getCurPlayerId()] = result.first;
        }
        if (!done) {
            int newPlayerId = (getCurPlayerId() + 1) % N_PLAYERS;
            setCurPlayerId(newPlayerId);
        }
        this.done = done;
        return new TicTacToeStep(manager.newSubManager(), this.actionSpace, preState, buildObservation(), action, reward, done);
    }

    @Override
    public Step[] getBatch() {
        return replayBuffer.getBatch();
    }

    @Override
    public void close() {
        manager.close();
    }

    @Override
    public void render() {
        logger.info("");
        if (this.done) {
            logger.info("GAME OVER");
        } else {
            logger.info("It is Player " + this.getCurPlayerId() + "'s turn to move");
        }
    }

    @Override
    public Model buildBaseModel() {
        Model policyModel = Model.newInstance("tictactoe-model");
        TicTacToePolicyBlock policyNet = new TicTacToePolicyBlock(manager, random, getActionSpace().size());
        policyModel.setBlock(policyNet);
        return policyModel;
    }

    @Override
    public Shape getObservationShape(int batchSize) {
        return new Shape(batchSize, 2, 3, 3);
    }

    /**
     * @return 构建并返回当前棋盘状态
     */
    private NDList buildObservation() {
        float[][] curPositions = new float[GRID_LENGTH][GRID_LENGTH];
        float[][] legalPositions = new float[GRID_LENGTH][GRID_LENGTH];
        for (int i = 0; i < this.board.size(); i++) {
            Token token = this.board.get(i);
            int h = i / GRID_LENGTH;
            int w = i % GRID_LENGTH;
            int num = getCurPlayerId() == 0 ? 0 : (getCurPlayerId() == token.getPlayerId() ? 1 : -1);
            curPositions[h][w] = num;
            if (token == Token.NONE) {
                legalPositions[h][w] = 1;
            } else {
                legalPositions[h][w] = 0;
            }
        }
        NDArray curPositionArr = manager.create(curPositions).expandDims(0);
        NDArray legalPositionArr = manager.create(legalPositions).expandDims(0);
        return new NDList(curPositionArr.concat(legalPositionArr, 0));
    }

    /**
     * @return 构建并返回井字棋的行为空间
     */
    private ActionSpace buildActionSpace() {
        ActionSpace actionSpace = new ActionSpace();
        for (int i = 0; i < NUM_SQUARES; i++) {
            actionSpace.add(new NDList(manager.create(i)));
        }
        return actionSpace;
    }

    /**
     * 判断棋局是否终了
     *
     * @return 二元组first-收益；二元组second-是否结束
     */
    public Tuple<Integer, Boolean> checkGameOver() {
        int curPlayerId = getCurPlayerId();
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
        NONE(".", -1),
        /**
         * 玩家1落子
         */
        X("X", 0),
        /**
         * 玩家2落子
         */
        O("O", 1),
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

    static final class TicTacToeStep implements RlEnv.Step {
        private NDManager manager;
        private ActionSpace actionSpace;
        private NDList preState;
        private NDList postState;
        private NDList action;
        private float[] reward;
        private boolean done;

        private TicTacToeStep(NDManager manager, ActionSpace actionSpace, NDList preState, NDList postState, NDList action, float[] reward, boolean done) {
            this.manager = manager;
            this.actionSpace = actionSpace;
            this.preState = preState;
            this.postState = postState;
            this.action = action;
            this.reward = reward;
            this.done = done;
        }

        @Override
        public NDList getPreObservation() {
            return preState;
        }

        @Override
        public NDList getAction() {
            return action;
        }

        @Override
        public NDList getPostObservation() {
            return postState;
        }

        @Override
        public ActionSpace getPostActionSpace() {
            return actionSpace;
        }

        @Override
        public NDArray getReward() {
            return manager.create(reward);
        }

        @Override
        public boolean isDone() {
            return done;
        }

        @Override
        public void close() {
            manager.close();
        }
    }
}
