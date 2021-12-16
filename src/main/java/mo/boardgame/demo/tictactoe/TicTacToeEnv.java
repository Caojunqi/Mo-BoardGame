package mo.boardgame.demo.tictactoe;

import ai.djl.Model;
import ai.djl.modality.rl.ActionSpace;
import ai.djl.modality.rl.env.RlEnv;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import common.Tuple;
import mo.boardgame.demo.tictactoe.gui.TicTacToeBoardPane;
import mo.boardgame.game.BaseBoardGameEnv;
import org.apache.commons.lang3.StringUtils;
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

    private final static Logger logger = LoggerFactory.getLogger(TicTacToeEnv.class);

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
     * 游戏是否结束
     */
    private boolean done;
    /**
     * 用于构建环境状态的管理器，可避免内存泄漏
     */
    private NDManager observationManager;
    private TicTacToeBoardPane boardPane;

    public TicTacToeEnv(NDManager manager, Random random, boolean verbose) {
        super(manager, random, GAME_NAME, N_PLAYERS, verbose);
        this.manager = manager;
        this.random = random;
        this.actionSpace = buildActionSpace();
        this.observationManager = this.manager.newSubManager();
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
        this.observationManager.close();
        this.observationManager = this.manager.newSubManager();
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
        int newPlayerId = (getCurPlayerId() + 1) % N_PLAYERS;
        setCurPlayerId(newPlayerId);
        this.done = done;
        render();
        return new TicTacToeStep(manager.newSubManager(), this.actionSpace, preState, buildObservation(), action, reward, done);
    }

    @Override
    public void render() {
        if (boardPane != null) {
            boardPane.requestLayout();
        }
    }

    @Override
    public Model buildBaseModel() {
        Model policyModel = Model.newInstance(getName());
        TicTacToePolicyBlock policyNet = new TicTacToePolicyBlock(policyModel.getNDManager(), random);
        policyModel.setBlock(policyNet);
        return policyModel;
    }

    @Override
    public Shape getObservationShape() {
        return new Shape(1, 1, 3, 3);
    }

    @Override
    public NDList parsePlayerAction(String actionStr) {
        if (StringUtils.isEmpty(actionStr)) {
            System.out.println("玩家行为不得为空！正确格式为：x");
            return null;
        }
        String[] strs = actionStr.split(" ");
        if (strs.length > 1) {
            System.out.println("玩家行为输入格式不正确！正确格式为：x");
            return null;
        }
        int actionData = Integer.parseInt(strs[0]);
        if (actionData >= NUM_SQUARES) {
            System.out.println("玩家行为输入格式不正确！格子数不得超过" + NUM_SQUARES);
            return null;
        }
        return new NDList(manager.create(actionData));
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
            int num = token == Token.NONE ? 0 : getCurPlayerId() == token.getPlayerId() ? 1 : -1;
            curPositions[h][w] = num;
            if (token == Token.NONE) {
                legalPositions[h][w] = 1;
            } else {
                legalPositions[h][w] = 0;
            }
        }
        NDArray curPositionArr = this.observationManager.create(curPositions).expandDims(0);
        NDArray legalPositionArr = this.observationManager.create(legalPositions).expandDims(0);
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
            if ((squareIsCurPlayer(i * GRID_LENGTH)
                    && squareIsCurPlayer(i * GRID_LENGTH + 1)
                    && squareIsCurPlayer(i * GRID_LENGTH + 2))
                    || (squareIsCurPlayer(i)
                    && squareIsCurPlayer(i + GRID_LENGTH)
                    && squareIsCurPlayer(i + GRID_LENGTH * 2))) {
                return new Tuple<>(1, true);
            }
        }

        // diagonals
        if ((squareIsCurPlayer(0)
                && squareIsCurPlayer(4)
                && squareIsCurPlayer(8))
                || (squareIsCurPlayer(6)
                && squareIsCurPlayer(4)
                && squareIsCurPlayer(2))) {
            return new Tuple<>(1, true);
        }

        if (this.turns == NUM_SQUARES) {
            return new Tuple<>(0, true);
        }
        return new Tuple<>(0, false);
    }

    /**
     * 判断棋盘上指定位置是否为当前玩家落子
     *
     * @param square 棋盘位置索引
     * @return true-square上是当前玩家的落子；false-square上不是当前玩家的落子。
     */
    private boolean squareIsCurPlayer(int square) {
        return this.board.get(square).playerId == getCurPlayerId();
    }

    public List<Token> getBoard() {
        return board;
    }

    public void setBoardPane(TicTacToeBoardPane boardPane) {
        this.boardPane = boardPane;
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
            this.preState.attach(this.manager);
            this.postState = postState;
            this.postState.attach(this.manager);
            this.action = action;
            this.action.attach(this.manager);
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
