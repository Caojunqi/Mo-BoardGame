package mo.boardgame.demo.gomoku;

import ai.djl.Model;
import ai.djl.modality.rl.ActionSpace;
import ai.djl.modality.rl.env.RlEnv;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import common.Tuple;
import mo.boardgame.game.BaseBoardGameEnv;
import org.apache.commons.lang3.StringUtils;
import org.apache.commons.lang3.Validate;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * 五子棋
 *
 * @author Caojunqi
 * @date 2021-12-07 22:29
 */
public class GomokuEnv extends BaseBoardGameEnv {

    private final static Logger logger = LoggerFactory.getLogger("chapters.introduction.HelloWorld1");

    /**
     * 游戏名称
     */
    private static final String GAME_NAME = "gomoku";
    /**
     * 棋盘长宽
     */
    private static final int GRID_LENGTH = 10;
    /**
     * X枚棋子连成一条直线后，游戏结束
     */
    private static final int N_IN_ROW = 5;
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

    public GomokuEnv(NDManager manager, Random random, boolean verbose) {
        super(manager, random, GAME_NAME, N_PLAYERS, verbose);
        this.manager = manager;
        this.random = random;
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
        return new GomokuStep(manager.newSubManager(), this.actionSpace, preState, buildObservation(), action, reward, done);
    }

    @Override
    public void render() {
        if (!isVerbose()) {
            return;
        }
        logger.info("");
        if (this.done) {
            logger.info("GAME OVER");
        } else {
            logger.info("It is Player " + this.getCurPlayerId() + "'s turn to move");
        }
        for (int i = 0; i < this.board.size(); i++) {
            Token token = this.board.get(i);
            System.out.print(token.symbol + " ");
            if ((i + 1) % GRID_LENGTH == 0) {
                System.out.print("\n");
            }
        }
    }

    @Override
    public Model buildBaseModel() {
        Model policyModel = Model.newInstance(getName());
        GomokuPolicyBlock policyNet = new GomokuPolicyBlock(manager.newSubManager(), random);
        policyModel.setBlock(policyNet);
        return policyModel;
    }

    @Override
    public Shape getObservationShape(int batchSize) {
        return new Shape(batchSize, 2, GRID_LENGTH, GRID_LENGTH);
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
        if (this.turns < N_IN_ROW * 2) {
            // 行动步数太少，不可能有人获胜
            return new Tuple<>(0, false);
        }

        for (int i = 0; i < NUM_SQUARES; i++) {
            Token token = this.board.get(i);
            if (token == Token.NONE) {
                continue;
            }
            if (token.playerId != curPlayerId) {
                continue;
            }
            int h = i / GRID_LENGTH;
            int w = i % GRID_LENGTH;

            // 水平检测
            if (w < GRID_LENGTH - N_IN_ROW + 1) {
                boolean finish = true;
                for (int j = w; j < w + N_IN_ROW; j++) {
                    if (squareIsCurPlayer(j)) {
                        finish = false;
                        break;
                    }
                }
                if (finish) {
                    return new Tuple<>(1, true);
                }
            }

            // 垂直检测
            if (h < GRID_LENGTH - N_IN_ROW + 1) {
                boolean finish = true;
                for (int j = h; j < h + N_IN_ROW * GRID_LENGTH; j += GRID_LENGTH) {
                    if (squareIsCurPlayer(j)) {
                        finish = false;
                        break;
                    }
                }
                if (finish) {
                    return new Tuple<>(1, true);
                }
            }

            // 左上向右下检测
            if (w < GRID_LENGTH - N_IN_ROW + 1 && h < GRID_LENGTH - N_IN_ROW + 1) {
                boolean finish = true;
                for (int j = w; j < w + (N_IN_ROW + 1); j += (GRID_LENGTH + 1)) {
                    if (squareIsCurPlayer(j)) {
                        finish = false;
                        break;
                    }
                }
                if (finish) {
                    return new Tuple<>(1, true);
                }
            }

            // 右上向左下检测
            if (w > N_IN_ROW - 1 && h < GRID_LENGTH - N_IN_ROW + 1) {
                boolean finish = true;
                for (int j = w; j < w + (N_IN_ROW - 1); j += (GRID_LENGTH - 1)) {
                    if (squareIsCurPlayer(j)) {
                        finish = false;
                        break;
                    }
                }
                if (finish) {
                    return new Tuple<>(1, true);
                }
            }
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

    static final class GomokuStep implements RlEnv.Step {
        private NDManager manager;
        private ActionSpace actionSpace;
        private NDList preState;
        private NDList postState;
        private NDList action;
        private float[] reward;
        private boolean done;

        private GomokuStep(NDManager manager, ActionSpace actionSpace, NDList preState, NDList postState, NDList action, float[] reward, boolean done) {
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
