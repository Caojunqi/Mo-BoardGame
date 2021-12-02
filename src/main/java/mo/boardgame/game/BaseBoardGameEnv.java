package mo.boardgame.game;

import env.action.core.IAction;
import env.action.space.IActionSpace;
import env.common.Environment;
import env.state.core.IState;
import env.state.space.IStateSpace;

/**
 * 棋类游戏环境基类
 *
 * @author Caojunqi
 * @date 2021-11-26 16:14
 */
public abstract class BaseBoardGameEnv<S extends IState<S>, A extends IAction> extends Environment<S, A> {
    /**
     * 游戏名称
     */
    private String name;
    /**
     * 参与玩家数目
     */
    private int playerNum;
    /**
     * 当前行动玩家索引，取值范围为[0, playerNum-1]
     */
    private int curPlayerId;

    public BaseBoardGameEnv(String name, IStateSpace<S> stateSpace, IActionSpace<A> actionSpace, int playerNum) {
        super(stateSpace, actionSpace);
        this.playerNum = playerNum;
    }

    public int getPlayerNum() {
        return playerNum;
    }

    public int getCurPlayerId() {
        return curPlayerId;
    }

    public void setCurPlayerId(int curPlayerId) {
        this.curPlayerId = curPlayerId;
    }
}
