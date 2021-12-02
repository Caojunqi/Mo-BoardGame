package mo.boardgame.game;

import ai.djl.util.RandomUtils;
import env.action.core.IAction;
import env.state.core.IState;
import utils.datatype.Snapshot;

/**
 * “左右互搏”环境
 *
 * @author Caojunqi
 * @date 2021-11-25 11:21
 */
public class SelfPlayEnv<S extends IState<S>, A extends IAction> {
    /**
     * 游戏环境
     */
    private BaseBoardGameEnv<S, A> gameEnv;
    /**
     * 当前AI主角，使用的是正在优化的模型，其对手使用的是上一次优化完成的模型
     */
    private int agentPlayerId;
    /**
     * 所有参与游戏的AI主体，处于{@link SelfPlayEnv#agentPlayerId}位置的Agent为null。
     */
    private Agent[] agents;


    public SelfPlayEnv(BaseBoardGameEnv<S, A> gameEnv) {
        this.gameEnv = gameEnv;
    }

    /**
     * 环境重置
     *
     * @return 重置后，游戏环境的初始状态
     */
    public S reset() {
        S gameEnvState = gameEnv.reset();
        this.agents = new Agent[gameEnv.getPlayerNum()];
        this.agentPlayerId = RandomUtils.nextInt(gameEnv.getPlayerNum());
        setupOpponents();
        if (this.agentPlayerId != gameEnv.getCurPlayerId()) {
            continueGame();
        }
        return gameEnvState;
    }

    /**
     * 构建对手
     */
    public void setupOpponents() {

    }

    public Snapshot<S> step(A action) {
        Snapshot<S> snapshot = this.gameEnv.step(action);
        if (!snapshot.isDone()) {
            snapshot = continueGame();
        }
    }

    public Snapshot<S> continueGame() {

    }
}
