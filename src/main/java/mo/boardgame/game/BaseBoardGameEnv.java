package mo.boardgame.game;

import ai.djl.Model;
import ai.djl.modality.rl.ActionSpace;
import ai.djl.modality.rl.env.RlEnv;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;

/**
 * 棋类游戏环境基类
 *
 * @author Caojunqi
 * @date 2021-11-26 16:14
 */
public abstract class BaseBoardGameEnv implements RlEnv, IEnvRender {
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

    public BaseBoardGameEnv(String name, int playerNum) {
        this.playerNum = playerNum;
    }

    /**
     * 注：此接口重写的目的是为了把Access权限由package扩展到public
     */
    @Override
    public abstract NDList getObservation();

    /**
     * 注：此接口重写的目的是为了把Access权限由package扩展到public
     */
    @Override
    public abstract ActionSpace getActionSpace();

    /**
     * 每个游戏构建自己的Actor-Critic模型，以供PPO算法使用
     *
     * @return 拥有随机参数的模型
     */
    public abstract Model buildBaseModel();

    public abstract Shape getObservationShape(int batchSize);

    public String getName() {
        return name;
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
