package mo.boardgame;

import ai.djl.ndarray.NDManager;
import javafx.application.Application;
import mo.boardgame.demo.gomoku.GomokuEnv;
import mo.boardgame.demo.gomoku.gui.GomokuApplication;
import mo.boardgame.demo.tictactoe.TicTacToeEnv;
import mo.boardgame.demo.tictactoe.gui.TicTacToeApplication;
import mo.boardgame.game.BaseBoardGameEnv;

import java.lang.reflect.Constructor;
import java.util.Random;

/**
 * 棋类游戏类型
 *
 * @author Caojunqi
 * @date 2021-12-23 10:55
 */
public enum BoardGameType {

    /**
     * 五子棋
     */
    GOMOKU(GomokuEnv.class, GomokuApplication.class),
    /**
     * 井字棋
     */
    TIC_TAC_TOE(TicTacToeEnv.class, TicTacToeApplication.class),
    ;

    /**
     * 游戏环境构建类
     */
    private Class<? extends BaseBoardGameEnv> gameEnvClz;
    /**
     * 游戏环境渲染类
     */
    private Class<? extends Application> renderApplicationClz;

    BoardGameType(Class<? extends BaseBoardGameEnv> gameEnvClz, Class<? extends Application> renderApplicationClz) {
        this.gameEnvClz = gameEnvClz;
        this.renderApplicationClz = renderApplicationClz;
    }

    /**
     * 构建棋类游戏环境
     *
     * @param manager 矩阵资源管理类
     * @param random  随机数生成器
     * @param verbose 是否渲染游戏环境状态
     * @return 棋类游戏环境
     */
    public BaseBoardGameEnv buildBoardGameEnv(NDManager manager, Random random, boolean verbose) {
        try {
            Constructor<? extends BaseBoardGameEnv> gameEnvConstructor = this.gameEnvClz.getConstructor(NDManager.class, Random.class, boolean.class);
            return gameEnvConstructor.newInstance(manager, random, verbose);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * 启动棋类游戏界面，测试AI性能时使用
     */
    public void launchRenderApplication(String[] args) {
        Application.launch(this.renderApplicationClz, args);
    }
}
