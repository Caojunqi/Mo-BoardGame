package mo.boardgame;

/**
 * 测试启动
 *
 * @author Caojunqi
 * @date 2021-12-23 11:06
 */
public class TestStart {

    public static void main(String[] args) {
        BoardGameType gameType = BoardGameType.GOMOKU;

        gameType.launchRenderApplication(args);
    }
}
