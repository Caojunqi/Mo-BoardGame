package mo.boardgame.demo.gomoku.gui;

import javafx.scene.layout.Pane;
import mo.boardgame.demo.gomoku.GomokuEnv;
import mo.boardgame.demo.gomoku.gui.controllers.IController;
import mo.boardgame.demo.gomoku.gui.controllers.impl.BoardPaneController;
import mo.boardgame.demo.gomoku.gui.events.IGomokuListener;

import java.util.ArrayList;
import java.util.List;

/**
 * 五子棋渲染类
 *
 * @author Caojunqi
 * @date 2021-12-08 17:06
 */
public class GomokuVisualizer {
    private final List<IGomokuListener> listeners;
    /**
     * 棋盘宽度
     */
    private int gridLength;

    public GomokuVisualizer(int gridLength) {
        this.gridLength = gridLength;
        this.listeners = new ArrayList<>();
    }

    public void addListener(IGomokuListener listener) {
        this.listeners.add(listener);
    }

    public boolean setUserMove(int row, int col) {
        return false;
    }

    public void paint(List<GomokuEnv.Token> board) {
        Pane boardPane = loadBoardPane(board);
    }

    private Pane loadBoardPane(List<GomokuEnv.Token> board) {
        BoardPane boardPane = new BoardPane(15);
        IController controller = new BoardPaneController(boardPane);
        controller.initialise(this);
        return boardPane;
    }

    public int getGridLength() {
        return gridLength;
    }
}
