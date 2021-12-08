package mo.boardgame.demo.gomoku.gui.controllers.impl;
/**
 * Created by Doston Hamrakulov
 */


import javafx.application.Platform;
import javafx.event.EventHandler;
import javafx.scene.input.MouseEvent;
import mo.boardgame.demo.gomoku.gui.BoardPane;
import mo.boardgame.demo.gomoku.gui.GomokuVisualizer;
import mo.boardgame.demo.gomoku.gui.controllers.IController;
import mo.boardgame.demo.gomoku.gui.events.GomokuEventAdapter;

public class BoardPaneController implements IController {

    private final BoardPane boardView;
    private EventHandler<MouseEvent> mouseListener;
    private GomokuVisualizer gomokuVisualizer;

    /**
     * Create a new BoardPaneController.
     *
     * @param view Board pane
     */
    public BoardPaneController(BoardPane view) {
        this.boardView = view;
    }

    @Override
    public void initialise(GomokuVisualizer gomokuVisualizer) {
        this.gomokuVisualizer = gomokuVisualizer;
        this.gomokuVisualizer.addListener(new GomokuEventAdapter() {
            EventHandler<MouseEvent> mouseListener;

            @Override
            public void gameStarted() {
                handleGameStarted();
            }

            @Override
            public void moveAdded(int playerIndex, int move) {
                handleMoveAdded(playerIndex, move);
            }

            @Override
            public void moveRemoved(int move) {
                handleMoveRemoved(move);
            }

            @Override
            public void gameFinished() {
                handleGameFinished();
            }

            @Override
            public void userMoveRequested(int playerIndex) {
                handleUserMoveRequested(playerIndex);
            }
        });
    }

    /**
     * Handle the userMoveRequested() event from the game.
     *
     * @param playerIndex Player index to retrieve move for
     */
    private void handleUserMoveRequested(int playerIndex) {
        // Enable the picker on the board to aid the user when picking a move
        Platform.runLater(() -> boardView.enableStonePicker(playerIndex));
        // Listener submits a move to the game, which can be declined if
        // invalid, if accepted the listener is removed and the picker is
        // disabled
        this.mouseListener = new EventHandler<MouseEvent>() {
            @Override
            public void handle(MouseEvent event) {
                int row = boardView.getClosestRow(event.getY());
                int col = boardView.getClosestCol(event.getX());
                if (gomokuVisualizer.setUserMove(row, col)) {
                    boardView.removeEventHandler(MouseEvent
                            .MOUSE_CLICKED, this);
                    Platform.runLater(boardView::disableStonePicker);
                }
            }
        };
        boardView.addEventHandler(MouseEvent.MOUSE_CLICKED,
                mouseListener);
    }

    /**
     * Handle the gameFinished() event from the game.
     */
    private void handleGameFinished() {
        if (mouseListener != null) {
            boardView.removeEventHandler(MouseEvent.MOUSE_CLICKED,
                    mouseListener);
        }
        Platform.runLater(() -> boardView.disableStonePicker());
    }

    /**
     * Handle the moveAdded() event from the game.
     *
     * @param playerIndex Player identifier
     * @param move        Move added to the state
     */
    private void handleMoveAdded(int playerIndex, int move) {
        int row = move / this.gomokuVisualizer.getGridLength();
        int col = move % this.gomokuVisualizer.getGridLength();
        Platform.runLater(() -> boardView.addStone(playerIndex, row,
                col, false));
    }

    /**
     * Handle the moveRemoved() event from the game.
     *
     * @param move Move removed from the state
     */
    private void handleMoveRemoved(int move) {
        int row = move / this.gomokuVisualizer.getGridLength();
        int col = move % this.gomokuVisualizer.getGridLength();
        Platform.runLater(() -> boardView.removeStone(row, col));
        Platform.runLater(() -> boardView.disableStonePicker());
    }

    /**
     * Handle the gameStarted() event from the game instance.
     */
    private void handleGameStarted() {
        // Clear the board in case of a previous game
        Platform.runLater(() -> boardView.clear());
    }

}
