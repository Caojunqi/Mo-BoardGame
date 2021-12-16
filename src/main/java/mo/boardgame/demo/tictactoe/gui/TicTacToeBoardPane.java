package mo.boardgame.demo.tictactoe.gui;

import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import javafx.event.EventHandler;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.input.MouseEvent;
import javafx.scene.layout.Pane;
import javafx.scene.paint.Color;
import javafx.scene.paint.CycleMethod;
import javafx.scene.paint.RadialGradient;
import javafx.scene.paint.Stop;
import javafx.scene.text.Font;
import mo.boardgame.demo.tictactoe.TicTacToeEnv;
import mo.boardgame.game.FightRobotEnv;

import java.util.List;

/**
 * 井字棋棋盘
 *
 * @author Caojunqi
 * @date 2021-12-16 15:44
 */
public class TicTacToeBoardPane extends Pane {
    private static int GRID_LENGTH = 3;
    private static Font BOARD_FONT = new Font("Arial", 11);
    private static RadialGradient whiteGradient = new RadialGradient(55,
            0.75, 0.5, 0.5, 0.5, true,
            CycleMethod.NO_CYCLE,
            new Stop(0, Color.WHITE),
            new Stop(1, Color.web("#A0A0A0"))
    );
    private static RadialGradient blackGradient = new RadialGradient(55,
            0.75, 0.5, 0.5, 0.5, true,
            CycleMethod.NO_CYCLE,
            new Stop(1, Color.web("#222")),
            new Stop(0, Color.web("#A0A0A0"))
    );
    private NDManager manager;
    private FightRobotEnv<TicTacToeEnv> fightRobotEnv;
    private Canvas canvas;
    // 绘图数
    private double paddingX;
    private double paddingY;
    private double cellSize;

    public TicTacToeBoardPane(NDManager manager, FightRobotEnv<TicTacToeEnv> fightRobotEnv) {
        this.manager = manager;
        this.fightRobotEnv = fightRobotEnv;
        this.canvas = new Canvas();
        this.getChildren().add(canvas);
        widthProperty().addListener((observable, oldValue, newValue) ->
                canvas.setWidth(newValue.intValue()));
        heightProperty().addListener((observable, oldValue, newValue) ->
                canvas.setHeight(newValue.intValue()));
        EventHandler<MouseEvent> actionEvent = event -> {
            int row = getClosestRow(event.getY());
            int col = getClosestCol(event.getX());
            int actionData = row * GRID_LENGTH + col;
            NDList action = new NDList(manager.create(actionData));
            fightRobotEnv.playerAction(action);
        };
        addEventFilter(MouseEvent.MOUSE_CLICKED, actionEvent);
    }

    @Override
    protected void layoutChildren() {
        super.layoutChildren();

        GraphicsContext gc = canvas.getGraphicsContext2D();
        gc.clearRect(0, 0, getWidth(), getHeight());

        double smallestAxis = Math.min(getHeight(), getWidth());
        this.cellSize = smallestAxis / GRID_LENGTH;

        // Add some padding around the grid, so we can display numbers/letters
        cellSize = (smallestAxis - cellSize) / GRID_LENGTH;
        double remainingSpaceX = getWidth() - (cellSize * GRID_LENGTH);
        double remainingSpaceY = getHeight() - (cellSize * GRID_LENGTH);

        this.paddingX = remainingSpaceX / 2;
        this.paddingY = remainingSpaceY / 2;

        drawGrid(gc, paddingX, paddingY, GRID_LENGTH, GRID_LENGTH, cellSize);

        // 绘制棋盘落子信息
        List<TicTacToeEnv.Token> chessInfo = this.fightRobotEnv.getGameEnv().getBoard();
        for (int w = 0; w < GRID_LENGTH; w++) {
            for (int h = 0; h < GRID_LENGTH; h++) {
                int index = h * GRID_LENGTH + w;
                TicTacToeEnv.Token token = chessInfo.get(index);
                drawChess(gc, paddingX, paddingY, cellSize, h, w, token);
            }
        }

    }

    /**
     * Draw a grid on some given graphics context.
     *
     * @param gc       Graphics context
     * @param startX   Start point on x axis
     * @param startY   Start point on y axis
     * @param rows     Number of rows
     * @param columns  Number of columns
     * @param cellSize Size of each cell in the grid
     */
    private void drawGrid(GraphicsContext gc, double startX, double startY, int
            rows, int columns, double cellSize) {
        gc.save();
        gc.setStroke(Color.rgb(0, 0, 0, 0.5));
        gc.setLineWidth(1.2);

        for (int i = 0; i <= columns; i++) {
            double offset = i * cellSize;
            gc.strokeLine(startX + offset, startY, startX + offset,
                    startY + cellSize * rows);
        }
        for (int i = 0; i <= rows; i++) {
            double offset = i * cellSize;
            gc.strokeLine(startX, startY + offset, startX +
                    cellSize * columns, startY + offset);
        }
        gc.restore();
    }

    /**
     * Paint a black/white stone onto a graphics context with a grid
     *
     * @param gc       Graphics context
     * @param startX   Start (top left) x coordinate of the grid
     * @param startY   Start (top left) y coordinate of the grid
     * @param cellSize Size of the grid cells
     * @param row      Row position of the stone
     * @param col      Column position of the stone
     * @param token    chess token
     */
    private void drawChess(GraphicsContext gc, double startX, double startY, double cellSize, int row, int col, TicTacToeEnv.Token token) {
        double x = startX + col * cellSize + 0.5 * cellSize;
        double y = startY + row * cellSize + 0.5 * cellSize;
        double offset = (cellSize * 0.7) / 2;
        gc.save();
        switch (token.getPlayerId()) {
            case 0:
                gc.setFill(blackGradient);
                gc.fillOval(x - offset, y - offset, cellSize * 0.7,
                        cellSize * 0.7);
                break;
            case 1:
                gc.setFill(whiteGradient);
                gc.fillOval(x - offset, y - offset, cellSize * 0.7,
                        cellSize * 0.7);
                break;
        }
        gc.restore();
    }

    /**
     * Given a mouse coordinate y axis value, return the closest row (0-n) on
     * the board
     *
     * @param mouseY Mouse y axis position
     * @return
     */
    private int getClosestRow(double mouseY) {
        int closest = (int) ((mouseY - paddingY) / cellSize);
        if (closest < 0) return 0;
        if (closest > GRID_LENGTH - 1) return GRID_LENGTH - 1;
        return closest;
    }

    /**
     * Given a mouse coordinate x axis value, return the closest column (0-n) on
     * the board
     *
     * @param mouseX Mouse x axis position
     * @return
     */
    private int getClosestCol(double mouseX) {
        int closest = (int) ((mouseX - paddingX) / cellSize);
        if (closest < 0) return 0;
        if (closest > GRID_LENGTH - 1) return GRID_LENGTH - 1;
        return closest;
    }

}
