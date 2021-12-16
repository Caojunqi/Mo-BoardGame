package mo.boardgame.demo.gomoku.gui;

import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import javafx.event.EventHandler;
import javafx.geometry.VPos;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.input.MouseEvent;
import javafx.scene.layout.Pane;
import javafx.scene.paint.Color;
import javafx.scene.paint.CycleMethod;
import javafx.scene.paint.RadialGradient;
import javafx.scene.paint.Stop;
import javafx.scene.text.Font;
import javafx.scene.text.TextAlignment;
import mo.boardgame.demo.gomoku.GomokuEnv;
import mo.boardgame.game.FightRobotEnv;

import java.util.Map;

/**
 * 五子棋棋盘
 *
 * @author Caojunqi
 * @date 2021-12-16 15:44
 */
public class GomokuBoardPane extends Pane {
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
    private FightRobotEnv<GomokuEnv> fightRobotEnv;
    private Canvas canvas;
    // 绘图数据
    private int gridLength;
    private double paddingX;
    private double paddingY;
    private double cellSize;

    public GomokuBoardPane(NDManager manager, FightRobotEnv<GomokuEnv> fightRobotEnv) {
        this.manager = manager;
        this.fightRobotEnv = fightRobotEnv;
        this.gridLength = fightRobotEnv.getGameEnv().getGridLength();
        this.canvas = new Canvas();
        this.getChildren().add(canvas);
        widthProperty().addListener((observable, oldValue, newValue) ->
                canvas.setWidth(newValue.intValue()));
        heightProperty().addListener((observable, oldValue, newValue) ->
                canvas.setHeight(newValue.intValue()));
        EventHandler<MouseEvent> actionEvent = event -> {
            int row = getClosestRow(event.getY());
            int col = getClosestCol(event.getX());
            int actionData = row * gridLength + col;
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
        this.cellSize = smallestAxis / (gridLength - 1);

        // Add some padding around the grid, so we can display numbers/letters
        cellSize = (smallestAxis - (cellSize * 2)) / (gridLength - 1);
        double remainingSpaceX = getWidth() - (cellSize * (gridLength - 1));
        double remainingSpaceY = getHeight() - (cellSize * (gridLength - 1));

        this.paddingX = remainingSpaceX / 2;
        this.paddingY = remainingSpaceY / 2;

        drawGrid(gc, paddingX, paddingY, gridLength - 1, gridLength - 1, cellSize);
        drawNumbers(gc, paddingX, paddingY, gridLength - 1, gridLength - 1, cellSize);
        drawLetters(gc, paddingX, paddingY, gridLength - 1, gridLength - 1, cellSize);

        // 绘制棋盘落子信息
        Map<Integer, GomokuEnv.Token> chessInfo = this.fightRobotEnv.getGameEnv().getChessInfo();
        for (int w = 0; w < gridLength; w++) {
            for (int h = 0; h < gridLength; h++) {
                int index = h * gridLength + w;
                GomokuEnv.Token token = chessInfo.get(index);
                if (token == null) {
                    token = GomokuEnv.Token.NONE;
                }
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
     * Draw numbers along the X axis for a previously drawn grid. Each number
     * aligns with the corresponding intersection on the grid.
     *
     * @param gc       Graphics context
     * @param startX   Start point on x axis
     * @param startY   Start point on y axis
     * @param rows     Number of rows
     * @param columns  Number of columns
     * @param cellSize Size of each cell in the grid
     */
    private void drawNumbers(GraphicsContext gc, double startX, double
            startY, int rows, int columns, double cellSize) {
        gc.save();
        gc.setFont(BOARD_FONT);
        gc.setFill(Color.rgb(0, 0, 0, 0.75));
        double distance = 0.7 * cellSize;
        for (int i = 0; i < this.gridLength; i++) {
            double offset = i * cellSize;
            gc.setTextAlign(TextAlignment.CENTER);
            gc.setTextBaseline(VPos.CENTER);
            gc.fillText(Integer.toString(i + 1), startX - distance,
                    startY + offset);
        }
        gc.restore();
    }

    /**
     * Draw letters along the Y axis for a previously drawn grid. Each letter
     * aligns with the corresponding intersection on the grid.
     *
     * @param gc       Graphics context
     * @param startX   Start point on x axis
     * @param startY   Start point on y axis
     * @param rows     Number of rows
     * @param columns  Number of columns
     * @param cellSize Size of each cell in the grid
     */
    private void drawLetters(GraphicsContext gc, double startX, double
            startY, int rows, int columns, double cellSize) {
        gc.save();
        gc.setFont(BOARD_FONT);
        gc.setFill(Color.rgb(0, 0, 0, 0.75));
        double distance = 0.7 * cellSize;
        for (int i = 0; i < this.gridLength; i++) {
            double offset = i * cellSize;
            gc.setTextAlign(TextAlignment.CENTER);
            gc.setTextBaseline(VPos.CENTER);
            gc.fillText(Character.toString((char) ('A' + i)), startX + offset,
                    startY + cellSize * (rows) + distance);
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
    private void drawChess(GraphicsContext gc, double startX, double startY, double cellSize, int row, int col, GomokuEnv.Token token) {
        double x = startX + col * cellSize;
        double y = startY + row * cellSize;
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
        int closest = (int) Math.round((mouseY - paddingY) / cellSize);
        if (closest < 0) return 0;
        if (closest > gridLength - 1) return gridLength - 1;
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
        int closest = (int) Math.round((mouseX - paddingX) / cellSize);
        if (closest < 0) return 0;
        if (closest > gridLength - 1) return gridLength - 1;
        return closest;
    }

}
