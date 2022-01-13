package mo.boardgame.demo.tictactoe.gui;

import ai.djl.ndarray.NDManager;
import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.image.Image;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.Pane;
import javafx.scene.text.Font;
import javafx.scene.text.FontWeight;
import javafx.stage.Stage;
import mo.boardgame.demo.tictactoe.TicTacToeEnv;
import mo.boardgame.game.FightRobotEnv;

import java.util.Random;

/**
 * 井字棋应用
 *
 * @author Caojunqi
 * @date 2021-12-16 20:50
 */
public class TicTacToeApplication extends Application {

	private final static String TICTACTOE_GUI_RESOURCE_DIR = "demo/tictactoe/";

	@Override
	public void start(Stage primaryStage) throws Exception {
		Font.loadFont(getClass().getClassLoader().getResource
				(TICTACTOE_GUI_RESOURCE_DIR + "FontAwesome.otf").toExternalForm(), 10);

		Random random = new Random(0);
		NDManager mainManager = NDManager.newBaseManager();
		TicTacToeEnv gameEnv = new TicTacToeEnv(mainManager.newSubManager(), random, true);
		FightRobotEnv<TicTacToeEnv> fightRobotEnv = new FightRobotEnv<>(mainManager.newSubManager(), random, gameEnv);

		Pane tictactoePane = loadTictactoePane(mainManager, fightRobotEnv);

		primaryStage.setTitle("TicTacToe");
		primaryStage.setScene(new Scene(tictactoePane, 800, 600));
		primaryStage.setMinWidth(800);
		primaryStage.setMinHeight(600);
		primaryStage.getIcons().add(new Image(getClass().getClassLoader()
				.getResource(TICTACTOE_GUI_RESOURCE_DIR + "AppIcon.png").toExternalForm()));
		primaryStage.show();
	}

	private Pane loadTictactoePane(NDManager mainManager, FightRobotEnv<TicTacToeEnv> fightRobotEnv) {
		BorderPane tictactoePane = new BorderPane();
		TicTacToeBoardPane boardPane = new TicTacToeBoardPane(mainManager.newSubManager(), fightRobotEnv);
		tictactoePane.setCenter(boardPane);
		fightRobotEnv.getGameEnv().setBoardPane(boardPane);

		Button button = new Button("Game Start");
		Font font = Font.font("Consolas", FontWeight.BOLD, 13);
		button.setFont(font);
		button.setPrefWidth(100);
		button.setPrefHeight(100);
		button.setOnAction(value -> fightRobotEnv.start());
		tictactoePane.setLeft(button);
		return tictactoePane;
	}
}
