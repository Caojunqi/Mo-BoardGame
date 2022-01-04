package mo.boardgame.demo.gomoku2.gui;

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
import mo.boardgame.demo.gomoku2.GomokuEnv2;
import mo.boardgame.game.FightRobotEnv;

import java.util.Random;

/**
 * 五子棋应用
 *
 * @author Caojunqi
 * @date 2021-12-16 15:14
 */
public class GomokuApplication2 extends Application {

    private final static String GOMOKU_GUI_RESOURCE_DIR = "demo/gomoku/";

    @Override
    public void start(Stage primaryStage) throws Exception {
        Font.loadFont(getClass().getClassLoader().getResource
                (GOMOKU_GUI_RESOURCE_DIR + "FontAwesome.otf").toExternalForm(), 10);

        Random random = new Random(0);
        NDManager mainManager = NDManager.newBaseManager();
        GomokuEnv2 gameEnv = new GomokuEnv2(mainManager.newSubManager(), random, true);
        FightRobotEnv<GomokuEnv2> fightRobotEnv = new FightRobotEnv<>(mainManager.newSubManager(), random, gameEnv);

        Pane gomokuPane = loadGomokuPane(mainManager, fightRobotEnv);

        primaryStage.setTitle("Gomoku");
        primaryStage.setScene(new Scene(gomokuPane, 800, 600));
        primaryStage.setMinWidth(800);
        primaryStage.setMinHeight(600);
        primaryStage.getIcons().add(new Image(getClass().getClassLoader()
                .getResource(GOMOKU_GUI_RESOURCE_DIR + "AppIcon.png").toExternalForm()));
        primaryStage.show();
    }

    private Pane loadGomokuPane(NDManager mainManager, FightRobotEnv<GomokuEnv2> fightRobotEnv) {
        BorderPane gomokuPane = new BorderPane();
        GomokuBoardPane2 boardPane = new GomokuBoardPane2(mainManager.newSubManager(), fightRobotEnv);
        gomokuPane.setCenter(boardPane);
        fightRobotEnv.getGameEnv().setBoardPane(boardPane);

        Button button = new Button("Start");
        Font font = Font.font("Consolas", FontWeight.BOLD, 13);
        button.setFont(font);
        button.setPrefWidth(100);
        button.setPrefHeight(100);
        button.setOnAction(value -> fightRobotEnv.start());
        gomokuPane.setLeft(button);
        return gomokuPane;
    }

}
