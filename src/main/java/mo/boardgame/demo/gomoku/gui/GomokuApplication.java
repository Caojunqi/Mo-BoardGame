package mo.boardgame.demo.gomoku.gui;

import ai.djl.ndarray.NDManager;
import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.image.Image;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.Pane;
import javafx.scene.text.Font;
import javafx.stage.Stage;
import mo.boardgame.demo.gomoku.GomokuEnv;
import mo.boardgame.game.FightRobotEnv;

import java.util.Random;

/**
 * 五子棋应用
 *
 * @author Caojunqi
 * @date 2021-12-16 15:14
 */
public class GomokuApplication extends Application {

    private final static String GOMOKU_GUI_RESOURCE_DIR = "demo/gomoku/";

    @Override
    public void start(Stage primaryStage) throws Exception {
        Font.loadFont(getClass().getClassLoader().getResource
                (GOMOKU_GUI_RESOURCE_DIR + "FontAwesome.otf").toExternalForm(), 10);

        Random random = new Random(0);
        NDManager mainManager = NDManager.newBaseManager();
        GomokuEnv gameEnv = new GomokuEnv(mainManager.newSubManager(), random, true);
        FightRobotEnv<GomokuEnv> fightRobotEnv = new FightRobotEnv<>(mainManager.newSubManager(), random, gameEnv);

        Pane gomokuPane = loadGomokuPane(mainManager, fightRobotEnv);

        primaryStage.setTitle("Gomoku");
        primaryStage.setScene(new Scene(gomokuPane, 800, 600));
        primaryStage.setMinWidth(800);
        primaryStage.setMinHeight(600);
        primaryStage.getIcons().add(new Image(getClass().getClassLoader()
                .getResource(GOMOKU_GUI_RESOURCE_DIR + "AppIcon.png").toExternalForm()));
        primaryStage.show();
    }

    private Pane loadGomokuPane(NDManager mainManager, FightRobotEnv<GomokuEnv> fightRobotEnv) {
        BorderPane gomokuPane = new BorderPane();
        GomokuBoardPane boardPane = new GomokuBoardPane(mainManager.newSubManager(), fightRobotEnv);
        gomokuPane.setCenter(boardPane);
        fightRobotEnv.getGameEnv().setBoardPane(boardPane);

        Button start = new Button("Start");
        start.setOnAction(value -> fightRobotEnv.start());
        gomokuPane.setLeft(start);
        return gomokuPane;
    }

}
