package mo.boardgame.demo.gomoku.gui.controllers;
/**
 * Created by Doston Hamrakulov
 */

import mo.boardgame.demo.gomoku.gui.GomokuVisualizer;

/**
 * Interface for a controller. Provides the controller with access to the game.
 */
public interface IController {

    /**
     * Initialise the controller with a game instance.
     */
    void initialise(GomokuVisualizer visualizer);
}
