package mo.boardgame.demo.gomoku.gui.events;
/**
 * Created by Doston Hamrakulov
 */

/**
 * Listener interface for receiving interesting game events.
 */
public interface IGomokuListener {

    /**
     * Called when a player makes a move in the game.
     *
     * @param playerIndex Player identifier
     * @param move        Move made
     */
    void moveAdded(int playerIndex, int move);

    /**
     * Called when a move is undone.
     */
    void moveRemoved(int move);

    /**
     * Called when the game time changes for a player.
     *
     * @param playerIndex Player identifier
     * @param timeMillis  New game time in milliseconds
     */
    void gameTimeChanged(int playerIndex, long timeMillis);

    /**
     * Called when the game time changes for a player.
     *
     * @param playerIndex Player identifier
     * @param timeMillis  New move time in milliseconds
     */
    void moveTimeChanged(int playerIndex, long timeMillis);

    /**
     * Called when a players turn has started.
     *
     * @param playerIndex Player identifier
     */
    void turnStarted(int playerIndex);

    /**
     * Called when the game has started.
     */
    void gameStarted();

    /**
     * Called when a previous game is resumed.
     */
    void gameResumed();

    /**
     * Called when the game has finished.
     */
    void gameFinished();

    /**
     * Called when the game requests a move from the user.
     */
    void userMoveRequested(int playerIndex);

}
