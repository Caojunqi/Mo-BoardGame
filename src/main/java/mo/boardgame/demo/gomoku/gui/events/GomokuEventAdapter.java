package mo.boardgame.demo.gomoku.gui.events;
/**
 * Created by Doston Hamrakulov
 */

/**
 * Convenience class to allow components to only implement handler methods
 * for game events they are interested in.
 *
 * @see IGomokuListener
 */
public class GomokuEventAdapter implements IGomokuListener {

    @Override
    public void moveAdded(int playerIndex, int move) {
    }

    @Override
    public void moveRemoved(int move) {
    }

    @Override
    public void gameTimeChanged(int playerIndex, long timeMillis) {
    }

    @Override
    public void moveTimeChanged(int playerIndex, long timeMillis) {
    }

    @Override
    public void turnStarted(int playerIndex) {
    }

    @Override
    public void userMoveRequested(int playerIndex) {
    }

    @Override
    public void gameStarted() {
    }

    @Override
    public void gameResumed() {
    }

    @Override
    public void gameFinished() {
    }
}
