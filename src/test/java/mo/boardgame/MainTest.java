package mo.boardgame;

import java.io.IOException;
import java.net.URL;
import java.util.Enumeration;

/**
 * @author Caojunqi
 * @date 2021-11-26 10:15
 */
public class MainTest {

    public static void main(String[] args) {
        Enumeration<URL> urls;
        try {
            urls =
                    Thread.currentThread()
                            .getContextClassLoader()
                            .getResources("native/lib/pytorch.properties");
            System.out.println("cc");
        } catch (IOException e) {
            throw new IllegalStateException(e);
        }
    }

}
