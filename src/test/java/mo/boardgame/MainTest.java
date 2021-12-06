package mo.boardgame;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;

/**
 * @author Caojunqi
 * @date 2021-11-26 10:15
 */
public class MainTest {

    public static void main(String[] args) {
        NDManager manager = NDManager.newBaseManager();
        NDManager manager1 = NDManager.newBaseManager();
        float[][] data1 = new float[][]{
                {0.1f, 0.2f, 0.3f},
                {0.2f, 0.3f, 0.4f},
                {0.3f, 0.4f, 0.5f}
        };

        float[][] data2 = new float[][]{
                {0.7f, 0.8f, 0.1f},
                {0.8f, 0.9f, 0.1f},
                {0.8f, 0.1f, 0.2f}
        };

        NDArray arr1 = manager.create(data1).expandDims(0);
        NDArray arr2 = manager.create(data2).expandDims(0);

        arr1.attach(manager1);
        arr1.close();
        manager.close();
        System.out.println(arr1);
        System.out.println(arr2);
    }

}
