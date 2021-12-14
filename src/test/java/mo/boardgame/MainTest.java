package mo.boardgame;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;

/**
 * @author Caojunqi
 * @date 2021-11-26 10:15
 */
public class MainTest {

    public static void main(String[] args) {
        NDManager manager = NDManager.newBaseManager();
        float[][] data1 = new float[][]{
                {0.1f, 0.2f, 0.3f},
                {0.2f, 0.3f, 0.4f},
                {0.3f, 0.4f, 0.5f}
        };

        float[][] data2 = new float[][]{
                {0f, 1f, 1},
                {1f, 0f, 1},
                {0f, 0f, 0}
        };

        NDArray arr1 = manager.create(data1).expandDims(0);
        NDArray arr2 = manager.create(data2).expandDims(0);

        NDArray totalArr = arr1.concat(arr2, 0);
        totalArr = totalArr.expandDims(0);
        NDArray arr3 = totalArr.get(new NDIndex(":,0,:,:"));
        NDArray arr4 = totalArr.get(new NDIndex(":,1,:,:"));

        NDArray where = NDArrays.where(arr2.toType(DataType.BOOLEAN, false), arr1, manager.create(-1e8f));

        arr1.close();
        manager.close();
        System.out.println(arr1);
        System.out.println(arr2);
    }

}
