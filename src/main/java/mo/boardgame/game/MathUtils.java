package mo.boardgame.game;

import java.util.Random;

/**
 * 数学相关工具类
 *
 * @author Caojunqi
 * @date 2021-12-11 12:02
 */
public final class MathUtils {

	/**
	 * 按照排名，概率性地随机选择
	 *
	 * @param random    随机数生成器
	 * @param totalSize 候选项总数
	 * @return 被选中的索引
	 */
	public static int rankRandomIndex(Random random, int totalSize) {
		int totalWeight = 0;
		int[] weights = new int[totalSize];
		for (int i = 0; i < totalSize; i++) {
			totalWeight += i + 1;
			weights[i] = i + 1;
		}
		int r = random.nextInt(totalWeight) + 1;
		int curCount = 0;
		for (int j = 0; j < totalSize; j++) {
			curCount += weights[j];
			if (r <= curCount) {
				return j;
			}
		}
		throw new IllegalStateException("rankRandomIndex 计算错误！！ totalSize:" + totalSize + "，r:" + r);
	}
}
