package mo.boardgame.demo.gomoku2;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.*;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.BatchNorm;
import ai.djl.training.ParameterStore;
import ai.djl.training.initializer.Initializer;
import ai.djl.training.initializer.XavierInitializer;
import ai.djl.util.PairList;
import utils.ActionSampler;

import java.util.Random;

/**
 * @author Caojunqi
 * @date 2021-12-07 22:29
 */
public class GomokuPolicyBlock2 extends AbstractBlock {

	private NDManager netManager;
	private Random random;

	private Block commonLayers;
	private Block policyHeadConv;
	private Block policyHeadDense;
	private Block valueHeadConv;
	private Block valueHeadDense;

	public GomokuPolicyBlock2(NDManager netManager, Random random) {
		this.netManager = netManager;
		this.random = random;

		this.commonLayers = addChildBlock("common_layers", buildCommonLayers());
		this.policyHeadConv = addChildBlock("policy_head_conv", buildPolicyHeadConv());
		this.policyHeadDense = addChildBlock("policy_head_dense", buildPolicyHeadDense());
		this.valueHeadConv = addChildBlock("value_head_conv", buildValueHeadConv());
		this.valueHeadDense = addChildBlock("value_head_dense", buildValueHeadDense());
	}

	@Override
	protected NDList forwardInternal(ParameterStore parameterStore, NDList inputs, boolean training, PairList<String, Object> params) {
		NDArray obs = inputs.singletonOrThrow().get(new NDIndex(":,0:4,:,:"));
		NDArray mask = inputs.singletonOrThrow().get(new NDIndex(":,4,:,:")).toType(DataType.BOOLEAN, false);

		NDList x = this.commonLayers.forward(parameterStore, new NDList(obs), training);

		NDList policyConv = this.policyHeadConv.forward(parameterStore, x, training);
		NDList policyFlatten = new NDList(policyConv.get(0).reshape(policyConv.get(0).getShape().get(0), -1));
		NDArray policy = this.policyHeadDense.forward(parameterStore, policyFlatten, training).singletonOrThrow();
		// Normalize
		policy = policy.sub(policy.exp().sum(new int[]{-1}, true).log());

		NDList valueConv = this.valueHeadConv.forward(parameterStore, x, training);
		NDList valueFlatten = new NDList(valueConv.get(0).reshape(valueConv.get(0).getShape().get(0), -1));
		NDList vf = this.valueHeadDense.forward(parameterStore, valueFlatten, training);

		NDArray maskFlatten = mask.reshape(mask.getShape().get(0), -1);
		NDArray maskPolicy = NDArrays.where(maskFlatten, policy, policy.getManager().create(-1e8f));
		maskPolicy = maskPolicy.sub(maskPolicy.exp().sum(new int[]{-1}, true).log());
		NDArray actionProb = maskPolicy.softmax(-1);
		NDArray actions;
		if (training) {
			actions = ActionSampler.sampleMultinomial(netManager, actionProb, random);
		} else {
			actions = actionProb.argMax().toType(DataType.INT32, false);
			actions.attach(netManager);
		}
		NDArray entropy = maskPolicy.mul(actionProb).sum(new int[]{-1}).neg();
		return new NDList(actions, vf.singletonOrThrow(), maskPolicy, entropy);
	}

	@Override
	public Shape[] getOutputShapes(Shape[] inputShapes) {
		return new Shape[]{
				new Shape(1), new Shape(1), new Shape(100)
		};
	}

	@Override
	public void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {

		this.commonLayers.setInitializer(new XavierInitializer(), Parameter.Type.WEIGHT);
		this.commonLayers.setInitializer(Initializer.ZEROS, Parameter.Type.BIAS);
		this.commonLayers.initialize(manager, dataType, inputShapes[0]);

		this.policyHeadConv.setInitializer(new XavierInitializer(), Parameter.Type.WEIGHT);
		this.policyHeadConv.setInitializer(Initializer.ZEROS, Parameter.Type.BIAS);
		this.policyHeadConv.initialize(manager, dataType, new Shape(1, 128, 10, 10));

		this.policyHeadDense.setInitializer(new XavierInitializer(), Parameter.Type.WEIGHT);
		this.policyHeadDense.setInitializer(Initializer.ZEROS, Parameter.Type.BIAS);
		this.policyHeadDense.initialize(manager, dataType, new Shape(400));

		this.valueHeadConv.setInitializer(new XavierInitializer(), Parameter.Type.WEIGHT);
		this.valueHeadConv.setInitializer(Initializer.ZEROS, Parameter.Type.BIAS);
		this.valueHeadConv.initialize(manager, dataType, new Shape(1, 128, 10, 10));

		this.valueHeadDense.setInitializer(new XavierInitializer(), Parameter.Type.WEIGHT);
		this.valueHeadDense.setInitializer(Initializer.ZEROS, Parameter.Type.BIAS);
		this.valueHeadDense.initialize(manager, dataType, new Shape(200));
	}

	public Block buildCommonLayers() {
		return new SequentialBlock()
				.add(Conv2d.builder()
						.setFilters(32)
						.setKernelShape(new Shape(3, 3))
						.optStride(new Shape(1, 1))
						.optPadding(new Shape(1, 1))
						.build())
				.add(BatchNorm.builder()
						.optMomentum(0.9f)
						.build())
				.add(Activation::relu)
				.add(Conv2d.builder()
						.setFilters(64)
						.setKernelShape(new Shape(3, 3))
						.optStride(new Shape(1, 1))
						.optPadding(new Shape(1, 1))
						.build())
				.add(BatchNorm.builder()
						.optMomentum(0.9f)
						.build())
				.add(Activation::relu)
				.add(Conv2d.builder()
						.setFilters(128)
						.setKernelShape(new Shape(3, 3))
						.optStride(new Shape(1, 1))
						.optPadding(new Shape(1, 1))
						.build())
				.add(BatchNorm.builder()
						.optMomentum(0.9f)
						.build())
				.add(Activation::relu);
	}

	private Block buildPolicyHeadConv() {
		return new SequentialBlock()
				.add(Conv2d.builder()
						.setFilters(4)
						.setKernelShape(new Shape(1, 1))
						.optStride(new Shape(1, 1))
						.build())
				.add(BatchNorm.builder()
						.optMomentum(0.9f)
						.build())
				.add(Activation::relu);
	}

	private Block buildPolicyHeadDense() {
		return Linear.builder()
				.setUnits(100)
				.build();
	}

	private Block buildValueHeadConv() {
		return new SequentialBlock()
				.add(Conv2d.builder()
						.setFilters(2)
						.setKernelShape(new Shape(1, 1))
						.optStride(new Shape(1, 1))
						.build())
				.add(BatchNorm.builder()
						.optMomentum(0.9f)
						.build())
				.add(Activation::relu);
	}

	private Block buildValueHeadDense() {
		return new SequentialBlock()
				.add(Linear.builder()
						.setUnits(64)
						.build())
				.add(Activation::relu)
				.add(Linear.builder()
						.setUnits(1)
						.build())
				.add(Activation::tanh);
	}
}
