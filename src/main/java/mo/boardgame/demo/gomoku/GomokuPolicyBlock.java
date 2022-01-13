package mo.boardgame.demo.gomoku;

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
public class GomokuPolicyBlock extends AbstractBlock {

	private NDManager netManager;
	private Random random;

	private Block featureExtractorConv;
	private Block featureExtractorResidual;
	private Block policyHeadConv;
	private Block policyHeadDense;
	private Block valueHeadConv;
	private Block valueHeadVDense;
	private Block valueHeadQDense;

	public GomokuPolicyBlock(NDManager netManager, Random random) {
		this.netManager = netManager;
		this.random = random;

		this.featureExtractorConv = addChildBlock("feature_extractor_conv", buildFeatureExtractorConv());
		this.featureExtractorResidual = addChildBlock("feature_extractor_residual", buildFeatureExtractorResidual());
		this.policyHeadConv = addChildBlock("policy_head_conv", buildPolicyHeadConv());
		this.policyHeadDense = addChildBlock("policy_head_dense", buildPolicyHeadDense());
		this.valueHeadConv = addChildBlock("value_head_conv", buildValueHeadConv());
		this.valueHeadVDense = addChildBlock("value_head_v_dense", buildValueHeadVDense());
		this.valueHeadQDense = addChildBlock("value_head_q_dense", buildValueHeadQDense());
	}

	@Override
	protected NDList forwardInternal(ParameterStore parameterStore, NDList inputs, boolean training, PairList<String, Object> params) {
		NDArray obs = inputs.singletonOrThrow().get(new NDIndex(":,0,:,:")).expandDims(1);
		NDArray mask = inputs.singletonOrThrow().get(new NDIndex(":,1,:,:")).expandDims(1).toType(DataType.BOOLEAN, false);

		NDList featureConv = this.featureExtractorConv.forward(parameterStore, new NDList(obs), training);
		NDList featureResidual = this.featureExtractorResidual.forward(parameterStore, featureConv, training);
		NDArray tmp = featureConv.singletonOrThrow().add(featureResidual.singletonOrThrow());
		NDList features = Activation.relu(new NDList(tmp));

		NDList policyConv = this.policyHeadConv.forward(parameterStore, features, training);
		NDList policyFlatten = new NDList(policyConv.get(0).reshape(policyConv.get(0).getShape().get(0), -1));
		NDArray policy = this.policyHeadDense.forward(parameterStore, policyFlatten, training).singletonOrThrow();
		// Normalize
		policy = policy.sub(policy.exp().sum(new int[]{-1}, true).log());

		NDList valueConv = this.valueHeadConv.forward(parameterStore, features, training);
		NDList valueFlatten = new NDList(valueConv.get(0).reshape(valueConv.get(0).getShape().get(0), -1));
		NDList vf = this.valueHeadVDense.forward(parameterStore, valueFlatten, training);
		NDList qf = this.valueHeadQDense.forward(parameterStore, valueFlatten, training);

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

		this.featureExtractorConv.setInitializer(new XavierInitializer(), Parameter.Type.WEIGHT);
		this.featureExtractorConv.setInitializer(Initializer.ZEROS, Parameter.Type.BIAS);
		this.featureExtractorConv.initialize(manager, dataType, inputShapes[0]);

		this.featureExtractorResidual.setInitializer(new XavierInitializer(), Parameter.Type.WEIGHT);
		this.featureExtractorResidual.setInitializer(Initializer.ZEROS, Parameter.Type.BIAS);
		this.featureExtractorResidual.initialize(manager, dataType, new Shape(1, 32, 10, 10));

		this.policyHeadConv.setInitializer(new XavierInitializer(), Parameter.Type.WEIGHT);
		this.policyHeadConv.setInitializer(Initializer.ZEROS, Parameter.Type.BIAS);
		this.policyHeadConv.initialize(manager, dataType, new Shape(1, 32, 10, 10));

		this.policyHeadDense.setInitializer(new XavierInitializer(), Parameter.Type.WEIGHT);
		this.policyHeadDense.setInitializer(Initializer.ZEROS, Parameter.Type.BIAS);
		this.policyHeadDense.initialize(manager, dataType, new Shape(400));

		this.valueHeadConv.setInitializer(new XavierInitializer(), Parameter.Type.WEIGHT);
		this.valueHeadConv.setInitializer(Initializer.ZEROS, Parameter.Type.BIAS);
		this.valueHeadConv.initialize(manager, dataType, new Shape(1, 32, 10, 10));

		this.valueHeadVDense.setInitializer(new XavierInitializer(), Parameter.Type.WEIGHT);
		this.valueHeadVDense.setInitializer(Initializer.ZEROS, Parameter.Type.BIAS);
		this.valueHeadVDense.initialize(manager, dataType, new Shape(400));

		this.valueHeadQDense.setInitializer(new XavierInitializer(), Parameter.Type.WEIGHT);
		this.valueHeadQDense.setInitializer(Initializer.ZEROS, Parameter.Type.BIAS);
		this.valueHeadQDense.initialize(manager, dataType, new Shape(400));
	}

	private Block buildFeatureExtractorConv() {
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
				.add(Activation::relu);
	}

	private Block buildFeatureExtractorResidual() {
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
						.setFilters(32)
						.setKernelShape(new Shape(3, 3))
						.optStride(new Shape(1, 1))
						.optPadding(new Shape(1, 1))
						.build())
				.add(BatchNorm.builder()
						.optMomentum(0.9f)
						.build());
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
						.setFilters(4)
						.setKernelShape(new Shape(1, 1))
						.optStride(new Shape(1, 1))
						.build())
				.add(BatchNorm.builder()
						.optMomentum(0.9f)
						.build())
				.add(Activation::relu);
	}

	private Block buildValueHeadVDense() {
		return new SequentialBlock()
				.add(Linear.builder()
						.setUnits(1)
						.build())
				.add(Activation::tanh);
	}

	private Block buildValueHeadQDense() {
		return new SequentialBlock()
				.add(Linear.builder()
						.setUnits(100)
						.build())
				.add(Activation::tanh);
	}
}
