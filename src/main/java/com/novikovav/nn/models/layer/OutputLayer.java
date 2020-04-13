package com.novikovav.nn.models.layer;

import com.novikovav.nn.models.neuron.BiasNeuron;
import com.novikovav.nn.models.neuron.Neuron;
import com.novikovav.nn.models.neuron.OutputNeuron;
import com.novikovav.nn.models.synapse.Synapse;
import com.novikovav.nn.models.synapse.SynapseType;

import java.util.ArrayList;

public class OutputLayer extends Layer {
    private final Layer prevLayer;
    private final boolean withBias;
    private final float[][] weights;
    private final float[] biasWeights;

    public OutputLayer(int numberOfNeurons, Layer prevLayer, boolean withBias, float[][] weights, float[] biasWeights) {
        super(LayerType.OUTPUT, numberOfNeurons);
        this.prevLayer = prevLayer;
        this.withBias = withBias;
        this.weights = weights;
        this.biasWeights = biasWeights;
        this.build();
    }

    public float[] getResult() {
        float[] result = new float[this.getNumberOfNeurons()];
        for (int i = 0; i < this.getNumberOfNeurons(); i++) {
            result[i] = this.getNeurons()[i].getValue();
        }
        return result;
    }

    public Float getError(ArrayList<Float> input) {
        float result = 0.0f;
        Neuron[] outputLayerNeurons = this.getNeurons();
        for (int i = 0; i < outputLayerNeurons.length; i++) {
            Float inputValue = input.get(i);
            Neuron neuron = outputLayerNeurons[i];
            result += (float) Math.pow((inputValue - neuron.getValue()), 2);
        }
        result = result / outputLayerNeurons.length;

        return result;
    }

    public void learn(ArrayList<Float> input) {
        Neuron[] outputLayerNeurons = this.getNeurons();
        for (int i = 0; i < outputLayerNeurons.length; i++) {
            Float inputValue = input.get(i);
            OutputNeuron outputNeuron = (OutputNeuron) outputLayerNeurons[i];
            outputNeuron.updateDelta(inputValue);
            for (Synapse synapse : outputNeuron.getInputSynapses()) {
                synapse.updateGradient();
                synapse.updateWeight();
            }
        }
    }

    private void build() {
        BiasNeuron biasNeuron = new BiasNeuron();
        for (int i = 0; i < this.getNumberOfNeurons(); i++) {
            OutputNeuron outputNeuron = new OutputNeuron();
            float[] layerWeights = null;
            if (this.weights != null) {
                layerWeights = this.weights[i];
            }
            for (int n = 0; n < prevLayer.getNeurons().length; n++) {
                Neuron neuron = prevLayer.getNeurons()[n];
                Synapse synapse = new Synapse();
                if (this.weights != null && layerWeights != null && n < layerWeights.length) {
                    synapse.setWeight(layerWeights[n]);
                }
                outputNeuron.addSynapse(synapse, SynapseType.IN);
                neuron.addSynapse(synapse, SynapseType.OUT);
                synapse.setInputNeuron(neuron);
                synapse.setOutputNeuron(outputNeuron);
            }
            if (this.withBias) {
                Synapse biasSynapse = new Synapse();
                if (this.biasWeights != null && i < this.biasWeights.length) {
                    biasSynapse.setWeight(this.biasWeights[i]);
                }
                outputNeuron.addSynapse(biasSynapse, SynapseType.IN);
                biasNeuron.addSynapse(biasSynapse, SynapseType.OUT);
                biasSynapse.setInputNeuron(biasNeuron);
                biasSynapse.setOutputNeuron(outputNeuron);
            }

            this.addNeuron(outputNeuron);
        }
        if (this.withBias) {
            this.setBiasNeuron(biasNeuron);
        }
    }
}
