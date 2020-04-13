package com.novikovav.nn.models.layer;

import com.novikovav.nn.models.network.Network;
import com.novikovav.nn.models.neuron.BiasNeuron;
import com.novikovav.nn.models.neuron.Neuron;
import com.novikovav.nn.models.synapse.Synapse;

import java.util.ArrayList;

public class Layer {
    private final LayerType layerType;
    private final int numberOfNeurons;
    private final Neuron[] neurons;
    private int neuronsIndex = 0;
    private BiasNeuron biasNeuron;

    public Layer(LayerType layerType, int numberOfNeurons) {
        this.layerType = layerType;
        this.numberOfNeurons = numberOfNeurons;
        this.neurons = new Neuron[numberOfNeurons];
    }

    public float[][] getWeights() {
        int numberOfNeurons = this.getNeurons().length;
        float[][] layerWeights = new float[this.getNeurons().length][];
        for (int n = 0; n < numberOfNeurons; n++) {
            Neuron neuron = this.getNeurons()[n];
            int numberOfSynapses = neuron.getInputSynapses().size();
            float[] neuronWeights = new float[numberOfSynapses];
            for (int s = 0; s < numberOfSynapses; s++) {
                neuronWeights[s] = neuron.getInputSynapses().get(s).getWeight();
            }
            layerWeights[n] = neuronWeights;
        }
        return layerWeights;
    }

    public float[] getBiasWeights() {
        int numberOfSynapses = this.biasNeuron.getOutputSynapses().size();
        float[] layerBiasWeights = new float[numberOfSynapses];
        for (int s = 0; s < numberOfSynapses; s++) {
            layerBiasWeights[s] = this.biasNeuron.getOutputSynapses().get(s).getWeight();
        }
        return layerBiasWeights;
    }

    public void addNeuron(Neuron neuron) {
        this.neurons[this.neuronsIndex++] = neuron;
    }

    public void calc() {
        Neuron[] neurons = this.getNeurons();
        for (Neuron neuron : neurons) {
            ArrayList<Synapse> inputSynapses = neuron.getInputSynapses();
            float calculatedValue = 0.0f;
            for (Synapse synapse : inputSynapses) {
                float weight = synapse.getWeight();
                float inputValue = synapse.getInputNeuron().getValue();
                calculatedValue += weight * inputValue;
            }

            neuron.setValue(Network.activationFunction.apply(calculatedValue));
        }
    }

    public LayerType getLayerType() {
        return layerType;
    }

    public int getNumberOfNeurons() {
        return numberOfNeurons;
    }

    public Neuron[] getNeurons() {
        return neurons;
    }

    public void setBiasNeuron(BiasNeuron biasNeuron) {
        this.biasNeuron = biasNeuron;
    }

    @Override
    public String toString() {
        StringBuilder weights = new StringBuilder();
        for (Neuron neuron : this.getNeurons()) {
            weights.append(neuron);
            weights.append(" ");
        }
        return weights.toString();
    }
}
