package com.novikovav.nn.models.layer;

import com.novikovav.nn.models.neuron.BiasNeuron;
import com.novikovav.nn.models.neuron.HiddenNeuron;
import com.novikovav.nn.models.neuron.Neuron;
import com.novikovav.nn.models.synapse.Synapse;
import com.novikovav.nn.models.synapse.SynapseType;

public class HiddenLayer extends Layer {
    private final Layer prevLayer;
    private final boolean withBias;
    private final float[][] weights;
    private final float[] biasWeights;

    public HiddenLayer(int numberOfNeurons, Layer prevLayer, boolean withBias, float[][] weights, float[] biasWeights) {
        super(LayerType.HIDDEN, numberOfNeurons);
        this.prevLayer = prevLayer;
        this.withBias = withBias;
        this.weights = weights;
        this.biasWeights = biasWeights;
        this.build();
    }

    public void learn() {
        for (Neuron neuron : this.getNeurons()) {
            ((HiddenNeuron) neuron).updateDelta();
            for (Synapse synapse : neuron.getInputSynapses()) {
                synapse.updateGradient();
                synapse.updateWeight();
            }
        }
    }

    private void build() {
        BiasNeuron biasNeuron = new BiasNeuron();
        for (int i = 0; i < this.getNumberOfNeurons(); i++) {
            HiddenNeuron hiddenNeuron = new HiddenNeuron();
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
                hiddenNeuron.addSynapse(synapse, SynapseType.IN);
                neuron.addSynapse(synapse, SynapseType.OUT);
                synapse.setInputNeuron(neuron);
                synapse.setOutputNeuron(hiddenNeuron);
            }
            if (this.withBias) {
                Synapse biasSynapse = new Synapse();
                if (this.biasWeights != null && i < this.biasWeights.length) {
                    biasSynapse.setWeight(this.biasWeights[i]);
                }
                hiddenNeuron.addSynapse(biasSynapse, SynapseType.IN);
                biasNeuron.addSynapse(biasSynapse, SynapseType.OUT);
                biasSynapse.setInputNeuron(biasNeuron);
                biasSynapse.setOutputNeuron(hiddenNeuron);
            }

            this.addNeuron(hiddenNeuron);
        }
        if (this.withBias) {
            this.setBiasNeuron(biasNeuron);
        }
    }
}
