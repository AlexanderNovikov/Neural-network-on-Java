package com.novikovav.nn.models.neuron;

import com.novikovav.nn.models.network.Network;
import com.novikovav.nn.models.synapse.Synapse;

public class HiddenNeuron extends Neuron {
    private float delta;

    public HiddenNeuron() {
        super(NeuronType.HIDDEN);
    }

    public float getDelta() {
        return delta;
    }

    public float updateDelta() {
        this.delta = Network.activationDerivativeFunction.apply(this.getValue());
        float weightsDeltaSum = 0.0f;
        for (Synapse synapse : this.getOutputSynapses()) {
            float weight = synapse.getWeight();
            Neuron neuron = synapse.getOutputNeuron();
            NeuronType neuronType = neuron.getNeuronType();
            switch (neuronType) {
                case OUTPUT -> {
                    OutputNeuron outputNeuron = (OutputNeuron) neuron;
                    weightsDeltaSum += weight * outputNeuron.getDelta();
                }
                case HIDDEN -> {
                    HiddenNeuron hiddenNeuron = (HiddenNeuron) neuron;
                    weightsDeltaSum += weight * hiddenNeuron.getDelta();
                }
            }
        }
        this.delta *= weightsDeltaSum;
        return this.delta;
    }
}
