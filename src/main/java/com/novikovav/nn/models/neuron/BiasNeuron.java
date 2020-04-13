package com.novikovav.nn.models.neuron;

public class BiasNeuron extends Neuron {

    public BiasNeuron() {
        super(NeuronType.BIAS);
        this.setValue(1.0f);
    }
}
