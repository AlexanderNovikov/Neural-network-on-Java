package com.novikovav.nn.models.neuron;

public class InputNeuron extends Neuron {

    public InputNeuron() {
        super(NeuronType.INPUT);
    }

    public void setValue(float value) {
        if (value > 1) {
            value = 1.0f / value;
        }
        super.setValue(value);
    }
}
