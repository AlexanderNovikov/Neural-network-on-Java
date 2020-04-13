package com.novikovav.nn.models.neuron;

import com.novikovav.nn.models.network.Network;

public class OutputNeuron extends Neuron {
    private float delta;

    public OutputNeuron() {
        super(NeuronType.OUTPUT);
    }

    public float getDelta() {
        return delta;
    }

    public void setDelta(float delta) {
        this.delta = delta;
    }

    public float updateDelta(float idealValue) {
        this.delta = (idealValue - this.getValue()) * Network.activationDerivativeFunction.apply(this.getValue());
        return delta;
    }
}
