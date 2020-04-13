package com.novikovav.nn.models.synapse;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.novikovav.nn.models.network.Network;
import com.novikovav.nn.models.neuron.HiddenNeuron;
import com.novikovav.nn.models.neuron.Neuron;
import com.novikovav.nn.models.neuron.NeuronType;
import com.novikovav.nn.models.neuron.OutputNeuron;

import java.util.Random;

public class Synapse {
    private float weight;
    private Neuron inputNeuron;
    private Neuron outputNeuron;
    private float gradient;
    private float weightDelta = 0.0f;

    public Synapse() {
        this.weight = new Random().nextFloat();
        this.weight = -1 + (1 - (-1)) * new Random().nextFloat();
    }

    public Synapse(float weight) {
        this.weight = weight;
    }

    public float getGradient() {
        return gradient;
    }

    public float updateGradient() {
        this.gradient = this.inputNeuron.getValue();
        Neuron neuron = this.outputNeuron;
        NeuronType neuronType = neuron.getNeuronType();
        switch (neuronType) {
            case OUTPUT -> {
                OutputNeuron outputNeuron = (OutputNeuron) neuron;
                this.gradient *= outputNeuron.getDelta();
            }
            case HIDDEN -> {
                HiddenNeuron hiddenNeuron = (HiddenNeuron) neuron;
                this.gradient *= hiddenNeuron.getDelta();
            }
        }
        return this.gradient;
    }

    public float updateWeight() {
        this.weightDelta = (Network.lr * this.gradient) + (Network.moment * this.weightDelta);
        this.weight += this.weightDelta;
        return this.weight;
    }

    public float getWeight() {
        return weight;
    }

    public void setWeight(float weight) {
        this.weight = weight;
    }

    @JsonIgnore
    public Neuron getInputNeuron() {
        return inputNeuron;
    }

    public void setInputNeuron(Neuron inputNeuron) {
        this.inputNeuron = inputNeuron;
    }

    @JsonIgnore
    public Neuron getOutputNeuron() {
        return outputNeuron;
    }

    public void setOutputNeuron(Neuron outputNeuron) {
        this.outputNeuron = outputNeuron;
    }
}
