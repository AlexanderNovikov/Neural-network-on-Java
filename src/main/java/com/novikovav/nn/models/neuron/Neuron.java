package com.novikovav.nn.models.neuron;

import com.novikovav.nn.models.synapse.Synapse;
import com.novikovav.nn.models.synapse.SynapseType;

import java.util.ArrayList;

public class Neuron {
    private float value;
    private final NeuronType neuronType;
    private final ArrayList<Synapse> synapses = new ArrayList<>();
    private final ArrayList<Synapse> inputSynapses = new ArrayList<>();
    private final ArrayList<Synapse> outputSynapses = new ArrayList<>();

    public Neuron(NeuronType neuronType) {
        this.neuronType = neuronType;
    }

    public void addSynapse(Synapse synapse, SynapseType synapseType) {
        this.synapses.add(synapse);
        switch (synapseType) {
            case IN -> this.inputSynapses.add(synapse);
            case OUT -> this.outputSynapses.add(synapse);
        }
    }

    public float getValue() {
        return value;
    }

    public void setValue(float value) {
        this.value = value;
    }

    public NeuronType getNeuronType() {
        return neuronType;
    }

    public ArrayList<Synapse> getInputSynapses() {
        return inputSynapses;
    }

    public ArrayList<Synapse> getOutputSynapses() {
        return outputSynapses;
    }

    @Override
    public String toString() {
        StringBuilder stringBuilder = new StringBuilder("[");
        this.synapses.forEach(synapse -> {
            stringBuilder.append(synapse.getWeight());
            stringBuilder.append(" ");
        });
        stringBuilder.delete(stringBuilder.length() - 1, stringBuilder.length());
        stringBuilder.append("]");
        return stringBuilder.toString();
    }
}
