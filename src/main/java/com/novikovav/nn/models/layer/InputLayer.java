package com.novikovav.nn.models.layer;

import com.novikovav.nn.models.neuron.InputNeuron;
import com.novikovav.nn.models.neuron.Neuron;

import java.util.ArrayList;

public class InputLayer extends Layer {

    public InputLayer(int numberOfNeurons) {
        super(LayerType.INPUT, numberOfNeurons);
        this.build();
    }

    public void setData(ArrayList<Float> input) {
        int index = 0;
        Neuron[] inputNeurons = this.getNeurons();
        for (Float val : input) {
            inputNeurons[index].setValue(val);
            index++;
        }
    }

    private void build() {
        for (int i = 0; i < this.getNumberOfNeurons(); i++) {
            this.addNeuron(new InputNeuron());
        }
    }
}
