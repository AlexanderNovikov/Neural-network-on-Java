package com.novikovav.nn.models.network;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.novikovav.nn.models.layer.*;
import com.novikovav.nn.utils.Utils;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.function.Function;

public class Network {
    private int numberOfInputNeurons;
    private int[] numberOfHiddenNeurons;
    private int numberOfOutputNeurons;
    private Layer[] layers;
    private int layersIndex = 0;
    private InputLayer inputLayer;
    private OutputLayer outputLayer;
    private HiddenLayer[] hiddenLayers;
    private int hiddenLayersIndex = 0;
    private final boolean withBias;
    private float[][][] hiddenWeights;
    private float[][] outputWeights;
    private float[] outputBiasWeights;
    private float[][] hiddenBiasWeights;

    public static Function<Float, Float> activationFunction;
    public static Function<Float, Float> activationDerivativeFunction;
    public static Float lr;
    public static Float moment;

    public Network(NetworkType networkType, boolean withBias, Float lr, Float moment) {
        this.withBias = withBias;
        switch (networkType) {
            case SIGMOID -> {
                Network.activationFunction = Utils::sigmoid;
                Network.activationDerivativeFunction = Utils::sigmoidDerivative;
            }
            case HTAN -> {
            }
        }
        Network.lr = lr;
        Network.moment = moment;
    }

    public Network(NetworkLayers networkLayers, NetworkType networkType, boolean withBias, Float lr, Float moment) {
        this.numberOfInputNeurons = networkLayers.getNumberOfInputNeurons();
        this.numberOfHiddenNeurons = networkLayers.getNumberOfHiddenNeurons();
        this.numberOfOutputNeurons = networkLayers.getNumberOfOutputNeurons();
        this.withBias = withBias;
        this.build();
        switch (networkType) {
            case SIGMOID -> {
                Network.activationFunction = Utils::sigmoid;
                Network.activationDerivativeFunction = Utils::sigmoidDerivative;
            }
            case HTAN -> {
            }
        }
        Network.lr = lr;
        Network.moment = moment;
    }

    public void save(String path) throws IOException {
        NetworkState networkState = new NetworkState();
        int numberOfHiddenLayers = this.getHiddenLayers().length;
        float[][][] hiddenWeights = new float[numberOfHiddenLayers][][];
        float[][] biasWeights = new float[numberOfHiddenLayers][];
        for (int h = 0; h < numberOfHiddenLayers; h++) {
            HiddenLayer hiddenLayer = this.getHiddenLayers()[h];
            hiddenWeights[h] = hiddenLayer.getWeights();
            biasWeights[h] = hiddenLayer.getBiasWeights();
        }
        networkState.setHiddenWeights(hiddenWeights);
        networkState.setOutputWeights(this.outputLayer.getWeights());
        networkState.setHiddenBiasWeights(biasWeights);
        networkState.setOutputBiasWeights(this.outputLayer.getBiasWeights());
        networkState.setNumberOfInputNeurons(this.numberOfInputNeurons);
        networkState.setNumberOfHiddenNeurons(this.numberOfHiddenNeurons);
        networkState.setNumberOfOutputNeurons(this.numberOfOutputNeurons);
        ObjectMapper mapper = new ObjectMapper();
        String json = mapper.writeValueAsString(networkState);
        BufferedWriter writer = new BufferedWriter(new FileWriter(path));
        writer.write(json);
        writer.close();
    }

    public void restore(String path, NetworkState networkState) throws IOException {
        ObjectMapper mapper = new ObjectMapper();
        if (networkState == null) {
            networkState = mapper.readValue(new File(path), NetworkState.class);
        }
        if (networkState != null) {
            this.numberOfInputNeurons = networkState.getNumberOfInputNeurons();
            this.numberOfHiddenNeurons = networkState.getNumberOfHiddenNeurons();
            this.numberOfOutputNeurons = networkState.getNumberOfOutputNeurons();
            this.hiddenWeights = networkState.getHiddenWeights();
            this.outputWeights = networkState.getOutputWeights();
            this.hiddenBiasWeights = networkState.getHiddenBiasWeights();
            this.outputBiasWeights = networkState.getOutputBiasWeights();
            this.build();
        }
    }

    public float[][] calc(ArrayList<ArrayList<Float>> inputs, ArrayList<ArrayList<Float>> ideals) throws Exception {
        float[][] result = new float[inputs.size()][];
        Object[] in = inputs.toArray();
        Object[] out = ideals.toArray();
        for (int i = 0; i < inputs.size(); i++) {
            ArrayList<Float> input = (ArrayList<Float>) in[i];
            ArrayList<Float> ideal = (ArrayList<Float>) out[i];

            result[i] = this.calcOne(input);
            this.learn(ideal);
        }
        return result;
    }

    public float[] calcOne(ArrayList<Float> input) throws Exception {
        if (input.size() != this.numberOfInputNeurons) {
            throw new Exception("Number of input values and input neurons does not match");
        }
        this.inputLayer.setData(input);
        for (int i = 1; i < this.layers.length; i++) {
            Layer layer = this.layers[i];
            layer.calc();
        }

        return this.outputLayer.getResult();
    }

    public Float getError(ArrayList<Float> input) throws Exception {
        if (input.size() != this.numberOfOutputNeurons) {
            throw new Exception("Number of ideal values and output neurons does not match");
        }
        return this.outputLayer.getError(input);
    }

    public void learn(ArrayList<Float> input) {
        this.outputLayer.learn(input);
        for (int i = this.hiddenLayers.length - 1; i >= 0; i--) {
            HiddenLayer hiddenLayer = this.hiddenLayers[i];
            hiddenLayer.learn();
        }
    }

    private void build() {
        this.layers = new Layer[2 + this.numberOfHiddenNeurons.length];
        this.hiddenLayers = new HiddenLayer[this.numberOfHiddenNeurons.length];
        this.inputLayer = null;
        this.outputLayer = null;
        Layer prevLayer = new InputLayer(this.numberOfInputNeurons);
        this.addLayer(prevLayer);
        for (int hiddenIndex = 0; hiddenIndex < numberOfHiddenNeurons.length; hiddenIndex++) {
            int hidden = numberOfHiddenNeurons[hiddenIndex];
            float[][] hiddenWeights = null;
            if (this.hiddenWeights != null && hiddenIndex < this.hiddenWeights.length) {
                hiddenWeights = this.hiddenWeights[hiddenIndex];
            }
            float[] hiddenBiasWeights = null;
            if (this.hiddenBiasWeights != null && hiddenIndex < this.hiddenBiasWeights.length) {
                hiddenBiasWeights = this.hiddenBiasWeights[hiddenIndex];
            }
            prevLayer = new HiddenLayer(hidden, prevLayer, this.withBias, hiddenWeights, hiddenBiasWeights);
            this.addLayer(prevLayer);
        }
        prevLayer = new OutputLayer(this.numberOfOutputNeurons, prevLayer, this.withBias, this.outputWeights, this.outputBiasWeights);
        this.addLayer(prevLayer);
    }

    private void addLayer(Layer layer) {
        this.layers[this.layersIndex++] = layer;
        LayerType layerType = layer.getLayerType();
        switch (layerType) {
            case INPUT -> this.inputLayer = (InputLayer) layer;
            case HIDDEN -> this.hiddenLayers[this.hiddenLayersIndex++] = (HiddenLayer) layer;
            case OUTPUT -> this.outputLayer = (OutputLayer) layer;
        }
    }

    public HiddenLayer[] getHiddenLayers() {
        return this.hiddenLayers;
    }

    public int getNumberOfInputNeurons() {
        return numberOfInputNeurons;
    }

    public void setNumberOfInputNeurons(int numberOfInputNeurons) {
        this.numberOfInputNeurons = numberOfInputNeurons;
    }

    public int[] getNumberOfHiddenNeurons() {
        return numberOfHiddenNeurons;
    }

    public void setNumberOfHiddenNeurons(int[] numberOfHiddenNeurons) {
        this.numberOfHiddenNeurons = numberOfHiddenNeurons;
    }

    public int getNumberOfOutputNeurons() {
        return numberOfOutputNeurons;
    }

    public void setNumberOfOutputNeurons(int numberOfOutputNeurons) {
        this.numberOfOutputNeurons = numberOfOutputNeurons;
    }

    public InputLayer getInputLayer() {
        return inputLayer;
    }
}
