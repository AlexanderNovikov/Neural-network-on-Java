package com.novikovav.nn.models.network;

public class NetworkState {
    private int numberOfInputNeurons;
    private int[] numberOfHiddenNeurons;
    private int numberOfOutputNeurons;
    private float[][][] hiddenWeights;
    private float[][] outputWeights;
    private float[] outputBiasWeights;
    private float[][] hiddenBiasWeights;

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

    public float[][][] getHiddenWeights() {
        return hiddenWeights;
    }

    public void setHiddenWeights(float[][][] hiddenWeights) {
        this.hiddenWeights = hiddenWeights;
    }

    public float[][] getOutputWeights() {
        return outputWeights;
    }

    public void setOutputWeights(float[][] outputWeights) {
        this.outputWeights = outputWeights;
    }

    public float[] getOutputBiasWeights() {
        return outputBiasWeights;
    }

    public void setOutputBiasWeights(float[] inputBiasWeights) {
        this.outputBiasWeights = inputBiasWeights;
    }

    public float[][] getHiddenBiasWeights() {
        return hiddenBiasWeights;
    }

    public void setHiddenBiasWeights(float[][] hiddenBiasWeights) {
        this.hiddenBiasWeights = hiddenBiasWeights;
    }
}
