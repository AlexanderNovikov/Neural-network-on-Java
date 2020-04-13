package com.novikovav.nn.models.network;

import java.util.ArrayList;

public class NetworkLayers {
    private int numberOfInputNeurons;
    private int[] numberOfHiddenNeurons;
    private int numberOfOutputNeurons;

    public NetworkLayers(String numberOfNeurons) throws Exception {
        this.parseNumberOfNeuronsString(numberOfNeurons);
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

    private void parseNumberOfNeuronsString(String numberOfNeurons) throws Exception {
        String[] layers = numberOfNeurons.split(",");
        int numberOfLayers = layers.length;
        if (numberOfLayers < 3) {
            throw new Exception("There should at least one input, hidden and output layers");
        }
        ArrayList<Integer> numberOfHiddenNeuronsArrayList = new ArrayList<>();
        for (int i = 0; i < numberOfLayers; i++) {
            if (i == 0) {
                this.numberOfInputNeurons = Integer.parseInt(layers[i]);
            } else if (i == numberOfLayers - 1) {
                this.numberOfOutputNeurons = Integer.parseInt(layers[i]);
            } else {
                numberOfHiddenNeuronsArrayList.add(Integer.parseInt(layers[i]));
            }
        }
        int[] numberOfHiddenNeurons = new int[numberOfHiddenNeuronsArrayList.size()];
        for (int i = 0; i < numberOfHiddenNeuronsArrayList.size(); i++) {
            numberOfHiddenNeurons[i] = numberOfHiddenNeuronsArrayList.get(i);
        }
        this.numberOfHiddenNeurons = numberOfHiddenNeurons;
    }
}
