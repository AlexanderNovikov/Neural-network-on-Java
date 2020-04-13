package com.novikovav.nn;

import com.novikovav.nn.models.network.Network;
import com.novikovav.nn.models.network.NetworkLayers;
import com.novikovav.nn.models.network.NetworkType;
import com.novikovav.nn.utils.Loader;

import java.util.ArrayList;

public class Main {
    private static String layers = "784,100,8";
    private static int epochs = 100;
    private static String stateFile = "data/state.json";
    private static boolean restore = false;
    private static boolean evaluate = false;

    public static void main(String[] args) {
        for (String arg : args) {
            if (arg.startsWith("--layers=")) {
                layers = arg.replace("--layers=", "");
            }
            if (arg.startsWith("--epochs=")) {
                epochs = Integer.parseInt(arg.replace("--epochs=", ""));
            }
            if (arg.startsWith("--state-file=")) {
                stateFile = arg.replace("--state-file=", "");
            }
            if (arg.startsWith("--restore-state")) {
                restore = true;
            }
            if (arg.startsWith("--evaluate")) {
                evaluate = true;
            }
        }

        Loader loader = new Loader();
        try {
            Network network;
            if (restore) {
                network = new Network(NetworkType.SIGMOID, true, 1.7f, 0.3f);
                network.restore(stateFile, null);
            } else {
                NetworkLayers networkLayers = new NetworkLayers(layers);
                network = new Network(networkLayers, NetworkType.SIGMOID, true, 0.05f, 0.3f);
            }

            if (evaluate) {
                String imagesPath = "sets/t10k-images-idx3-ubyte";
                String labelsPath = "sets/t10k-labels-idx1-ubyte";
                ArrayList<ArrayList<Float>> images2 = loader.readImages(imagesPath);
                ArrayList<ArrayList<Float>> labels2 = loader.readLabels(labelsPath);
                ArrayList<Float> image = images2.get(128);
                ArrayList<Float> label = labels2.get(128);

                StringBuilder inp = new StringBuilder("Input:");
                for (Float i : label) {
                    inp.append(String.format("%f", i));
                    inp.append(" ");
                }
                System.out.println(inp);
                float[] result = network.calcOne(image);
                StringBuilder res = new StringBuilder("Output:");
                for (Float r : result) {
                    res.append(String.format("%f", r));
                    res.append(" ");
                }
                System.out.println(res);
                Float error = network.getError(label);
                System.out.println("Error:");
                System.out.println(String.format("%f", error));
            } else {
                ArrayList<ArrayList<Float>> images = loader.readImages("sets/train-images-idx3-ubyte");
                ArrayList<ArrayList<Float>> labels = loader.readLabels("sets/train-labels-idx1-ubyte");

                for (int epoch = 1; epoch < epochs; epoch++) {
                    network.calc(images, labels);
                    network.save(stateFile);
                    System.out.println("epoch: " + epoch + " of " + epochs);
                }
            }
        } catch (Exception e) {
            System.out.println(e.getLocalizedMessage());
        }
    }
}
