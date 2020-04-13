package com.novikovav.nn.utils;

import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;

public class Loader {

    private final ArrayList<Float> LABELS = new ArrayList<>() {{
        add(0.0f);
        add(0.0f);
        add(0.0f);
        add(0.0f);
        add(0.0f);
        add(0.0f);
        add(0.0f);
        add(0.0f);
        add(0.0f);
        add(0.0f);
    }};

    public ArrayList<ArrayList<Float>> readImages(String imageFile) throws IOException {
        ArrayList<ArrayList<Float>> images = new ArrayList<>();

        DataInputStream imageFileData = new DataInputStream(new FileInputStream(imageFile));
        imageFileData.readInt(); //Skip the "magic number"
        int n = imageFileData.readInt();
        int height = imageFileData.readInt();
        int width = imageFileData.readInt();
        int length = height * width;
        byte[] all = imageFileData.readAllBytes();

        for (int i = 0; i < all.length; i += length) {
            byte[] image = Arrays.copyOfRange(all, i, Math.min(all.length, i + length));
            ArrayList<Float> newImage = new ArrayList<>();
            for (byte b : image) {
                newImage.add(b / 255f);
            }
            images.add(newImage);
        }

        imageFileData.close();

        return images;
    }

    public ArrayList<ArrayList<Float>> readLabels(String labelFile) throws IOException {
        ArrayList<ArrayList<Float>> labels = new ArrayList<>();

        DataInputStream labelFileData = new DataInputStream(new FileInputStream(labelFile));

        labelFileData.readInt(); //Skip the "magic number"
        int n = labelFileData.readInt();

        for (int i = 0; i < n; i++) {
            float label = labelFileData.readUnsignedByte();
            ArrayList<Float> newLabel = new ArrayList<>(this.LABELS);
            newLabel.set((int) label, 1.0f);
            labels.add(newLabel);
        }

        labelFileData.close();

        return labels;
    }
}
