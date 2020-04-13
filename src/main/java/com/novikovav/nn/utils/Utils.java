package com.novikovav.nn.utils;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.novikovav.nn.models.NetworkData;
import com.novikovav.nn.models.network.Network;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;

import static java.awt.image.BufferedImage.TYPE_3BYTE_BGR;

public class Utils {
    private final static float divider = 255.0f;

    public static Float sigmoid(Float x) {
        return 1.0f / (1.0f + (float) Math.exp(-x));
    }

    public static Float sigmoidDerivative(Float x) {
        return (1 - x) * x;
    }

    public static void writeNetworkToFile(String path, Network network) {
        ObjectMapper mapper = new ObjectMapper();
        try {
            String json = mapper.writeValueAsString(network);
            BufferedWriter writer = new BufferedWriter(new FileWriter(path));
            writer.write(json);
            writer.close();
        } catch (JsonProcessingException jpe) {
            System.out.println(jpe.getMessage());
        } catch (IOException ioe) {
            System.out.println(ioe.getLocalizedMessage());
        }
    }

    public static NetworkData imageToData(String path) {
        File f = new File(path);
        BufferedImage bufferedImage = null;
        try {
            bufferedImage = ImageIO.read(f);
        } catch (IOException ioe) {
            System.out.println(ioe.getLocalizedMessage());
        }
        ArrayList<Float> result = new ArrayList<>();

        NetworkData networkData = new NetworkData();
        if (bufferedImage != null) {
            int width = bufferedImage.getHeight();
            int height = bufferedImage.getWidth();
            networkData.setWidth(width);
            networkData.setHeight(height);

            for (int x = 0; x < height; x++) {
                for (int y = 0; y < width; y++) {
                    int p = bufferedImage.getRGB(x, y);
                    float r = ((p >> 16) & 0xff) / divider;
                    float g = ((p >> 8) & 0xff) / divider;
                    float b = (p & 0xff) / divider;
                    result.add(r);
                    result.add(g);
                    result.add(b);
                }
            }
            networkData.setData(result);
        }

        return networkData;
    }

    public static void dataToImage(ArrayList<Float> data, String path, int width, int height) {
        Object[] dataArr = data.toArray();
        BufferedImage bufferedImage = new BufferedImage(height, width, TYPE_3BYTE_BGR);
        int x = 0;
        for (int rowIndex = 0; rowIndex < dataArr.length; rowIndex += width * 3) {
            Object[] row = Arrays.copyOfRange(dataArr, rowIndex, Math.min(dataArr.length, rowIndex + width * 3));
            for (int p = 0; p < row.length; p += 3) {
                Object[] rgb = Arrays.copyOfRange(row, p, Math.min(row.length, p + 3));
                float r = (float) rgb[0];
                float g = (float) rgb[1];
                float b = (float) rgb[2];
                Color color = new Color(r, g, b);
                bufferedImage.setRGB(x, p / 3, color.getRGB());
            }
            x++;
        }

        File file = new File(path);
        try {
            ImageIO.write(bufferedImage, "jpg", file);
        } catch (IOException ioe) {
            System.out.println(ioe.getLocalizedMessage());
        }
    }
}
