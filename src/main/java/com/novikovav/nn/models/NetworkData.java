package com.novikovav.nn.models;

import java.util.ArrayList;

public class NetworkData {
    private ArrayList<Float> data;
    private int height;
    private int width;

    public ArrayList<Float> getData() {
        return data;
    }

    public void setData(ArrayList<Float> data) {
        this.data = data;
    }

    public int getHeight() {
        return height;
    }

    public void setHeight(int height) {
        this.height = height;
    }

    public int getWidth() {
        return width;
    }

    public void setWidth(int width) {
        this.width = width;
    }
}
