package com.novikovav.nn;

import com.novikovav.nn.models.network.Network;
import com.novikovav.nn.models.network.NetworkState;
import com.novikovav.nn.models.network.NetworkType;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.util.ArrayList;

import static org.junit.Assert.assertEquals;

public class NetworkTest {

    private Network network;

    @Before
    public void initTest() {
        NetworkState networkState = new NetworkState();
        float[][][] hiddenWeights = new float[][][]{
                new float[][]{
                        new float[]{
                                0.45f,
                                -0.12f
                        },
                        new float[]{
                                0.78f,
                                0.13f
                        }
                }
        };
        float[][] outputWeights = new float[][]{
                new float[]{
                        1.5f,
                        -2.3f
                }
        };
        float[] outputBiasWeights = new float[]{
                -1.8f
        };
        networkState.setNumberOfInputNeurons(2);
        networkState.setNumberOfHiddenNeurons(new int[]{2});
        networkState.setNumberOfOutputNeurons(1);
        networkState.setHiddenWeights(hiddenWeights);
        networkState.setOutputBiasWeights(outputBiasWeights);
        networkState.setOutputWeights(outputWeights);
        this.network = new Network(NetworkType.SIGMOID, false, 0.7f, 0.3f);
        try {
            this.network.restore(null, networkState);
        } catch (IOException ioe) {
            System.out.println(ioe.getLocalizedMessage());
        }
    }

    @After
    public void afterTest() {
        this.network = null;
    }

    @Test
    public void calcOneTest() throws Exception {
        float[] out = this.network.calcOne(new ArrayList<>() {{
            add(1.0f);
            add(0.0f);
        }});
        assertEquals(String.valueOf(0.34049135), String.valueOf(out[0]));
    }

    @Test
    public void getErrorTest() throws Exception {
        this.network.calcOne(new ArrayList<>() {{
            add(1.0f);
            add(0.0f);
        }});
        Float error = this.network.getError(new ArrayList<>() {{
            add(1.0f);
        }});
        assertEquals(String.valueOf(0.43495166), String.valueOf(error));
    }

    @Test
    public void learnTest() throws Exception {
        ArrayList<Float> input = new ArrayList<>() {{
            add(1.0f);
            add(0.0f);
        }};
        float[] resultBefore = this.network.calcOne(input);
        ArrayList<Float> ideal = new ArrayList<>() {{
            add(1.0f);
        }};
        Float errorBefore = this.network.getError(ideal);
        this.network.learn(ideal);
        float[] resultAfter = this.network.calcOne(input);
        Float errorAfter = this.network.getError(ideal);
        assertEquals(String.valueOf(0.34049135), String.valueOf(resultBefore[0]));
        assertEquals(String.valueOf(0.43495166), String.valueOf(errorBefore));
        assertEquals(String.valueOf(0.36927965), String.valueOf(resultAfter[0]));
        assertEquals(String.valueOf(0.3978082), String.valueOf(errorAfter));
    }
}
