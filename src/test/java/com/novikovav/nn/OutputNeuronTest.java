package com.novikovav.nn;

import com.novikovav.nn.models.neuron.OutputNeuron;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class OutputNeuronTest {

    private OutputNeuron outputNeuron;

    @Before
    public void initTest() {
        this.outputNeuron = new OutputNeuron();
        this.outputNeuron.setValue(0.34049134000389103f);
    }

    @After
    public void afterTest() {
        this.outputNeuron = null;
    }

    @Test
    public void setDeltaTest() throws Exception {
        Float out = this.outputNeuron.updateDelta(1.0f);
        assertEquals(String.valueOf(0.14809728), String.valueOf(out));
    }

    @Test
    public void getDeltaTest() throws Exception {
        this.outputNeuron.updateDelta(1.0f);
        assertEquals(String.valueOf(0.14809728), String.valueOf(this.outputNeuron.getDelta()));
    }
}
