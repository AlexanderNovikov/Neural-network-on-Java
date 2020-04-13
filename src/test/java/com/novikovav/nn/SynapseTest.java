package com.novikovav.nn;

import com.novikovav.nn.models.neuron.HiddenNeuron;
import com.novikovav.nn.models.neuron.OutputNeuron;
import com.novikovav.nn.models.synapse.Synapse;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class SynapseTest {

    private Synapse synapse;

    @Before
    public void initTest() {
        this.synapse = new Synapse(1.5f);
        HiddenNeuron hiddenNeuron = new HiddenNeuron();
        hiddenNeuron.setValue(0.34049134000389103f);
        this.synapse.setInputNeuron(hiddenNeuron);
        OutputNeuron outputNeuron = new OutputNeuron();
        outputNeuron.setDelta(0.14809727784386606f);
        this.synapse.setOutputNeuron(outputNeuron);
    }

    @After
    public void afterTest() {
        this.synapse = null;
    }

    @Test
    public void setGradientTest() throws Exception {
        Float out = this.synapse.updateGradient();
        assertEquals(String.valueOf(0.050425842), String.valueOf(out));
    }

    @Test
    public void getGradientTest() throws Exception {
        this.synapse.updateGradient();
        assertEquals(String.valueOf(0.050425842), String.valueOf(this.synapse.getGradient()));
    }

    @Test
    public void updateWeightTest() throws Exception {
        this.synapse.updateGradient();
        Float out = this.synapse.updateWeight();
        assertEquals(String.valueOf(1.5352981), String.valueOf(out));
    }

    @Test
    public void getWeightTest() throws Exception {
        this.synapse.updateGradient();
        this.synapse.updateWeight();
        assertEquals(String.valueOf(1.5352981), String.valueOf(this.synapse.getWeight()));
    }
}
