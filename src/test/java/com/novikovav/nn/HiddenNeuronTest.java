package com.novikovav.nn;

import com.novikovav.nn.models.network.Network;
import com.novikovav.nn.models.network.NetworkType;
import com.novikovav.nn.models.neuron.HiddenNeuron;
import com.novikovav.nn.models.neuron.OutputNeuron;
import com.novikovav.nn.models.synapse.Synapse;
import com.novikovav.nn.models.synapse.SynapseType;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class HiddenNeuronTest {

    private HiddenNeuron hiddenNeuron;

    @Before
    public void initTest() {
        new Network(NetworkType.SIGMOID, true, 0.7f, 0.3f);
        this.hiddenNeuron = new HiddenNeuron();
        this.hiddenNeuron.setValue(0.34049134000389103f);
        Synapse synapse = new Synapse(1.5f);
        OutputNeuron outputNeuron = new OutputNeuron();
        outputNeuron.setDelta(0.14809727784386606f);
        synapse.setOutputNeuron(outputNeuron);
        this.hiddenNeuron.addSynapse(synapse, SynapseType.OUT);
    }

    @After
    public void afterTest() {
        this.hiddenNeuron = null;
    }

    @Test
    public void setDeltaTest() throws Exception {
        Float out = this.hiddenNeuron.updateDelta();
        assertEquals(String.valueOf(0.04988442), String.valueOf(out));
    }

    @Test
    public void getDeltaTest() throws Exception {
        this.hiddenNeuron.updateDelta();
        assertEquals(String.valueOf(0.04988442), String.valueOf(this.hiddenNeuron.getDelta()));
    }
}
