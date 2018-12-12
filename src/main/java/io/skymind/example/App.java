package io.skymind.example;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import org.apache.commons.lang3.ArrayUtils;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.impl.ActivationLReLU;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.NoOpUpdater;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.NoOp;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;

public class App {
    private static final double LEARNING_RATE = 0.0002;
    private static final double L2 = 0.01;
    private static final double GRADIENT_THRESHOLD = 100.0;
    private static final IUpdater UPDATER = Adam.builder().learningRate(LEARNING_RATE).beta1(0.5).build();

    private static JFrame frame;
    private static JPanel panel;

    private static Layer[] genLayers() {
        return new Layer[] {
                new DenseLayer.Builder().nIn(100).nOut(256).weightInit(WeightInit.NORMAL).build(),
                new ActivationLayer.Builder(new ActivationLReLU(0.2)).build(),
                new DenseLayer.Builder().nIn(256).nOut(512).build(),
                new ActivationLayer.Builder(new ActivationLReLU(0.2)).build(),
                new DenseLayer.Builder().nIn(512).nOut(1024).build(),
                new ActivationLayer.Builder(new ActivationLReLU(0.2)).build(),
                new DenseLayer.Builder().nIn(1024).nOut(784).activation(Activation.TANH).build()
        };
    }

    /**
     * Returns a network config that takes in a 10x10 random number and produces a 28x28 grayscale image.
     *
     * @return config
     */
    private static MultiLayerConfiguration generator() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(42)
                .updater(UPDATER)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .gradientNormalizationThreshold(GRADIENT_THRESHOLD)
                .l2(L2)
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.IDENTITY)
                .list(genLayers())
                .build();

        return conf;
    }

    private static Layer[] disLayers(IUpdater updater) {
        return new Layer[] {
                new DenseLayer.Builder().nIn(784).nOut(1024).updater(updater).build(),
                new ActivationLayer.Builder(new ActivationLReLU(0.2)).build(),
                new DropoutLayer.Builder(1 - 0.5).build(),
                new DenseLayer.Builder().nIn(1024).nOut(512).updater(updater).build(),
                new ActivationLayer.Builder(new ActivationLReLU(0.2)).build(),
                new DropoutLayer.Builder(1 - 0.5).build(),
                new DenseLayer.Builder().nIn(512).nOut(256).updater(updater).build(),
                new ActivationLayer.Builder(new ActivationLReLU(0.2)).build(),
                new DropoutLayer.Builder(1 - 0.5).build(),
                new OutputLayer.Builder(LossFunctions.LossFunction.XENT).nIn(256).nOut(1).activation(Activation.SIGMOID).updater(updater).build()
        };
    }

    private static MultiLayerConfiguration discriminator() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(42)
                .updater(UPDATER)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .gradientNormalizationThreshold(GRADIENT_THRESHOLD)
                .l2(L2)
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.IDENTITY)
                .list(disLayers(UPDATER))
                .build();

        return conf;
    }

    private static MultiLayerConfiguration gan() {
        Layer[] genLayers = genLayers();
        Layer[] disLayers = disLayers(new NoOp()); // Freeze discriminator layers in combined network.
        Layer[] layers = ArrayUtils.addAll(genLayers, disLayers);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(42)
                .updater(UPDATER)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .gradientNormalizationThreshold(GRADIENT_THRESHOLD)
                .l2(L2)
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.IDENTITY)
                .list(layers)
                .build();

        return conf;
    }

    public static void main(String... args) throws Exception {
        //Nd4j.getMemoryManager().setAutoGcWindow(15000);

        MnistDataSetIterator trainData = new MnistDataSetIterator(128, true, 42);

        MultiLayerNetwork gen = new MultiLayerNetwork(generator());
        MultiLayerNetwork dis = new MultiLayerNetwork(discriminator());
        MultiLayerNetwork gan = new MultiLayerNetwork(gan());
        gen.init();
        dis.init();
        gan.init();

        copyParams(gen, dis, gan);

        gen.setListeners(new PerformanceListener(50, true));
        dis.setListeners(new PerformanceListener(50, true));
        gan.setListeners(new PerformanceListener(50, true));

        trainData.reset();

        while (true) {
            int j = 0;
            while (trainData.hasNext()) {
                j++;

                // generate data
                INDArray real = trainData.next().getFeatures().muli(2).subi(1);
                int batchSize = (int)real.shape()[0];

                INDArray fakeIn = Nd4j.rand(new int[]{batchSize,  100});
                INDArray fake = gen.output(fakeIn, false);

                DataSet realSet = new DataSet(real, Nd4j.zeros(batchSize, 1));
                DataSet fakeSet = new DataSet(fake, Nd4j.ones(batchSize, 1));
                DataSet fakeSet2 = new DataSet(fakeIn, Nd4j.ones(batchSize, 1));

                //DataSet data = DataSet.merge(Arrays.asList(realSet, fakeSet));

                dis.fit(realSet);
                dis.fit(fakeSet);

                // Update the discriminator in the GAN network
                updateGan(gen, dis, gan);

                gan.fit(new DataSet(Nd4j.rand(new int[] { batchSize, 100}), Nd4j.zeros(batchSize, 1)));
                //gan.fit(fakeSet2);


                // Copy the GANs generator to gen.
                updateGen(gen, gan);

                if (j % 10 == 1) {
                    System.out.println("Iteration " + j + " Visualizing...");
                    INDArray[] samples = new INDArray[9];
                    for (int k = 0; k < 9; k++) {
                        INDArray input = fakeSet2.get(k).getFeatures();
                        samples[k] = gen.output(input, false);
                    }
                    visualize(samples);
                }
            }
            trainData.reset();
        }
    }

    private static void copyParams(MultiLayerNetwork gen, MultiLayerNetwork dis, MultiLayerNetwork gan) {
        int genLayerCount = gen.getLayers().length;
        for (int i = 0; i < gan.getLayers().length; i++) {
            if (i < genLayerCount) {
                gen.getLayer(i).setParams(gan.getLayer(i).params());
            } else {
                dis.getLayer(i - genLayerCount).setParams(gan.getLayer(i).params());
            }
        }
    }

    private static void updateGen(MultiLayerNetwork gen, MultiLayerNetwork gan) {
        for (int i = 0; i < gen.getLayers().length; i++) {
            gen.getLayer(i).setParams(gan.getLayer(i).params());
        }
    }

    private static void updateGan(MultiLayerNetwork gen, MultiLayerNetwork dis, MultiLayerNetwork gan) {
        int genLayerCount = gen.getLayers().length;
        for (int i = genLayerCount; i < gan.getLayers().length; i++) {
            gan.getLayer(i).setParams(dis.getLayer(i - genLayerCount).params());
        }
    }

    private static void visualize(INDArray[] samples) {
        if (frame == null) {
            frame = new JFrame();
            frame.setTitle("Viz");
            frame.setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
            frame.setLayout(new BorderLayout());

            panel = new JPanel();

            panel.setLayout(new GridLayout(samples.length / 3, 1, 8, 8));
            frame.add(panel, BorderLayout.CENTER);
            frame.setVisible(true);
        }

        panel.removeAll();

        for (int i = 0; i < samples.length; i++) {
            panel.add(getImage(samples[i]));
        }

        frame.revalidate();
        frame.pack();
    }

    private static JLabel getImage(INDArray tensor) {
        BufferedImage bi = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);
        for (int i = 0; i < 784; i++) {
            int pixel = (int)(((tensor.getDouble(i) + 1) * 2) * 255);
            bi.getRaster().setSample(i % 28, i / 28, 0, pixel);
        }
        ImageIcon orig = new ImageIcon(bi);
        Image imageScaled = orig.getImage().getScaledInstance((int) (8 * 28), (int) (8 * 28), Image.SCALE_REPLICATE);

        ImageIcon scaled = new ImageIcon(imageScaled);

        return new JLabel(scaled);
    }
}
