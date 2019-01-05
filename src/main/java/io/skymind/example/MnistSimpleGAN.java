package io.skymind.example;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ActivationLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.DropoutLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.impl.ActivationLReLU;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import javax.swing.*;


/**
 * Relatively small GAN example using only Dense layers with dropout to generate handwritten
 * digits from MNIST data.
 */
public class MnistSimpleGAN {

    private static final int LATENT_DIM = 100;

    private static final double LEARNING_RATE = 0.0002;
    private static final IUpdater UPDATER_ZERO = Sgd.builder().learningRate(0.0).build();
    private static final IUpdater UPDATER = Adam.builder().learningRate(LEARNING_RATE).beta1(0.5).build();


    public static MultiLayerNetwork getGenerator() {
        MultiLayerConfiguration genConf = new NeuralNetConfiguration.Builder()
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.IDENTITY)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .gradientNormalizationThreshold(100)
                .list()
                .layer(new DenseLayer.Builder().nIn(100).nOut(256).weightInit(WeightInit.NORMAL).build())
                .layer(new ActivationLayer.Builder(new ActivationLReLU(0.2)).build())
                .layer(new DenseLayer.Builder().nIn(256).nOut(512).build())
                .layer(new ActivationLayer.Builder(new ActivationLReLU(0.2)).build())
                .layer(new DenseLayer.Builder().nIn(512).nOut(1024).build())
                .layer(new ActivationLayer.Builder(new ActivationLReLU(0.2)).build())
                .layer(new DenseLayer.Builder().nIn(1024).nOut(784).activation(Activation.TANH).build())
                .build();
        return new MultiLayerNetwork(genConf);
    }


    public static MultiLayerNetwork getDiscriminator(IUpdater updater) {
        MultiLayerConfiguration discConf = new NeuralNetConfiguration.Builder()
                .seed(42)
                .updater(updater)
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.IDENTITY)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .gradientNormalizationThreshold(100)
                .list()
                .layer(new DenseLayer.Builder().nIn(784).nOut(1024).updater(updater).build())
                .layer(new ActivationLayer.Builder(new ActivationLReLU(0.2)).build())
                .layer(new DropoutLayer.Builder(1 - 0.5).build())
                .layer(new DenseLayer.Builder().nIn(1024).nOut(512).updater(updater).build())
                .layer(new ActivationLayer.Builder(new ActivationLReLU(0.2)).build())
                .layer(new DropoutLayer.Builder(1 - 0.5).build())
                .layer(new DenseLayer.Builder().nIn(512).nOut(256).updater(updater).build())
                .layer(new ActivationLayer.Builder(new ActivationLReLU(0.2)).build())
                .layer(new DropoutLayer.Builder(1 - 0.5).build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.XENT).nIn(256).nOut(1)
                        .activation(Activation.SIGMOID).updater(updater).build())
                .build();

        return new MultiLayerNetwork(discConf);
    }

    public static void main(String[] args) throws Exception {
        GAN gan = new GAN.Builder()
                .generator(MnistSimpleGAN::getGenerator)
                .discriminator(MnistSimpleGAN::getDiscriminator)
                .latentDimension(LATENT_DIM)
                .seed(42)
                .updater(UPDATER)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .gradientNormalizationThreshold(100)
                .build();

        Nd4j.getMemoryManager().setAutoGcWindow(15 * 1000);

        int batchSize = 128;
        MnistDataSetIterator trainData = new MnistDataSetIterator(batchSize, true, 42);


        // Sample from latent space once to visualize progress on image generation.
        int numSamples = 9;
        JFrame frame = GANVisualizationUtils.initFrame();
        JPanel panel = GANVisualizationUtils.initPanel(frame, numSamples);

        for (int i = 0; i < 100; i++) {
            trainData.reset();
            int j = 0;
            while (trainData.hasNext()) {
                gan.fit(trainData.next());
                //gan.fit(trainData, 1);

                if (j % 10 == 0) {
                    INDArray fakeIn = Nd4j.rand(new int[]{batchSize, LATENT_DIM});
                    System.out.println("Epoch " + (i + 1) + " Iteration " + j + " Visualizing...");
                    INDArray[] samples = new INDArray[numSamples];
                    for (int k = 0; k < numSamples; k++) {
                        INDArray input = fakeIn.getRow(k);
                        samples[k] = gan.getGenerator().output(input, false);
                    }
                    GANVisualizationUtils.visualize(samples, frame, panel);
                }
                j++;
            }
        }
    }
}