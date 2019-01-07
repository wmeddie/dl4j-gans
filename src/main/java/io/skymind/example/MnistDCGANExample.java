package io.skymind.example;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToCnnPreProcessor;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.function.Supplier;


/**
 * Training and visualizing a deep convolutional generative adversarial network (DCGAN) on handwritten digits.
 *
 * @author Max Pumperla, wmeddie
 */
public class MnistDCGANExample {

    private static JFrame frame;
    private static JPanel panel;

    private static final int latentDim = 100;
    private static final int height = 28;
    private static final int width = 28;
    private static final int channels = 1;


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
            int pixel = (int) (((tensor.getDouble(i) + 1) * 2) * 255);
            bi.getRaster().setSample(i % 28, i / 28, 0, pixel);
        }
        ImageIcon orig = new ImageIcon(bi);
        Image imageScaled = orig.getImage().getScaledInstance((8 * 28), (8 * 28), Image.SCALE_REPLICATE);

        ImageIcon scaled = new ImageIcon(imageScaled);

        return new JLabel(scaled);
    }

    public static void main(String[] args) throws Exception {
        Supplier<MultiLayerNetwork> genSupplier = () -> {
            return new MultiLayerNetwork(new NeuralNetConfiguration.Builder().list()
                    .layer(0, new DenseLayer.Builder().nIn(latentDim).nOut(width / 2 * height / 2 * 128)
                            .activation(Activation.LEAKYRELU).weightInit(WeightInit.NORMAL).build())
                    .layer(1, new Convolution2D.Builder().nIn(128).nOut(128).kernelSize(5, 5)
                            .convolutionMode(ConvolutionMode.Same).activation(Activation.LEAKYRELU).build())
                    // Up-sampling to 28x28x256
                    .layer(2, new Deconvolution2D.Builder().nIn(128).nOut(128).stride(2, 2)
                            .kernelSize(5, 5).convolutionMode(ConvolutionMode.Same)
                            .activation(Activation.LEAKYRELU).build())
                    .layer(3, new Convolution2D.Builder().nIn(128).nOut(128).kernelSize(5, 5)
                            .convolutionMode(ConvolutionMode.Same).activation(Activation.LEAKYRELU).build())
                    .layer(4, new Convolution2D.Builder().nIn(128).nOut(128).kernelSize(5, 5)
                            .convolutionMode(ConvolutionMode.Same).activation(Activation.LEAKYRELU).build())
                    .layer(5, new Convolution2D.Builder().nIn(128).nOut(channels).kernelSize(7, 7)
                            .convolutionMode(ConvolutionMode.Same).activation(Activation.LEAKYRELU).build())
                    .layer(6, new ActivationLayer.Builder().activation(Activation.TANH).build())
                    .inputPreProcessor(1,
                            new FeedForwardToCnnPreProcessor(height / 2, width / 2, 128))
                    .inputPreProcessor(6, new CnnToFeedForwardPreProcessor(height, width, channels))
                    .setInputType(InputType.feedForward(latentDim))
                    .build());
        };

        GAN.DiscriminatorProvider discriminatorProvider = (updater) -> {
            return new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
                    .updater(new RmsProp.Builder().learningRate(0.0008).rmsDecay(1e-8).build())
                    //.gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                    //.gradientNormalizationThreshold(100.0)
                    .list()
                    .layer(0, new Convolution2D.Builder().nIn(channels).nOut(64).kernelSize(3, 3)
                            .activation(Activation.LEAKYRELU).build())
                    .layer(1, new Convolution2D.Builder().nIn(64).nOut(64).kernelSize(3, 3).stride(2, 2)
                            .activation(Activation.LEAKYRELU).build())
                    .layer(2, new Convolution2D.Builder().nIn(64).nOut(64).kernelSize(3, 3).stride(2, 2)
                            .activation(Activation.LEAKYRELU).build())
                    .layer(3, new Convolution2D.Builder().nIn(64).nOut(64).kernelSize(3, 3).stride(2, 2)
                            .activation(Activation.LEAKYRELU).build())
                    .layer(4, new DropoutLayer.Builder().dropOut(0.5).build())
                    .layer(5, new DenseLayer.Builder().nIn(64 * 2 * 2).nOut(1).activation(Activation.SIGMOID).build())
                    .layer(6, new LossLayer.Builder().lossFunction(LossFunctions.LossFunction.XENT).build())
                    .inputPreProcessor(0, new FeedForwardToCnnPreProcessor(height, width, channels))
                    .inputPreProcessor(4, new CnnToFeedForwardPreProcessor(2, 2, 64))
                    .setInputType(InputType.convolutionalFlat(height, width, channels))
                    .build());
        };

        GAN gan = new GAN.Builder()
                .generator(genSupplier)
                .discriminator(discriminatorProvider)
                .latentDimension(latentDim)
                //.gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                //.gradientNormalizationThreshold(1.0)
                .updater(new RmsProp.Builder().learningRate(0.0008).rmsDecay(1e-8).build())
                .build();

        gan.getGenerator().setListeners(new PerformanceListener(1, true));
        gan.getDiscriminator().setListeners(new PerformanceListener(1, true));

        Nd4j.getMemoryManager().setAutoGcWindow(15 * 1000);

        int batchSize = 64;
        MnistDataSetIterator trainData = new MnistDataSetIterator(batchSize, true, 42);

        for (int i = 0; i < 10; i++) {
            //gan.fit(trainData, 1);

            System.out.println("Starting epoch: " + (i + 1));

            trainData.reset();
            int j = 0;
            while (trainData.hasNext()) {
                DataSet next = trainData.next();
                gan.fit(next);

                if (j % 1 == 0) {
                    System.out.println("Epoch " + (i + 1) + " iteration " + j + " Visualizing...");
                    INDArray fakeIn = Nd4j.rand(new int[]{batchSize, latentDim});

                    INDArray[] samples = new INDArray[9];
                    for (int k = 0; k < 9; k++) {
                        samples[k] = gan.getGenerator().output(fakeIn.getRow(k), false);
                    }
                    visualize(samples);
                }
                j++;
            }

            System.out.println("Finished epoch: " + (i + 1));
        }
    }
}