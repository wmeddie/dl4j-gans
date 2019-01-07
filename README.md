# GANs with Deeplearning4j

This repository shows how you can train som sample Generative Adversarial 
Networks (GAN) with in Deeplearning4j.

To run the App example use the following command:

```
mvn compile exec:java -Dexec.mainClass="io.skymind.example.App"
```

This app will just continuously train a simple GAN that uses dense layers and 
display sample outputs from the generator in a window

You should start to see the following output after about 1 hour:

![sample image result](screenshot.png)

The MnistSimpleGAN and MnistDCGAN use a new GAN helper and are 
still works in progress.