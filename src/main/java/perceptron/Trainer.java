package perceptron;

import utils.Augmentation;

import java.util.Random;

public class Trainer {
    private final double initialRate;
    private final double decaySteps;
    private final int maxEpochs;
    private final int patience;


    public Trainer(double initialRate, double decaySteps, int maxEpochs, int patience) {
        this.initialRate = initialRate;
        this.decaySteps = decaySteps;
        this.maxEpochs = maxEpochs;
        this.patience = patience;
    }

    /**
     * MSE = (1/N) · soma de (t − o)². Só lê o perceptrão, nunca chama update.
     */
    static double mse(Perceptron perceptron, double[][] images, double[] targets) {
        double sum = 0;
        for (int i = 0; i < images.length; i++) {
            double error = targets[i] - perceptron.output(images[i]);
            sum += Math.pow(error, 2);
        }
        return (1.0 / images.length) * sum;
    }

    public TrainingResult train(Augmentation.Dataset train, double[][] validationImages, double[] validationTargets, Random random) {
        return null;
    }

    public record TrainingResult(Perceptron best, double[] trainMse, double[] validationMse, int bestEpoch) {
    }
}
