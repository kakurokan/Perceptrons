package perceptron;

import utils.Augmentation;
import utils.CsvUtils;

import java.util.Arrays;
import java.util.Random;

public class Trainer {
    private final double initialRate;
    private final double decaySteps;
    private final int maxEpochs;
    private final int patience;
    private final double minDelta;


    public Trainer(double initialRate, double decaySteps, int maxEpochs, int patience, double minDelta) {
        this.initialRate = initialRate;
        this.decaySteps = decaySteps;
        this.maxEpochs = maxEpochs;
        this.patience = patience;
        this.minDelta = minDelta;
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
        Perceptron trainPerceptron = new Perceptron(validationImages[0].length, random);

        double[] trainMse = new double[maxEpochs];
        double[] validationMse = new double[maxEpochs];

        Perceptron bestPerceptron = null;
        double bestMse = Double.POSITIVE_INFINITY;
        int bestEpoch = 0, epochsWithoutImprovement = 0, totalEpochs = 0;
        long updates = 0;

        for (int epoch = 0; epoch < maxEpochs; epoch++) {
            int[] order = CsvUtils.shuffledIndices(train.x.length, random);
            for (int j : order) {
                double learningRate = initialRate * decaySteps / (decaySteps + updates);
                trainPerceptron.update(train.x[j], train.t[j], learningRate);
                updates++;
            }
            //   3. guardar mse(perceptron, train.x, train.t) e mse(perceptron, xVal, tVal)
            trainMse[epoch] = mse(trainPerceptron, train.x, train.t);
            validationMse[epoch] = mse(trainPerceptron, validationImages, validationTargets);
            totalEpochs = epoch + 1;

            if (validationMse[epoch] < bestMse - minDelta) {
                //        guardar best = perceptron.copy(), bestMse, bestEpoch; semMelhorar = 0
                bestPerceptron = trainPerceptron.copy();
                bestMse = validationMse[epoch];
                bestEpoch = epoch;
                epochsWithoutImprovement = 0;
            } else {
                epochsWithoutImprovement++;
                if (epochsWithoutImprovement >= patience) {
                    break;
                }
            }
        }

        return new TrainingResult(bestPerceptron,
                Arrays.copyOf(trainMse, totalEpochs),
                Arrays.copyOf(validationMse, totalEpochs),
                bestEpoch);
    }

    public record TrainingResult(Perceptron best, double[] trainMse, double[] validationMse, int bestEpoch) {
    }
}
