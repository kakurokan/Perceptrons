package perceptron;

import java.util.Random;

import static java.lang.Math.exp;

public class Perceptron {
    private final double[] weights;

    public Perceptron(int imagePixels, Random random) {
        weights = new double[imagePixels + 1];
        double r = Math.sqrt(6.0 / (imagePixels + 1));   // Xavier Glorot: N_in = imagePixels, N_out = 1
        for (int j = 0; j < weights.length; j++) {
            weights[j] = (random.nextDouble() * 2 - 1) * r;   // uniforme em [-r, r)
        }
    }

    public Perceptron(double[] weights) {
        this.weights = weights.clone();
    }

    /**
     * Função sigmoide g(z) = 1 / (1 + e^-z).
     */
    public static double sigmoid(double z) {
        return 1.0 / (1.0 + exp(-z));
    }

    public double output(double[] image) {
        if (image.length != weights.length - 1) {
            throw new IllegalArgumentException("Esperava: " + (weights.length - 1) + " pixels na imagem");
        }

        double sum = weights[0];
        for (int i = 0; i < image.length; i++) {
            sum += image[i] * weights[i + 1];
        }
        return sigmoid(sum);
    }

    public int classify(double[] image, double threshold) {
        return output(image) > threshold ? 1 : 0;
    }

    /**
     * Um passo da regra delta para um exemplo (x, t).
     */
    public void update(double[] image, double target, double learningRate) {
        double result = output(image);
        double delta = (target - result) * result * (1 - result);
        double step = learningRate * delta;
        weights[0] += step;

        for (int i = 0; i < image.length; i++) {
            weights[i + 1] += step * image[i];
        }
    }

    public Perceptron copy() {
        return new Perceptron(weights);
    }

    public double[] getWeights() {
        return this.weights.clone();
    }

    public int weightCount() {
        return weights.length;
    }
}
