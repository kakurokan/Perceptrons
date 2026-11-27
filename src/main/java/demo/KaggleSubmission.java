package demo;

import perceptron.Perceptron;
import utils.Augmentation;
import utils.CsvUtils;

/**
 * Kaggle submission of Problem 4 (Perceptron for 3 or 8).
 *
 * <p>This program does not train. It loads the weights of the best perceptron (chosen in the
 * experiments and saved with {@link CsvUtils#writeWeights}), classifies every image of
 * {@code evalX.csv} and writes the submission file with header {@code id,label}, where
 * {@code id} is the row index in {@code evalX.csv} (starting at 0) and {@code label} is
 * 0 (digit 3) or 1 (digit 8), obtained by thresholding the perceptron output at 0.5.</p>
 *
 * <p>Usage:</p>
 * <pre>
 * java -cp out demo.KaggleSubmission weights.csv data/evalX.csv submission.csv
 * </pre>
 *
 * <p>Test before the Kaggle window, using the labelled data as if it were the evaluation set
 * (the fourth argument is optional and only used to print the accuracy):</p>
 * <pre>
 * java -cp out demo.KaggleSubmission weights.csv data/dataset.csv test.csv data/labels.csv
 * </pre>
 */
public final class KaggleSubmission {

    /** Number of weights: one per pixel plus w0. */
    private static final int N_WEIGHTS = Augmentation.SIZE + 1;

    private KaggleSubmission() {
    }

    /**
     * Entry point.
     *
     * @param args weights file, evaluation file, output file and, optionally, a label file
     *             to check the accuracy when testing with labelled data
     * @throws Exception if a file cannot be read or written
     */
    public static void main(String[] args) throws Exception {
        String weightsPath = args.length > 0 ? args[0] : "weights.csv";
        String evalPath = args.length > 1 ? args[1] : "data/evalX.csv";
        String outPath = args.length > 2 ? args[2] : "submission.csv";
        String labelPath = args.length > 3 ? args[3] : null;

        // 1. load the best perceptron
        double[] weights = CsvUtils.readWeights(weightsPath);
        if (weights.length != N_WEIGHTS) {
            throw new IllegalArgumentException(weightsPath + ": expected " + N_WEIGHTS
                    + " weights but found " + weights.length);
        }
        Perceptron p = new Perceptron(weights);

        // 2. read the evaluation images and prepare them exactly like the training data
        double[][] x = CsvUtils.clip01(CsvUtils.readMatrix(evalPath));
        checkSize(x, evalPath);

        // 3. classify: output g(z), then threshold at 0.5 (never submit the raw outputs)
        double[] o = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            o[i] = p.output(x[i]);
        }
        int[] labels = CsvUtils.threshold(o);

        // 4. write the submission
        CsvUtils.writeSubmission(outPath, labels);
        int eights = 0;
        for (int label : labels) {
            eights += label;
        }
        System.out.printf("wrote %s: %d images, %d classified as 8 and %d as 3%n",
                outPath, labels.length, eights, labels.length - eights);

        // optional: accuracy, when testing with labelled data
        if (labelPath != null) {
            double[] t = CsvUtils.encodeLabels(CsvUtils.readLabels(labelPath));
            if (t.length != labels.length) {
                throw new IllegalArgumentException(labelPath + " has " + t.length
                        + " labels but " + evalPath + " has " + labels.length + " images");
            }
            int correct = 0;
            for (int i = 0; i < labels.length; i++) {
                if (labels[i] == (int) t[i]) {
                    correct++;
                }
            }
            System.out.printf("accuracy: %.2f%%%n", 100.0 * correct / labels.length);
        }
    }

    /**
     * Checks that every row has exactly 400 pixels (for example, that the file has no id column).
     *
     * @param x    images
     * @param path file name, for the error message
     * @throws IllegalArgumentException if a row does not have 400 values
     */
    private static void checkSize(double[][] x, String path) {
        for (int i = 0; i < x.length; i++) {
            if (x[i].length != Augmentation.SIZE) {
                throw new IllegalArgumentException(path + ", row " + (i + 1) + ": expected "
                        + Augmentation.SIZE + " pixels but found " + x[i].length);
            }
        }
    }
}