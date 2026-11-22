package demo;

import utils.Augmentation;
import utils.CsvUtils;

import java.util.Random;

/**
 * Example of how to use {@link CsvUtils} and {@link Augmentation}.
 *
 * <p>Run from the project root (the default paths are {@code data/dataset.csv} and
 * {@code data/labels.csv}; other paths can be given as arguments):</p>
 * <pre>
 * javac -d /tmp/out src/main/java/demo/*.java
 * java -cp /tmp/out demo.Demo
 * </pre>
 * It prints a summary of the data, shows one digit as text, and writes
 * {@code originals.pgm} and {@code augmented.pgm} so that you can look at the augmentations.
 */
public class Demo {

    private Demo() {
    }

    /**
     * Entry point.
     *
     * @param args optional: path of dataset.csv and path of labels.csv
     * @throws Exception if a file cannot be read or written
     */
    public static void main(String[] args) throws Exception {
        String dataPath = args.length > 0 ? args[0] : "data/dataset.csv";
        String labelPath = args.length > 1 ? args[1] : "data/labels.csv";

        // 1. read and prepare the data
        double[][] raw = CsvUtils.readMatrix(dataPath);
        double[][] x = CsvUtils.clip01(raw);
        double[] t = CsvUtils.encodeLabels(CsvUtils.readLabels(labelPath));
        int eights = 0;
        for (double v : t) {
            eights += (int) v;
        }
        System.out.printf("%d images with %d pixels; %d threes (target 0), %d eights (target 1)%n",
                x.length, x[0].length, x.length - eights, eights);

        // 2. random, disjoint training/validation split (80/20)
        Random rng = new Random(42);
        int[][] split = CsvUtils.trainValidationSplit(x.length, 0.8, rng);
        double[][] xTrain = CsvUtils.selectRows(x, split[0]);
        double[] tTrain = CsvUtils.selectRows(t, split[0]);
        double[][] xVal = CsvUtils.selectRows(x, split[1]);
        System.out.printf("training: %d images, validation: %d images%n", xTrain.length, xVal.length);

        // 3. augmentation: originals + 4 random copies of each training image
        Augmentation.Dataset aug = Augmentation.augmentDataset(xTrain, tTrain, 4, new Random(7));
        System.out.printf("augmented training set: %d images%n", aug.x.length);

        // 4. look at the result
        System.out.println("First training image:");
        System.out.print(Augmentation.toAscii(xTrain[0]));
        System.out.println("A rotated version (+15 degrees):");
        System.out.print(Augmentation.toAscii(Augmentation.rotate(xTrain[0], 15)));

        int n = 10;
        double[][] originals = new double[n][];
        double[][] augmented = new double[4 * n][];
        for (int i = 0; i < n; i++) {
            originals[i] = aug.x[i];
            for (int c = 1; c <= 4; c++) {
                augmented[(c - 1) * n + i] = aug.x[c * xTrain.length + i];   // copy c of image i
            }
        }
        Augmentation.savePgmGrid("originals.pgm", originals, n, 4);
        Augmentation.savePgmGrid("augmented.pgm", augmented, n, 4);
        System.out.println("Wrote originals.pgm and augmented.pgm (each column shows the same digit).");
    }
}
