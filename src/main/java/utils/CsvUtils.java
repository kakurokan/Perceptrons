package utils;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Helper functions for Problem 4 (Perceptron for 3 or 8): reading the CSV files,
 * preparing inputs and targets, splitting the data, and writing the Kaggle submission.
 *
 * <p>Typical use:</p>
 * <pre>{@code
 * double[][] x = utils.CsvUtils.clip01(utils.CsvUtils.readMatrix("dataset.csv"));
 * double[]   t = utils.CsvUtils.encodeLabels(utils.CsvUtils.readLabels("labels.csv"));
 * int[][] split = utils.CsvUtils.trainValidationSplit(x.length, 0.8, new Random(42));
 * double[][] xTrain = utils.CsvUtils.selectRows(x, split[0]);
 * double[]   tTrain = utils.CsvUtils.selectRows(t, split[0]);
 * double[][] xVal   = utils.CsvUtils.selectRows(x, split[1]);
 * double[]   tVal   = utils.CsvUtils.selectRows(t, split[1]);
 * }</pre>
 *
 * <p>All methods are static; the class cannot be instantiated.</p>
 */
public final class CsvUtils {

    /** Digit encoded as target 0 (mandatory encoding, used by the Kaggle evaluation). */
    public static final int DIGIT_ZERO = 3;
    /** Digit encoded as target 1 (mandatory encoding, used by the Kaggle evaluation). */
    public static final int DIGIT_ONE = 8;

    private CsvUtils() {
    }

    /**
     * Reads a numeric CSV file into a matrix, one row per line.
     *
     * <p>Values are separated by commas. Empty lines are ignored. If the first line is not
     * numeric (for example a header such as {@code id,label}), it is skipped. All rows must
     * have the same number of columns.</p>
     *
     * @param path path of the CSV file (e.g. {@code "dataset.csv"} or {@code "evalX.csv"})
     * @return matrix with {@code rows x columns} values
     * @throws IOException              if the file cannot be read
     * @throws IllegalArgumentException if a value is not a number or rows have different lengths
     */
    public static double[][] readMatrix(String path) throws IOException {
        List<double[]> rows = new ArrayList<>();
        int lineNumber = 0;
        int columns = -1;
        try (BufferedReader in = Files.newBufferedReader(Paths.get(path), StandardCharsets.UTF_8)) {
            String line;
            while ((line = in.readLine()) != null) {
                lineNumber++;
                line = line.trim();
                if (line.isEmpty()) {
                    continue;
                }
                if (lineNumber == 1 && line.startsWith("﻿")) {   // UTF-8 byte order mark
                    line = line.substring(1);
                }
                String[] parts = line.split(",", -1);
                if (rows.isEmpty() && !isNumeric(parts[0].trim())) {
                    continue;                                          // header line
                }
                if (columns < 0) {
                    columns = parts.length;
                } else if (parts.length != columns) {
                    throw new IllegalArgumentException(path + ", line " + lineNumber + ": expected "
                            + columns + " columns but found " + parts.length);
                }
                double[] row = new double[parts.length];
                for (int j = 0; j < parts.length; j++) {
                    try {
                        row[j] = Double.parseDouble(parts[j].trim());
                    } catch (NumberFormatException e) {
                        throw new IllegalArgumentException(path + ", line " + lineNumber + ", column "
                                + (j + 1) + ": not a number: \"" + parts[j] + "\"", e);
                    }
                }
                rows.add(row);
            }
        }
        return rows.toArray(new double[0][]);
    }

    /**
     * Reads a label file with one integer label per line (e.g. {@code labels.csv}, values 3 or 8).
     * If the file has several columns, the last column is used; a header line is skipped.
     *
     * @param path path of the label file
     * @return the labels, in file order
     * @throws IOException              if the file cannot be read
     * @throws IllegalArgumentException if a label is not an integer
     */
    public static int[] readLabels(String path) throws IOException {
        double[][] m = readMatrix(path);
        int[] labels = new int[m.length];
        for (int i = 0; i < m.length; i++) {
            double v = m[i][m[i].length - 1];
            if (v != Math.rint(v)) {
                throw new IllegalArgumentException(path + ", row " + (i + 1) + ": label is not an integer: " + v);
            }
            labels[i] = (int) v;
        }
        return labels;
    }

    /**
     * Converts digit labels into perceptron targets: 3 becomes 0 and 8 becomes 1.
     *
     * @param digits labels as read from {@code labels.csv}
     * @return targets in {0, 1}
     * @throws IllegalArgumentException if a label is neither 3 nor 8
     */
    public static double[] encodeLabels(int[] digits) {
        double[] t = new double[digits.length];
        for (int i = 0; i < digits.length; i++) {
            if (digits[i] == DIGIT_ZERO) {
                t[i] = 0.0;
            } else if (digits[i] == DIGIT_ONE) {
                t[i] = 1.0;
            } else {
                throw new IllegalArgumentException("row " + (i + 1) + ": unexpected label " + digits[i]
                        + " (expected " + DIGIT_ZERO + " or " + DIGIT_ONE + ")");
            }
        }
        return t;
    }

    /**
     * Returns a copy of the matrix with every value clipped to [0, 1]
     * (values below 0 become 0, values above 1 become 1).
     *
     * <p>{@code dataset.csv} contains a few values slightly outside [0, 1] because of image
     * resampling; apply this method to both the training data and the evaluation data.</p>
     *
     * @param x input matrix (not modified)
     * @return clipped copy
     */
    public static double[][] clip01(double[][] x) {
        double[][] out = new double[x.length][];
        for (int i = 0; i < x.length; i++) {
            out[i] = new double[x[i].length];
            for (int j = 0; j < x[i].length; j++) {
                out[i][j] = Math.min(1.0, Math.max(0.0, x[i][j]));
            }
        }
        return out;
    }

    /**
     * Returns the indices 0..n-1 in random order (Fisher-Yates shuffle).
     *
     * @param n   number of indices
     * @param rng random number generator (use a fixed seed for reproducible experiments)
     * @return shuffled indices
     */
    public static int[] shuffledIndices(int n, Random rng) {
        int[] idx = new int[n];
        for (int i = 0; i < n; i++) {
            idx[i] = i;
        }
        for (int i = n - 1; i > 0; i--) {
            int j = rng.nextInt(i + 1);
            int tmp = idx[i];
            idx[i] = idx[j];
            idx[j] = tmp;
        }
        return idx;
    }

    /**
     * Splits the indices 0..n-1 into disjoint, randomized training and validation sets.
     *
     * <p>Remember that {@code labels.csv} is sorted (all 3s first, then all 8s), so the split
     * must be random, never "the first 80% of the rows".</p>
     *
     * @param n             number of examples
     * @param trainFraction fraction used for training, e.g. 0.8
     * @param rng           random number generator
     * @return {@code result[0]} = training indices, {@code result[1]} = validation indices
     */
    public static int[][] trainValidationSplit(int n, double trainFraction, Random rng) {
        if (trainFraction <= 0 || trainFraction >= 1) {
            throw new IllegalArgumentException("trainFraction must be in (0, 1): " + trainFraction);
        }
        int[] idx = shuffledIndices(n, rng);
        int nTrain = (int) Math.round(n * trainFraction);
        int[] train = new int[nTrain];
        int[] val = new int[n - nTrain];
        System.arraycopy(idx, 0, train, 0, nTrain);
        System.arraycopy(idx, nTrain, val, 0, n - nTrain);
        return new int[][] {train, val};
    }

    /**
     * Selects rows of a matrix (the rows are copied).
     *
     * @param x       matrix
     * @param indices rows to select, in the desired order
     * @return new matrix with the selected rows
     */
    public static double[][] selectRows(double[][] x, int[] indices) {
        double[][] out = new double[indices.length][];
        for (int i = 0; i < indices.length; i++) {
            out[i] = x[indices[i]].clone();
        }
        return out;
    }

    /**
     * Selects elements of a vector.
     *
     * @param t       vector (e.g. targets)
     * @param indices elements to select, in the desired order
     * @return new vector with the selected elements
     */
    public static double[] selectRows(double[] t, int[] indices) {
        double[] out = new double[indices.length];
        for (int i = 0; i < indices.length; i++) {
            out[i] = t[indices[i]];
        }
        return out;
    }

    /**
     * Converts perceptron outputs g(z) into class labels: 1 (digit 8) if the output is
     * greater than 0.5, 0 (digit 3) otherwise.
     *
     * @param outputs perceptron outputs in [0, 1]
     * @return labels in {0, 1}
     */
    public static int[] threshold(double[] outputs) {
        int[] labels = new int[outputs.length];
        for (int i = 0; i < outputs.length; i++) {
            labels[i] = outputs[i] > 0.5 ? 1 : 0;
        }
        return labels;
    }

    /**
     * Writes a Kaggle submission file with header {@code id,label}. The id of each row is its
     * index in {@code evalX.csv}, starting at 0.
     *
     * @param path   output file, e.g. {@code "submission.csv"}
     * @param labels predicted labels in {0, 1}, in the order of the rows of {@code evalX.csv}
     * @throws IOException              if the file cannot be written
     * @throws IllegalArgumentException if a label is neither 0 nor 1
     */
    public static void writeSubmission(String path, int[] labels) throws IOException {
        Path p = Paths.get(path);
        try (BufferedWriter out = Files.newBufferedWriter(p, StandardCharsets.UTF_8)) {
            out.write("id,label");
            out.newLine();
            for (int i = 0; i < labels.length; i++) {
                if (labels[i] != 0 && labels[i] != 1) {
                    throw new IllegalArgumentException("row " + i + ": label must be 0 or 1, got " + labels[i]);
                }
                out.write(i + "," + labels[i]);
                out.newLine();
            }
        }
    }

    /**
     * Writes the weights of a perceptron to a file, one weight per line, starting with w0.
     * The values are written with full precision, so {@link #readWeights} gives back exactly
     * the same weights.
     *
     * @param path    output file, e.g. {@code "weights.csv"}
     * @param weights weights w0, w1, ..., wn
     * @throws IOException if the file cannot be written
     */
    public static void writeWeights(String path, double[] weights) throws IOException {
        try (BufferedWriter out = Files.newBufferedWriter(Paths.get(path), StandardCharsets.UTF_8)) {
            for (double w : weights) {
                out.write(Double.toString(w));
                out.newLine();
            }
        }
    }

    /**
     * Reads the weights written by {@link #writeWeights}. A file with all the weights on a
     * single line, separated by commas, is also accepted.
     *
     * @param path weights file
     * @return weights w0, w1, ..., wn
     * @throws IOException              if the file cannot be read
     * @throws IllegalArgumentException if the file is empty or is not a single row or column
     */
    public static double[] readWeights(String path) throws IOException {
        double[][] m = readMatrix(path);
        if (m.length == 0) {
            throw new IllegalArgumentException(path + ": no weights found");
        }
        if (m.length == 1) {
            return m[0];                       // all weights on one line
        }
        if (m[0].length != 1) {
            throw new IllegalArgumentException(path + ": expected one weight per line or all on one line");
        }
        double[] w = new double[m.length];
        for (int i = 0; i < m.length; i++) {
            w[i] = m[i][0];
        }
        return w;
    }

    private static boolean isNumeric(String s) {
        try {
            Double.parseDouble(s);
            return true;
        } catch (NumberFormatException e) {
            return false;
        }
    }
}
