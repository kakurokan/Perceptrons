package demo;

import perceptron.Perceptron;
import perceptron.Trainer;
import utils.Augmentation;
import utils.CsvUtils;

import java.io.PrintWriter;
import java.util.Locale;
import java.util.Random;

public final class Experiments {
    public static void main(String[] args) throws Exception {
        double[][] x = CsvUtils.clip01(CsvUtils.readMatrix("data/dataset.csv"));
        double[] t = CsvUtils.encodeLabels(CsvUtils.readLabels("data/labels.csv"));

        int[] seeds = {1, 2, 3, 4, 5};
        int[] copies = {0, 5, 10};
        double[] rates = {0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0};
        double[] decays = {1000, 10000, 100000};

        Trainer.TrainingResult best = null;
        double bestMse = Double.POSITIVE_INFINITY;
        String bestConfig = "";


        // ... abrir results.csv e escrever o cabeçalho
        try (PrintWriter results = new PrintWriter("results.csv", "UTF-8")) {
            results.println("Seed, copias, a0, c, valMSE, valAcc, melhorEpoca, epocas");

            for (int seed : seeds) {
                int[][] split = CsvUtils.trainValidationSplit(x.length, 0.8, new Random(seed));
                double[][] xTr = CsvUtils.selectRows(x, split[0]);
                double[] tTr = CsvUtils.selectRows(t, split[0]);
                double[][] xVal = CsvUtils.selectRows(x, split[1]);
                double[] tVal = CsvUtils.selectRows(t, split[1]);

                for (int k : copies) {
                    Augmentation.Dataset train = (k == 0)
                            ? new Augmentation.Dataset(xTr, tTr)
                            : Augmentation.augmentDataset(xTr, tTr, k, new Random(seed));


                    for (double a0 : rates) {
                        for (double c : decays) {
                            Trainer trainer = new Trainer(a0, c, 200, 10, 1e-5);
                            Trainer.TrainingResult r = trainer.train(train, xVal, tVal, new Random(seed));

                            double valMse = r.validationMse()[r.bestEpoch()];
                            double acc = accuracy(r.best(), xVal, tVal);


                            // ... escrever a linha em results.csv
                            results.printf(Locale.ROOT, "%d,%d,%s,%s,%.6f,%.4f,%d,%d%n",
                                    seed, k, a0, c, valMse, acc,
                                    r.bestEpoch() + 1, r.trainMse().length);

                            // ... guardar em "best" se for o melhor até agora
                            if (valMse < bestMse) {
                                bestMse = valMse;
                                best = r;
                                bestConfig = "seed=" + seed + " copias=" + k + " a0=" + a0 + " c=" + c;
                            }
                         }
                       }
                }
            }
        }

        CsvUtils.writeWeights("weights.csv", best.best().getWeights());
        // ... escrever mse.csv a partir de best.trainMse() e best.validationMse()
        try (PrintWriter mse = new PrintWriter("mse.csv", "UTF-8")) {
            mse.println("epoca,mseTreino,mseValidacao");
            for (int e = 0; e < best.trainMse().length; e++) {
                mse.printf(Locale.ROOT, "%d,%.6f,%.6f%n",
                        e + 1, best.trainMse()[e], best.validationMse()[e]);
            }
        }
        System.out.printf(Locale.ROOT, "melhor: %s, valMSE=%.4f, melhor epoca %d%n",
                bestConfig, bestMse, best.bestEpoch() + 1);
    }
    private static double accuracy(Perceptron p, double[][] x, double[] t) {
        int correct = 0;
        for (int i = 0; i < x.length; i++) {
            if (p.classify(x[i], 0.5) == (int) t[i]) {
                correct++;
            }
        }
        return (double) correct / x.length;
    }
}