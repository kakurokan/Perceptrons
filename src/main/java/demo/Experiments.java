package demo;

import perceptron.Trainer;
import utils.Augmentation;
import utils.CsvUtils;

import java.util.Random;

public final class Experiments {
    public static void main(String[] args) throws Exception {
        double[][] x = CsvUtils.clip01(CsvUtils.readMatrix("data/dataset.csv"));
        double[] t = CsvUtils.encodeLabels(CsvUtils.readLabels("data/labels.csv"));

        int[] seeds = {1, 2, 3, 4, 5};
        int[] copies = {0, 5, 10};
        double[] rates = {0.01, 0.05, 0.1, 0.5, 1.0};
        double[] decays = {1000, 10000};

        Trainer.TrainingResult best = null;
        // ... abrir results.csv e escrever o cabeçalho

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
                        // ... guardar em "best" se for o melhor até agora
                    }
                }
            }
        }

        CsvUtils.writeWeights("weights.csv", best.best().getWeights());
        // ... escrever mse.csv a partir de best.trainMse() e best.validationMse()
    }
}