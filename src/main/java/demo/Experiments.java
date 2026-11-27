package demo;

import perceptron.Perceptron;
import perceptron.Trainer;
import utils.Augmentation;
import utils.CsvUtils;

import java.io.PrintWriter;
import java.util.*;

public final class Experiments {

    private record Run(int seed, int copies, double a0, double de, double valMse, double acc, Trainer.TrainingResult result) {
        String config() {
            return copies + "," + a0 + "," + de;
        }
    }

    public static void main(String[] args) throws Exception {
        double[][] x = CsvUtils.clip01(CsvUtils.readMatrix("data/dataset.csv"));
        double[] t = CsvUtils.encodeLabels(CsvUtils.readLabels("data/labels.csv"));

        int[] seeds = {1, 2, 3, 4, 5};
        int[] copies = {0, 5, 10};
        double[] rates = {0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0};
        double[] decays = {1, 10, 100};


        List<Run> runs = new ArrayList<>();

        // divisão treino/validação feita uma vez: todos os treinos usam a mesma validação
        int[][] split = CsvUtils.trainValidationSplit(x.length, 0.8, new Random(42));
        double[][] xTr = CsvUtils.selectRows(x, split[0]);
        double[] tTr = CsvUtils.selectRows(t, split[0]);
        double[][] xVal = CsvUtils.selectRows(x, split[1]);
        double[] tVal = CsvUtils.selectRows(t, split[1]);

        // abrir results.csv e escrever o cabeçalho
        try (PrintWriter results = new PrintWriter("results.csv", "UTF-8")) {
            results.println("seed,copias,a0,decayEpocas,valMSE,valAcc,melhorEpoca,epocas");

            for (int seed : seeds) {
                for (int k : copies) {
                    Augmentation.Dataset train = (k == 0)
                            ? new Augmentation.Dataset(xTr, tTr)
                            : Augmentation.augmentDataset(xTr, tTr, k, new Random(seed));


                    for (double a0 : rates) {
                        for (double de : decays) {
                            double c = de * train.x.length;   // épocas -> atualizações
                            Trainer trainer = new Trainer(a0, c, 200, 10, 1e-5);
                            Trainer.TrainingResult r = trainer.train(train, xVal, tVal, new Random(seed));

                            double valMse = r.validationMse()[r.bestEpoch()];
                            double acc = accuracy(r.best(), xVal, tVal);


                            // ... escrever a linha em results.csv
                            results.printf(Locale.ROOT, "%d,%d,%s,%s,%.6f,%.4f,%d,%d%n",
                                    seed, k, a0, de, valMse, acc,
                                    r.bestEpoch() + 1, r.trainMse().length);

                            // guardar todos os treinos para os agrupar no fim
                            runs.add(new Run(seed, k, a0, de, valMse, acc, r));
                        }
                    }
                }
            }
        }

        Map<String, List<Run>> byConfig = new LinkedHashMap<>();
        for(Run run: runs){
            byConfig.computeIfAbsent(run.config(), k -> new ArrayList<>()).add(run);
        }


        List<List<Run>> configs = new ArrayList<>(byConfig.values());
        configs.sort((g1, g2) -> Double.compare(meanMse(g1), meanMse(g2)));

        try (PrintWriter summary = new PrintWriter("summary.csv", "UTF-8")) {
            summary.println("copias,a0,decayEpocas,valMSE_media,valMSE_dp,valAcc_media,valAcc_dp");
            for (List<Run> g : configs) {
                double[] mse = new double[g.size()];
                double[] acc = new double[g.size()];
                for (int i = 0; i < g.size(); i++) {
                    mse[i] = g.get(i).valMse();
                    acc[i] = g.get(i).acc();
                }
                summary.printf(Locale.ROOT, "%s,%.6f,%.6f,%.4f,%.4f%n",
                        g.get(0).config(), mean(mse), std(mse), mean(acc), std(acc));
            }
        }

        // 4. melhor configuração = a primeira; dentro dela, a seed com menor MSE de validação
        List<Run> bestConfig = configs.get(0);
        Run best = bestConfig.get(0);
        for (Run run : bestConfig) {
            if (run.valMse() < best.valMse()) {
                best = run;
            }
        }


        Trainer.TrainingResult r = best.result();
        CsvUtils.writeWeights("weights.csv", r.best().getWeights());

        // escrever mse.csv a partir do melhor treino
        try (PrintWriter mse = new PrintWriter("mse.csv", "UTF-8")) {
            mse.println("epoca,mseTreino,mseValidacao");
            for (int e = 0; e < r.trainMse().length; e++) {
                mse.printf(Locale.ROOT, "%d,%.6f,%.6f%n",
                        e + 1, r.trainMse()[e], r.validationMse()[e]);
            }
        }
        System.out.printf(Locale.ROOT,
                "melhor configuracao: copias=%d a0=%s decayEpocas=%s, valMSE medio=%.4f%n",
                best.copies(), best.a0(), best.de(), meanMse(bestConfig));
        System.out.printf(Locale.ROOT, "melhor seed dessa configuracao: %d, valMSE=%.4f, melhor epoca %d%n",
                best.seed(), best.valMse(), r.bestEpoch() + 1);
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

    /** Media do MSE**/
    private static double meanMse(List<Run> g) {
        double sum = 0;
        for (Run run : g) {
            sum += run.valMse();
        }
        return sum / g.size();
    }

    private static double mean(double[] v) {
        double sum = 0;
        for (double a : v) {
            sum += a;
        }
        return sum / v.length;
    }

    /** Desvio-padrão amostral (divide por n - 1). */
    private static double std(double[] v) {
        if (v.length < 2) {
            return 0;
        }
        double m = mean(v);
        double sum = 0;
        for (double a : v) {
            sum += (a - m) * (a - m);
        }
        return Math.sqrt(sum / (v.length - 1));
    }
}