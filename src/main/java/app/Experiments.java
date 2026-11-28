package app;

import perceptron.Perceptron;
import perceptron.Trainer;
import utils.Augmentation;
import utils.CsvUtils;

import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.util.*;

/**
 * Experiências de treino do perceptrão que distingue o dígito 3 do 8.
 * <p>
 * Divide {@code data/dataset.csv} uma única vez em treino (80%) e validação (20%) e treina um
 * perceptrão para cada combinação de seed, número de cópias aumentadas por imagem, taxa de
 * aprendizagem inicial e decaimento (em épocas). Todos os treinos usam a mesma validação.
 * <p>
 * Ficheiros escritos no diretório atual:
 * <ul>
 *   <li>{@code results.csv}: uma linha por treino;</li>
 *   <li>{@code summary.csv}: média e desvio-padrão de cada configuração sobre as seeds,
 *       ordenado por MSE de validação médio;</li>
 *   <li>{@code weights.csv}: pesos do melhor perceptrão, lidos depois por {@link KaggleSubmission};</li>
 *   <li>{@code mse.csv}: MSE de treino e de validação por época do melhor treino.</li>
 * </ul>
 * A melhor configuração é a de menor MSE de validação médio; dentro dela, escolhe-se a seed
 * com menor MSE de validação.
 * <p>
 * Uso:
 * <pre>
 * java -cp out app.Experiments
 * </pre>
 */
public final class Experiments {

    /**
     * Classe utilitária: não é instanciável.
     */
    private Experiments() {
    }

    /**
     * Ponto de entrada: corre todas as experiências e escreve os ficheiros de resultados.
     *
     * @param args Ignorados.
     * @throws Exception se um ficheiro não puder ser lido ou escrito.
     * @pre {@code data/dataset.csv} e {@code data/labels.csv} existem e têm o mesmo número de linhas.
     * @pos {@code results.csv}, {@code summary.csv}, {@code weights.csv} e {@code mse.csv} são (re)escritos.
     */
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
        try (PrintWriter results = new PrintWriter("results.csv", StandardCharsets.UTF_8)) {
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
        for (Run run : runs) {
            byConfig.computeIfAbsent(run.config(), k -> new ArrayList<>()).add(run);
        }


        List<List<Run>> configs = new ArrayList<>(byConfig.values());
        configs.sort(Comparator.comparingDouble(Experiments::meanMse));

        try (PrintWriter summary = new PrintWriter("summary.csv", StandardCharsets.UTF_8)) {
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
        try (PrintWriter mse = new PrintWriter("mse.csv", StandardCharsets.UTF_8)) {
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

    /**
     * Calcula a exatidão de um perceptrão com limiar 0.5.
     *
     * @param p Perceptrão a avaliar. Não pode ser nulo.
     * @param x Imagens, uma por linha. Não pode ser nulo.
     * @param t Rótulos (0 ou 1), um por imagem. Não pode ser nulo.
     * @return A fração de imagens bem classificadas, em [0, 1].
     * @pre p != null &amp;&amp; x != null &amp;&amp; t != null
     * @pre x.length == t.length &amp;&amp; x.length > 0
     * @pos O perceptrão, as imagens e os rótulos não são modificados.
     */
    private static double accuracy(Perceptron p, double[][] x, double[] t) {
        int correct = 0;
        for (int i = 0; i < x.length; i++) {
            if (p.classify(x[i], 0.5) == (int) t[i]) {
                correct++;
            }
        }
        return (double) correct / x.length;
    }

    /**
     * Calcula a média do MSE de validação de um grupo de treinos.
     *
     * @param g Treinos de uma mesma configuração. Não pode ser nulo nem vazio.
     * @return A média de {@code valMse()} dos treinos.
     * @pre g != null &amp;&amp; !g.isEmpty()
     * @pos A lista não é modificada.
     */
    private static double meanMse(List<Run> g) {
        double sum = 0;
        for (Run run : g) {
            sum += run.valMse();
        }
        return sum / g.size();
    }

    /**
     * Calcula a média aritmética de um vetor.
     *
     * @param v Valores. Não pode ser nulo nem vazio.
     * @return A média dos valores.
     * @pre v != null &amp;&amp; v.length > 0
     * @pos O vetor não é modificado.
     */
    private static double mean(double[] v) {
        double sum = 0;
        for (double a : v) {
            sum += a;
        }
        return sum / v.length;
    }

    /**
     * Calcula o desvio-padrão amostral de um vetor (divide por n - 1).
     *
     * @param v Valores. Não pode ser nulo.
     * @return O desvio-padrão amostral, ou 0 se v tiver menos de 2 valores.
     * @pre v != null
     * @pos std(v) >= 0
     * @pos O vetor não é modificado.
     */
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

    /**
     * Resultado de um treino.
     *
     * @param seed   Seed usada na aumentação e no treino.
     * @param copies Número de cópias aumentadas por imagem de treino (0 = sem aumentação).
     * @param a0     Taxa de aprendizagem inicial.
     * @param de     Número de épocas ao fim do qual a taxa fica a metade.
     * @param valMse MSE de validação na melhor época.
     * @param acc    Exatidão de validação do melhor perceptrão, em [0, 1].
     * @param result Resultado completo devolvido pelo {@link Trainer}.
     */
    private record Run(int seed, int copies, double a0, double de, double valMse, double acc,
                       Trainer.TrainingResult result) {
        /**
         * Identifica a configuração deste treino, sem a seed.
         *
         * @return {@code "copies,a0,de"}, usado para agrupar treinos e como prefixo das linhas de {@code summary.csv}.
         */
        String config() {
            return copies + "," + a0 + "," + de;
        }
    }
}