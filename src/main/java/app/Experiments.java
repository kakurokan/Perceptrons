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
 * A validação é também aumentada uma vez (3 cópias de cada imagem, com seed fixa) para medir
 * a robustez do perceptrão. Essa validação aumentada nunca é usada para treinar nem para
 * escolher o melhor perceptrão.
 * <p>
 * Ficheiros escritos no diretório atual:
 * <ul>
 *   <li>{@code results.csv}: uma linha por treino, com MSE de validação e exatidão na
 *       validação e na validação aumentada;</li>
 *   <li>{@code summary.csv}: média e desvio-padrão de cada configuração sobre as seeds
 *       (MSE de validação, exatidão e exatidão aumentada), ordenado por MSE de validação médio;</li>
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

    /** Número máximo de épocas de cada treino. */
    private static final int MAX_EPOCHS = 200;
    /** Número de épocas seguidas sem melhoria no MSE de validação até parar o treino. */
    private static final int PATIENCE = 20;
    /** Descida mínima do MSE de validação para contar como melhoria. */
    private static final double MIN_DELTA = 1e-4;

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
        double[][] images = CsvUtils.clip01(CsvUtils.readMatrix("data/dataset.csv"));
        double[] targets = CsvUtils.encodeLabels(CsvUtils.readLabels("data/labels.csv"));

        int[] seeds = {1, 2, 3, 4, 5};
        int[] copies = {0, 5, 10};
        double[] rates = {0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0};
        double[] decays = {10, 100, 1e6};

        List<Run> runs = new ArrayList<>();

        // divisão treino/validação feita uma vez: todos os treinos usam a mesma validação
        int[][] split = CsvUtils.trainValidationSplit(images.length, 0.8, new Random(42));
        double[][] trainImages = CsvUtils.selectRows(images, split[0]);
        double[] trainTargets = CsvUtils.selectRows(targets, split[0]);
        double[][] validationImages = CsvUtils.selectRows(images, split[1]);
        double[] validationTargets = CsvUtils.selectRows(targets, split[1]);

        // validação aumentada: só para medir robustez, nunca para escolher nem treinar
        int valCopies = 3;
        Random valAugRandom = new Random(7);
        double[][] validationImagesAug = new double[validationImages.length * valCopies][];
        double[] validationTargetsAug = new double[validationImages.length * valCopies];
        for (int copy = 0; copy < valCopies; copy++) {
            for (int i = 0; i < validationImages.length; i++) {
                validationImagesAug[copy * validationImages.length + i] = Augmentation.randomAugment(validationImages[i], valAugRandom);
                validationTargetsAug[copy * validationImages.length + i] = validationTargets[i];
            }
        }

        // abrir results.csv e escrever o cabeçalho
        try (PrintWriter results = new PrintWriter("results.csv", StandardCharsets.UTF_8)) {
            results.println("seed,copias,a0,decayEpocas,valMSE,valAcc,valAugAcc,melhorEpoca,epocas");

            for (int seed : seeds) {
                for (int k : copies) {
                    Random augRandom = new Random(seed * 1000L + 1);
                    Augmentation.Dataset train = (k == 0)
                            ? new Augmentation.Dataset(trainImages, trainTargets)
                            : Augmentation.augmentDataset(trainImages, trainTargets, k, augRandom);


                    for (double a0 : rates) {
                        for (double de : decays) {
                            Random trainRandom = new Random(seed * 1000L + 2);
                            double c = de * train.x.length;   // épocas -> atualizações
                            Trainer trainer = new Trainer(a0, c, MAX_EPOCHS, PATIENCE, MIN_DELTA);
                            Trainer.TrainingResult r = trainer.train(train, validationImages, validationTargets, trainRandom);

                            double valMse = r.validationMse()[r.bestEpoch()];
                            double acc = accuracy(r.best(), validationImages, validationTargets);
                            double augAcc = accuracy(r.best(), validationImagesAug, validationTargetsAug);
                            
                            results.printf(Locale.ROOT, "%d,%d,%s,%s,%.6f,%.4f,%.4f,%d,%d%n",
                                    seed, k, a0, de, valMse, acc, augAcc,
                                    r.bestEpoch() + 1, r.trainMse().length);

                            // guardar todos os treinos para os agrupar no fim
                            runs.add(new Run(seed, k, a0, de, valMse, acc, augAcc, r));
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
            summary.println("copias,a0,decayEpocas,valMSE_media,valMSE_dp,valAcc_media,valAcc_dp,valAugAcc_media,valAugAcc_dp");
            for (List<Run> g : configs) {
                double[] mse = new double[g.size()];
                double[] acc = new double[g.size()];
                double[] augAcc = new double[g.size()];
                for (int i = 0; i < g.size(); i++) {
                    mse[i] = g.get(i).valMse();
                    acc[i] = g.get(i).acc();
                    augAcc[i] = g.get(i).augAcc();
                }
                summary.printf(Locale.ROOT, "%s,%.6f,%.6f,%.4f,%.4f,%.4f,%.4f%n",
                        g.get(0).config(), mean(mse), std(mse), mean(acc), std(acc), mean(augAcc), std(augAcc));
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
     * @param p       Perceptrão a avaliar. Não pode ser nulo.
     * @param images  Imagens, uma por linha. Não pode ser nulo.
     * @param targets Rótulos (0 ou 1), um por imagem. Não pode ser nulo.
     * @return A fração de imagens bem classificadas, em [0, 1].
     * @pre p != null &amp;&amp; images != null &amp;&amp; targets != null
     * @pre images.length == targets.length &amp;&amp; images.length > 0
     * @pos O perceptrão, as imagens e os rótulos não são modificados.
     */
    private static double accuracy(Perceptron p, double[][] images, double[] targets) {
        int correct = 0;
        for (int i = 0; i < images.length; i++) {
            if (p.classify(images[i], 0.5) == (int) targets[i]) {
                correct++;
            }
        }
        return (double) correct / images.length;
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
     * @param augAcc Exatidão do melhor perceptrão na validação aumentada, em [0, 1].
     * @param result Resultado completo devolvido pelo {@link Trainer}.
     */
    private record Run(int seed, int copies, double a0, double de, double valMse, double acc, double augAcc,
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