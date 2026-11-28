package app;

import perceptron.Perceptron;
import utils.Augmentation;
import utils.CsvUtils;

/**
 * Submissão ao Kaggle do Problema 4 (perceptrão que distingue o dígito 3 do 8).
 * <p>
 * Este programa não treina. Carrega os pesos do melhor perceptrão (escolhido em
 * {@link Experiments} e guardado com {@link CsvUtils#writeWeights}), classifica todas as
 * imagens de {@code evalX.csv} e escreve o ficheiro de submissão com cabeçalho
 * {@code id,label}, onde {@code id} é o índice da linha em {@code evalX.csv} (a começar em 0)
 * e {@code label} é 0 (dígito 3) ou 1 (dígito 8), obtido com limiar 0.5 na saída do perceptrão.
 * <p>
 * Uso:
 * <pre>
 * java -cp out app.KaggleSubmission weights.csv data/evalX.csv submission.csv
 * </pre>
 * Para testar antes da janela do Kaggle, usa-se o conjunto rotulado como se fosse o de
 * avaliação (o quarto argumento é opcional e só serve para mostrar a exatidão):
 * <pre>
 * java -cp out app.KaggleSubmission weights.csv data/dataset.csv test.csv data/labels.csv
 * </pre>
 */
public final class KaggleSubmission {

    /** Número de pesos do perceptrão: um por pixel mais o bias. */
    private static final int N_WEIGHTS = Augmentation.SIZE + 1;

    /** Classe utilitária: não é instanciável. */
    private KaggleSubmission() {
    }

    /**
     * Ponto de entrada: carrega os pesos, classifica as imagens de avaliação e escreve a submissão.
     *
     * @param args Argumentos opcionais, por ordem: ficheiro de pesos (por omissão {@code weights.csv}),
     *             ficheiro de avaliação ({@code data/evalX.csv}), ficheiro de saída
     *             ({@code submission.csv}) e ficheiro de rótulos para calcular a exatidão (nenhum).
     * @throws Exception                se um ficheiro não puder ser lido ou escrito.
     * @throws IllegalArgumentException se o ficheiro de pesos não tiver {@code Augmentation.SIZE + 1}
     *                                  pesos, se uma imagem não tiver {@code Augmentation.SIZE} pixels
     *                                  ou se o número de rótulos for diferente do número de imagens.
     * @pre args != null
     * @pos O ficheiro de saída contém uma linha {@code id,label} por imagem, com label em {0, 1}.
     * @pos Os ficheiros de entrada não são modificados.
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
     * Verifica que todas as imagens têm exatamente {@code Augmentation.SIZE} pixels
     * (por exemplo, que o ficheiro não tem uma coluna de id).
     *
     * @param x    Imagens, uma por linha. Não pode ser nulo.
     * @param path Nome do ficheiro, usado na mensagem de erro.
     * @throws IllegalArgumentException se alguma linha não tiver {@code Augmentation.SIZE} valores.
     * @pre x != null
     * @pos As imagens não são modificadas.
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