package perceptron;

import utils.Augmentation;
import utils.CsvUtils;

import java.util.Arrays;
import java.util.Random;

/**
 * Treina um {@link Perceptron} com a regra delta, em modo estocástico (um exemplo de cada vez).
 * <p>
 * Em cada época:
 * <ol>
 *   <li>percorre os exemplos de treino por uma ordem aleatória, aplicando
 *       {@link Perceptron#update} a cada um com uma taxa de aprendizagem que decai
 *       com o número de atualizações: initialRate · decaySteps / (decaySteps + atualizações);</li>
 *   <li>calcula o MSE nos conjuntos de treino e de validação;</li>
 *   <li>se o MSE de validação melhorou mais do que minDelta em relação ao melhor até agora,
 *       guarda uma cópia do perceptrão; caso contrário, conta mais uma época sem melhorar.</li>
 * </ol>
 * O treino termina ao fim de maxEpochs ou quando passam patience épocas seguidas sem
 * melhoria (early stopping).
 *
 * @inv initialRate > 0, decaySteps > 0, maxEpochs > 0, patience > 0, minDelta >= 0
 */
public class Trainer {
    /** Taxa de aprendizagem na primeira atualização. */
    private final double initialRate;
    /** Número de atualizações ao fim do qual a taxa de aprendizagem fica a metade da inicial. */
    private final double decaySteps;
    /** Número máximo de épocas de treino. */
    private final int maxEpochs;
    /** Número de épocas seguidas sem melhoria no MSE de validação até parar o treino. */
    private final int patience;
    /** Descida mínima do MSE de validação para contar como melhoria. */
    private final double minDelta;


    /**
     * Cria um treinador com os parâmetros indicados.
     *
     * @param initialRate Taxa de aprendizagem inicial. Tem de ser positiva.
     * @param decaySteps  Número de atualizações ao fim do qual a taxa fica a metade. Tem de ser positivo.
     * @param maxEpochs   Número máximo de épocas. Tem de ser positivo.
     * @param patience    Épocas seguidas sem melhoria até parar. Tem de ser positivo.
     * @param minDelta    Descida mínima do MSE de validação para contar como melhoria. Não pode ser negativa.
     * @pre initialRate > 0 &amp;&amp; decaySteps > 0 &amp;&amp; maxEpochs > 0 &amp;&amp; patience > 0 &amp;&amp; minDelta >= 0
     */
    public Trainer(double initialRate, double decaySteps, int maxEpochs, int patience, double minDelta) {
        this.initialRate = initialRate;
        this.decaySteps = decaySteps;
        this.maxEpochs = maxEpochs;
        this.patience = patience;
        this.minDelta = minDelta;
    }

    /**
     * MSE = (1/N) · soma de (t − o)². Só lê o perceptrão, nunca chama update.
     *
     * @param perceptron Perceptrão a avaliar. Não pode ser nulo.
     * @param images     Imagens do conjunto. Não pode ser nulo nem vazio.
     * @param targets    Saídas desejadas, uma por imagem. Não pode ser nulo.
     * @return O erro quadrático médio do perceptrão no conjunto.
     * @pre perceptron != null &amp;&amp; images != null &amp;&amp; targets != null
     * @pre images.length > 0 &amp;&amp; images.length == targets.length
     * @pos mse(...) >= 0
     * @pos O perceptrão, as imagens e os alvos não são modificados.
     */
    static double mse(Perceptron perceptron, double[][] images, double[] targets) {
        double sum = 0;
        for (int i = 0; i < images.length; i++) {
            double error = targets[i] - perceptron.output(images[i]);
            sum += Math.pow(error, 2);
        }
        return (1.0 / images.length) * sum;
    }

    /**
     * Treina um perceptrão novo, com pesos aleatórios, e devolve o que teve menor
     * MSE de validação.
     *
     * @param train             Conjunto de treino (imagens em train.x e alvos em train.t). Não pode ser nulo.
     * @param validationImages  Imagens do conjunto de validação. Não pode ser nulo nem vazio.
     * @param validationTargets Saídas desejadas do conjunto de validação, uma por imagem. Não pode ser nulo.
     * @param random            Gerador usado para inicializar os pesos e baralhar os exemplos. Não pode ser nulo.
     * @return O resultado do treino: o melhor perceptrão, o MSE de treino e de validação
     *         de cada época e a época (a contar de 0) em que o melhor foi obtido.
     * @pre train != null &amp;&amp; validationImages != null &amp;&amp; validationTargets != null &amp;&amp; random != null
     * @pre validationImages.length > 0 &amp;&amp; validationImages.length == validationTargets.length
     * @pre Todas as imagens (de treino e de validação) têm o mesmo número de pixels.
     * @pos 1 &lt;= result.trainMse().length == result.validationMse().length &lt;= maxEpochs
     * @pos 0 &lt;= result.bestEpoch() &lt; result.trainMse().length
     * @pos O conjunto de treino e o de validação não são modificados.
     */
    public TrainingResult train(Augmentation.Dataset train, double[][] validationImages, double[] validationTargets, Random random) {
        Perceptron trainPerceptron = new Perceptron(validationImages[0].length, random);

        double[] trainMse = new double[maxEpochs];
        double[] validationMse = new double[maxEpochs];

        Perceptron bestPerceptron = null;
        double bestMse = Double.POSITIVE_INFINITY;
        int bestEpoch = 0, epochsWithoutImprovement = 0, totalEpochs = 0;
        long updates = 0;

        for (int epoch = 0; epoch < maxEpochs; epoch++) {
            int[] order = CsvUtils.shuffledIndices(train.x.length, random);
            for (int j : order) {
                double learningRate = initialRate * decaySteps / (decaySteps + updates);
                trainPerceptron.update(train.x[j], train.t[j], learningRate);
                updates++;
            }
            //   3. guardar mse(perceptron, train.x, train.t) e mse(perceptron, xVal, tVal)
            trainMse[epoch] = mse(trainPerceptron, train.x, train.t);
            validationMse[epoch] = mse(trainPerceptron, validationImages, validationTargets);
            totalEpochs = epoch + 1;

            if (validationMse[epoch] < bestMse - minDelta) {
                //        guardar best = perceptron.copy(), bestMse, bestEpoch; semMelhorar = 0
                bestPerceptron = trainPerceptron.copy();
                bestMse = validationMse[epoch];
                bestEpoch = epoch;
                epochsWithoutImprovement = 0;
            } else {
                epochsWithoutImprovement++;
                if (epochsWithoutImprovement >= patience) {
                    break;
                }
            }
        }

        return new TrainingResult(bestPerceptron,
                Arrays.copyOf(trainMse, totalEpochs),
                Arrays.copyOf(validationMse, totalEpochs),
                bestEpoch);
    }

    /**
     * Resultado de um treino.
     *
     * @param best          Cópia do perceptrão com menor MSE de validação.
     * @param trainMse      MSE no conjunto de treino ao fim de cada época executada.
     * @param validationMse MSE no conjunto de validação ao fim de cada época executada.
     * @param bestEpoch     Época (a contar de 0) em que {@code best} foi obtido.
     * @inv trainMse.length == validationMse.length
     * @inv 0 &lt;= bestEpoch &lt; trainMse.length
     */
    public record TrainingResult(Perceptron best, double[] trainMse, double[] validationMse, int bestEpoch) {
    }
}
