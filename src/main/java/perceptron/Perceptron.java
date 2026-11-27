package perceptron;

import java.util.Random;

import static java.lang.Math.exp;

/**
 * Perceptrão de uma só unidade com função de ativação sigmoide, usado para
 * classificação binária de imagens.
 * <p>
 * O vetor de pesos tem um elemento a mais do que o número de pixels: a posição 0
 * guarda o bias e a posição {@code i + 1} guarda o peso do pixel {@code i}.
 * O treino é feito com a regra delta ({@link #update}).
 *
 * @inv weightCount() >= 1
 * @inv weightCount() nunca muda depois da construção.
 */
public class Perceptron {
    /** Pesos do perceptrão: {@code weights[0]} é o bias e {@code weights[i + 1]} o peso do pixel i. */
    private final double[] weights;

    /**
     * Cria um perceptrão com pesos aleatórios, inicializados com Xavier Glorot
     * (uniforme em [-r, r), com r = sqrt(6 / (imagePixels + 1))).
     *
     * @param imagePixels Número de pixels de cada imagem. Não pode ser negativo.
     * @param random      Gerador de números aleatórios usado na inicialização. Não pode ser nulo.
     * @pre imagePixels >= 0
     * @pre random != null
     * @pos weightCount() == imagePixels + 1
     * @pos Cada peso está em [-r, r), com r = sqrt(6 / (imagePixels + 1)).
     */
    public Perceptron(int imagePixels, Random random) {
        weights = new double[imagePixels + 1];
        double r = Math.sqrt(6.0 / (imagePixels + 1));   // Xavier Glorot: N_in = imagePixels, N_out = 1
        for (int j = 0; j < weights.length; j++) {
            weights[j] = (random.nextDouble() * 2 - 1) * r;   // uniforme em [-r, r)
        }
    }

    /**
     * Cria um perceptrão com os pesos indicados.
     *
     * @param weights Pesos a usar, com o bias na posição 0. Não pode ser nulo nem vazio.
     * @pre weights != null &amp;&amp; weights.length >= 1
     * @pos getWeights() tem os mesmos valores que weights.
     * @pos O array weights recebido não é modificado nem partilhado (é copiado).
     */
    public Perceptron(double[] weights) {
        this.weights = weights.clone();
    }

    /**
     * Função sigmoide g(z) = 1 / (1 + e^-z).
     *
     * @param z Valor de entrada.
     * @return O valor da sigmoide em z.
     * @pos 0 &lt;= sigmoid(z) &lt;= 1
     */
    public static double sigmoid(double z) {
        return 1.0 / (1.0 + exp(-z));
    }

    /**
     * Calcula a saída do perceptrão para uma imagem: a sigmoide da soma pesada
     * dos pixels mais o bias.
     *
     * @param image Pixels da imagem. Não pode ser nulo.
     * @return A saída do perceptrão, em [0, 1].
     * @throws IllegalArgumentException se image.length != weightCount() - 1.
     * @pre image != null
     * @pre image.length == weightCount() - 1
     * @pos 0 &lt;= output(image) &lt;= 1
     * @pos Os pesos e a imagem não são modificados.
     */
    public double output(double[] image) {
        if (image.length != weights.length - 1) {
            throw new IllegalArgumentException("Esperava: " + (weights.length - 1) + " pixels na imagem");
        }

        double sum = weights[0];
        for (int i = 0; i < image.length; i++) {
            sum += image[i] * weights[i + 1];
        }
        return sigmoid(sum);
    }

    /**
     * Classifica uma imagem comparando a saída do perceptrão com um limiar.
     *
     * @param image     Pixels da imagem. Não pode ser nulo.
     * @param threshold Limiar de decisão.
     * @return 1 se output(image) &gt; threshold; 0 caso contrário.
     * @throws IllegalArgumentException se image.length != weightCount() - 1.
     * @pre image != null
     * @pre image.length == weightCount() - 1
     * @pos Os pesos e a imagem não são modificados.
     */
    public int classify(double[] image, double threshold) {
        return output(image) > threshold ? 1 : 0;
    }

    /**
     * Um passo da regra delta para um exemplo (x, t): cada peso w_i passa a
     * w_i + learningRate · (t − o) · o · (1 − o) · x_i, com x_0 = 1 para o bias.
     *
     * @param image        Pixels da imagem (x). Não pode ser nulo.
     * @param target       Saída desejada (t), normalmente 0 ou 1.
     * @param learningRate Taxa de aprendizagem. Deve ser positiva.
     * @throws IllegalArgumentException se image.length != weightCount() - 1.
     * @pre image != null
     * @pre image.length == weightCount() - 1
     * @pre learningRate > 0
     * @pos Os pesos são atualizados segundo a regra delta; a imagem não é modificada.
     */
    public void update(double[] image, double target, double learningRate) {
        double result = output(image);
        double delta = (target - result) * result * (1 - result);
        double step = learningRate * delta;
        weights[0] += step;

        for (int i = 0; i < image.length; i++) {
            weights[i + 1] += step * image[i];
        }
    }

    /**
     * Cria uma cópia independente deste perceptrão.
     *
     * @return Um novo perceptrão com os mesmos pesos.
     * @pos Alterações à cópia não afetam este perceptrão, e vice-versa.
     */
    public Perceptron copy() {
        return new Perceptron(weights);
    }

    /**
     * Devolve os pesos do perceptrão.
     *
     * @return Uma cópia dos pesos, com o bias na posição 0.
     * @pos Este perceptrão não é modificado.
     */
    public double[] getWeights() {
        return this.weights.clone();
    }

    /**
     * Devolve o número de pesos do perceptrão, incluindo o bias.
     *
     * @return O número de pixels esperado mais 1.
     * @pos weightCount() >= 1
     */
    public int weightCount() {
        return weights.length;
    }
}
