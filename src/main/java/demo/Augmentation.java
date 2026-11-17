package demo;

import java.util.Random;

/**
 * Data augmentation for the 20x20 digit images of Problem 4 (Perceptron for 3 or 8).
 *
 * <p>An image is a {@code double[400]} with values in [0, 1], stored row by row: pixel
 * (row r, column c) is at index {@code r * 20 + c}, and row 0 is the top of the image.
 * This is exactly one line of {@code dataset.csv} or {@code evalX.csv}.</p>
 *
 * <p>Every method returns a <em>new</em> image and never modifies its input. Every random
 * method takes a {@link Random}, so experiments are reproducible with a fixed seed.</p>
 *
 * <p>Typical use, creating 4 augmented copies of every training image:</p>
 * <pre>{@code
 * Random rng = new Random(1);
 * demo.Augmentation.Dataset aug = demo.Augmentation.augmentDataset(xTrain, tTrain, 4, rng);
 * // aug.x and aug.t contain the originals followed by the augmented copies
 * }</pre>
 *
 * <p>Augment the training set only. The validation set may be augmented too (with a separate,
 * fixed random seed) to estimate how the perceptron copes with noisy, transformed digits.</p>
 */
public final class Augmentation {

    /** Width and height of the images, in pixels. */
    public static final int SIDE = 20;
    /** Number of pixels per image. */
    public static final int SIZE = SIDE * SIDE;
    /** Centre of the image, used as the centre of rotations and scalings. */
    private static final double CENTER = (SIDE - 1) / 2.0;

    private Augmentation() {
    }

    /**
     * Images and targets after augmentation.
     */
    public static final class Dataset {
        /** Images, one per row. */
        public final double[][] x;
        /** Targets (0 for digit 3, 1 for digit 8), one per image. */
        public final double[] t;

        /**
         * Creates a data set.
         *
         * @param x images
         * @param t targets
         */
        public Dataset(double[][] x, double[] t) {
            this.x = x;
            this.t = t;
        }
    }

    // ------------------------------------------------------------------ geometric transformations

    /**
     * Rotates an image around its centre.
     *
     * @param img     image (400 values)
     * @param degrees rotation angle; positive values rotate counter-clockwise
     * @return rotated image; pixels coming from outside the image are 0 (black)
     */
    public static double[] rotate(double[] img, double degrees) {
        return affine(img, degrees, 1.0, 0.0, 0.0, 0.0);
    }

    /**
     * Shifts an image.
     *
     * @param img image
     * @param dx  horizontal shift in pixels (positive = to the right)
     * @param dy  vertical shift in pixels (positive = down)
     * @return shifted image
     */
    public static double[] translate(double[] img, double dx, double dy) {
        return affine(img, 0.0, 1.0, 0.0, dx, dy);
    }

    /**
     * Scales an image around its centre.
     *
     * @param img    image
     * @param factor scale factor (e.g. 0.9 = 10% smaller, 1.1 = 10% larger)
     * @return scaled image
     */
    public static double[] scale(double[] img, double factor) {
        return affine(img, 0.0, factor, 0.0, 0.0, 0.0);
    }

    /**
     * General affine transformation around the image centre: shear, then scale, then rotate,
     * then translate. Uses bilinear interpolation.
     *
     * @param img     image (400 values)
     * @param degrees rotation angle (positive = counter-clockwise)
     * @param factor  scale factor (1 = no scaling)
     * @param shear   horizontal shear (0 = none; 0.2 slants the digit noticeably)
     * @param dx      horizontal shift in pixels (positive = right)
     * @param dy      vertical shift in pixels (positive = down)
     * @return transformed image
     */
    public static double[] affine(double[] img, double degrees, double factor, double shear, double dx, double dy) {
        checkImage(img);
        if (factor <= 0) {
            throw new IllegalArgumentException("scale factor must be positive: " + factor);
        }
        // Forward map in screen coordinates (x to the right, y down) relative to the centre:
        //   p' = F p + d, with F = R * S * H.
        // In screen coordinates a counter-clockwise rotation by a uses [[cos, sin], [-sin, cos]].
        double a = Math.toRadians(degrees);
        double cos = Math.cos(a);
        double sin = Math.sin(a);
        // H = [[1, shear], [0, 1]], S = factor * I
        double f00 = factor * cos;
        double f01 = factor * (cos * shear + sin);
        double f10 = -factor * sin;
        double f11 = factor * (-sin * shear + cos);
        double det = f00 * f11 - f01 * f10;
        // inverse of F
        double i00 = f11 / det;
        double i01 = -f01 / det;
        double i10 = -f10 / det;
        double i11 = f00 / det;
        double[] out = new double[SIZE];
        for (int r = 0; r < SIDE; r++) {
            for (int c = 0; c < SIDE; c++) {
                double x = c - CENTER - dx;
                double y = r - CENTER - dy;
                double sx = i00 * x + i01 * y + CENTER;   // source column
                double sy = i10 * x + i11 * y + CENTER;   // source row
                out[r * SIDE + c] = bilinear(img, sy, sx);
            }
        }
        return out;
    }

    /**
     * Elastic distortion (Simard et al., 2003): every pixel is displaced by a smooth random
     * field, which imitates the natural variations of handwriting.
     *
     * @param img   image
     * @param alpha maximum displacement in pixels (1 to 2 gives realistic distortions)
     * @param sigma smoothness of the field, in pixels (2.5 to 4; smaller = more wobbly)
     * @param rng   random number generator
     * @return distorted image
     */
    public static double[] elasticDistortion(double[] img, double alpha, double sigma, Random rng) {
        checkImage(img);
        double[] fx = randomField(sigma, rng);
        double[] fy = randomField(sigma, rng);
        double[] out = new double[SIZE];
        for (int r = 0; r < SIDE; r++) {
            for (int c = 0; c < SIDE; c++) {
                int k = r * SIDE + c;
                out[k] = bilinear(img, r + alpha * fy[k], c + alpha * fx[k]);
            }
        }
        return out;
    }

    // ------------------------------------------------------------------ noise

    /**
     * Adds Gaussian noise to every pixel and clips the result to [0, 1].
     *
     * @param img   image
     * @param sigma standard deviation of the noise (0.05 = light, 0.2 = strong)
     * @param rng   random number generator
     * @return noisy image
     */
    public static double[] gaussianNoise(double[] img, double sigma, Random rng) {
        checkImage(img);
        double[] out = new double[SIZE];
        for (int k = 0; k < SIZE; k++) {
            out[k] = clip(img[k] + sigma * rng.nextGaussian());
        }
        return out;
    }

    /**
     * Salt-and-pepper noise: each pixel is, with the given probability, replaced by a random
     * grey level in [0, 1].
     *
     * @param img         image
     * @param probability fraction of pixels to replace (e.g. 0.05)
     * @param rng         random number generator
     * @return noisy image
     */
    public static double[] saltAndPepper(double[] img, double probability, Random rng) {
        checkImage(img);
        double[] out = img.clone();
        for (int k = 0; k < SIZE; k++) {
            if (rng.nextDouble() < probability) {
                out[k] = rng.nextDouble();
            }
        }
        return out;
    }

    // ------------------------------------------------------------------ stroke thickness

    /**
     * Makes the strokes thicker: each pixel takes the maximum of its 3x3 neighbourhood
     * (grey-level dilation), blended with the original.
     *
     * @param img      image
     * @param strength 0 = no change, 1 = full dilation (0.5 is usually enough)
     * @return image with thicker strokes
     */
    public static double[] thicken(double[] img, double strength) {
        return morph(img, strength, true);
    }

    /**
     * Makes the strokes thinner: each pixel takes the minimum of its 3x3 neighbourhood
     * (grey-level erosion), blended with the original.
     *
     * @param img      image
     * @param strength 0 = no change, 1 = full erosion (0.5 is usually enough; thin digits may vanish at 1)
     * @return image with thinner strokes
     */
    public static double[] thin(double[] img, double strength) {
        return morph(img, strength, false);
    }

    // ------------------------------------------------------------------ random combination

    /**
     * Applies a random combination of the transformations above, with moderate default ranges:
     * <ul>
     *   <li>affine: rotation in [-15, 15] degrees, scale in [0.9, 1.1], shear in [-0.15, 0.15],
     *       shift in [-1.5, 1.5] pixels;</li>
     *   <li>elastic distortion with probability 0.5 (alpha in [1, 2], sigma in [2.5, 4]);</li>
     *   <li>thicker strokes with probability 0.15, thinner with probability 0.15;</li>
     *   <li>Gaussian noise (sigma in [0.05, 0.2]) with probability 0.3, or salt-and-pepper
     *       noise (2% to 8% of the pixels) with probability 0.2 (never both).</li>
     * </ul>
     * Experiment with these ranges: they are a starting point, not an optimum.
     *
     * @param img image
     * @param rng random number generator
     * @return augmented image, values in [0, 1]
     */
    public static double[] randomAugment(double[] img, Random rng) {
        double[] out = affine(img,
                uniform(rng, -15, 15),
                uniform(rng, 0.9, 1.1),
                uniform(rng, -0.15, 0.15),
                uniform(rng, -1.5, 1.5),
                uniform(rng, -1.5, 1.5));
        if (rng.nextDouble() < 0.5) {
            out = elasticDistortion(out, uniform(rng, 1.0, 2.0), uniform(rng, 2.5, 4.0), rng);
        }
        double p = rng.nextDouble();
        if (p < 0.15) {
            out = thicken(out, 0.5);
        } else if (p < 0.30) {
            out = thin(out, 0.5);
        }
        p = rng.nextDouble();
        if (p < 0.3) {
            out = gaussianNoise(out, uniform(rng, 0.05, 0.2), rng);
        } else if (p < 0.5) {
            out = saltAndPepper(out, uniform(rng, 0.02, 0.08), rng);
        }
        return clip(out);
    }

    /**
     * Builds an augmented training set: the original images followed by {@code copies}
     * randomly augmented versions of each of them. Targets are copied accordingly.
     *
     * @param x      original images (not modified)
     * @param t      targets, one per image
     * @param copies number of augmented copies per image (0 returns a copy of the originals)
     * @param rng    random number generator
     * @return augmented data set with {@code x.length * (copies + 1)} images
     */
    public static Dataset augmentDataset(double[][] x, double[] t, int copies, Random rng) {
        if (x.length != t.length) {
            throw new IllegalArgumentException("x has " + x.length + " rows but t has " + t.length);
        }
        if (copies < 0) {
            throw new IllegalArgumentException("copies must be >= 0: " + copies);
        }
        int n = x.length;
        double[][] ax = new double[n * (copies + 1)][];
        double[] at = new double[n * (copies + 1)];
        for (int i = 0; i < n; i++) {
            ax[i] = x[i].clone();
            at[i] = t[i];
        }
        for (int c = 1; c <= copies; c++) {
            for (int i = 0; i < n; i++) {
                ax[c * n + i] = randomAugment(x[i], rng);
                at[c * n + i] = t[i];
            }
        }
        return new Dataset(ax, at);
    }

    // ------------------------------------------------------------------ utilities

    /**
     * Clips every pixel to [0, 1].
     *
     * @param img image
     * @return clipped copy
     */
    public static double[] clip(double[] img) {
        double[] out = new double[img.length];
        for (int k = 0; k < img.length; k++) {
            out[k] = clip(img[k]);
        }
        return out;
    }

    /**
     * Returns the image as text, one character per pixel (useful for quick debugging):
     * ' ' for dark pixels, then '.', ':', '+', '#' for increasingly bright ones.
     *
     * @param img image
     * @return 20 lines of 20 characters
     */
    public static String toAscii(double[] img) {
        checkImage(img);
        final String levels = " .:+#";
        StringBuilder sb = new StringBuilder();
        for (int r = 0; r < SIDE; r++) {
            for (int c = 0; c < SIDE; c++) {
                double v = clip(img[r * SIDE + c]);
                sb.append(levels.charAt(Math.min(levels.length() - 1, (int) (v * levels.length()))));
            }
            sb.append('\n');
        }
        return sb.toString();
    }

    /**
     * Saves images as a grid in a PGM file (a simple greyscale format that GIMP, IrfanView,
     * Python and most image viewers can open). Useful to inspect augmentations or for the report.
     *
     * @param path     output file, e.g. {@code "augmented.pgm"}
     * @param images   images to save
     * @param columns  number of images per row of the grid
     * @param zoom     size of each pixel on screen (e.g. 4)
     * @throws java.io.IOException if the file cannot be written
     */
    public static void savePgmGrid(String path, double[][] images, int columns, int zoom) throws java.io.IOException {
        int rows = (images.length + columns - 1) / columns;
        int gap = 1;
        int w = columns * (SIDE + gap) * zoom;
        int h = rows * (SIDE + gap) * zoom;
        byte[] pixels = new byte[w * h];
        java.util.Arrays.fill(pixels, (byte) 128);   // grey separators
        for (int n = 0; n < images.length; n++) {
            checkImage(images[n]);
            int top = (n / columns) * (SIDE + gap) * zoom;
            int left = (n % columns) * (SIDE + gap) * zoom;
            for (int y = 0; y < SIDE * zoom; y++) {
                for (int x = 0; x < SIDE * zoom; x++) {
                    double v = clip(images[n][(y / zoom) * SIDE + x / zoom]);
                    pixels[(top + y) * w + left + x] = (byte) Math.round(v * 255);
                }
            }
        }
        try (java.io.OutputStream out = new java.io.BufferedOutputStream(new java.io.FileOutputStream(path))) {
            out.write(("P5\n" + w + " " + h + "\n255\n").getBytes(java.nio.charset.StandardCharsets.US_ASCII));
            out.write(pixels);
        }
    }

    // ------------------------------------------------------------------ private helpers

    /** Bilinear interpolation at a real-valued position; pixels outside the image count as 0. */
    private static double bilinear(double[] img, double row, double col) {
        int r0 = (int) Math.floor(row);
        int c0 = (int) Math.floor(col);
        double fr = row - r0;
        double fc = col - c0;
        return (1 - fr) * (1 - fc) * pixel(img, r0, c0)
                + (1 - fr) * fc * pixel(img, r0, c0 + 1)
                + fr * (1 - fc) * pixel(img, r0 + 1, c0)
                + fr * fc * pixel(img, r0 + 1, c0 + 1);
    }

    private static double pixel(double[] img, int r, int c) {
        return (r < 0 || r >= SIDE || c < 0 || c >= SIDE) ? 0.0 : img[r * SIDE + c];
    }

    /** Smooth random field with values in [-1, 1] (uniform noise, Gaussian blur, rescaled). */
    private static double[] randomField(double sigma, Random rng) {
        double[] f = new double[SIZE];
        for (int k = 0; k < SIZE; k++) {
            f[k] = 2 * rng.nextDouble() - 1;
        }
        f = gaussianBlur(f, sigma);
        double max = 1e-12;
        for (double v : f) {
            max = Math.max(max, Math.abs(v));
        }
        for (int k = 0; k < SIZE; k++) {
            f[k] /= max;
        }
        return f;
    }

    /** Separable Gaussian blur; borders are handled by repeating the edge values. */
    private static double[] gaussianBlur(double[] img, double sigma) {
        int radius = Math.max(1, (int) Math.ceil(3 * sigma));
        double[] kernel = new double[2 * radius + 1];
        double sum = 0;
        for (int i = -radius; i <= radius; i++) {
            kernel[i + radius] = Math.exp(-i * i / (2 * sigma * sigma));
            sum += kernel[i + radius];
        }
        for (int i = 0; i < kernel.length; i++) {
            kernel[i] /= sum;
        }
        double[] tmp = new double[SIZE];
        double[] out = new double[SIZE];
        for (int r = 0; r < SIDE; r++) {
            for (int c = 0; c < SIDE; c++) {
                double s = 0;
                for (int i = -radius; i <= radius; i++) {
                    int cc = Math.min(SIDE - 1, Math.max(0, c + i));
                    s += kernel[i + radius] * img[r * SIDE + cc];
                }
                tmp[r * SIDE + c] = s;
            }
        }
        for (int r = 0; r < SIDE; r++) {
            for (int c = 0; c < SIDE; c++) {
                double s = 0;
                for (int i = -radius; i <= radius; i++) {
                    int rr = Math.min(SIDE - 1, Math.max(0, r + i));
                    s += kernel[i + radius] * tmp[rr * SIDE + c];
                }
                out[r * SIDE + c] = s;
            }
        }
        return out;
    }

    private static double[] morph(double[] img, double strength, boolean dilate) {
        checkImage(img);
        double[] out = new double[SIZE];
        for (int r = 0; r < SIDE; r++) {
            for (int c = 0; c < SIDE; c++) {
                double v = img[r * SIDE + c];
                double m = v;
                for (int i = -1; i <= 1; i++) {
                    for (int j = -1; j <= 1; j++) {
                        double p = pixel(img, r + i, c + j);
                        m = dilate ? Math.max(m, p) : Math.min(m, p);
                    }
                }
                out[r * SIDE + c] = (1 - strength) * v + strength * m;
            }
        }
        return out;
    }

    private static double uniform(Random rng, double lo, double hi) {
        return lo + (hi - lo) * rng.nextDouble();
    }

    private static double clip(double v) {
        return v < 0 ? 0 : (v > 1 ? 1 : v);
    }

    private static void checkImage(double[] img) {
        if (img == null || img.length != SIZE) {
            throw new IllegalArgumentException("an image must have " + SIZE + " values, got "
                    + (img == null ? "null" : img.length));
        }
    }
}
