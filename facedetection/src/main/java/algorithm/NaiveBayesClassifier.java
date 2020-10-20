package algorithm;

import lombok.extern.slf4j.Slf4j;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

@Slf4j
public class NaiveBayesClassifier {

    private static final int TRAINING_DATA_SIZE = 451;
    private static final int TESTING_DATA_SIZE = 150;
    private static final int TOTAL_PIXELS_IN_AN_IMAGE = 4200;
    // setting of smooth constant
    private static final float SMOOTH_K = 1f;
    private static final String TRAINING_DATA = "/faceDataTrain";
    private static final String TRAINING_DATA_OUTPUT = "/faceDataTrainLabels";
    private static final String TESTING_DATA = "/faceDataTest";
    private static final String TESTING_DATA_OUTPUT = "/faceDataTestLabels";


    private int[] readOutput(String fileName, int outputSize) throws IOException {
        BufferedReader br1 = new BufferedReader(new InputStreamReader(getClass().getResourceAsStream(fileName)));
        int[] output = new int[outputSize];
        for (int i = 0; i < outputSize; i++) {
            if (br1.read() == '0') {
                output[i] = 0;

            } else {
                output[i] = 1;

            }
            br1.readLine();
        }
        br1.close();
        return output;
    }


    private int[] getPredictedOutput(float pf, float pnf, char[][] fTest, float[] condSpaceFaceProb,
                                     float[] condHashFaceProb, float[] condSpaceNonProb, float[] condHashNonProb) {
        int[] predictedOutput = new int[TESTING_DATA_SIZE];
        double[] probFace = calculateProbability(pf, condSpaceFaceProb, condHashFaceProb, fTest);
        double[] probNonFace = calculateProbability(pnf, condSpaceNonProb, condHashNonProb, fTest);
        for (int i = 0; i < TESTING_DATA_SIZE; i++) {
            if (probNonFace[i] > probFace[i]) {
                predictedOutput[i] = 0;
            } else {
                predictedOutput[i] = 1;
            }

        }
        return predictedOutput;
    }

    private double[] calculateProbability(float prob, float[] condSpaceProb, float[] condHashProb, char[][] fTest) {
        double[] probability = new double[TESTING_DATA_SIZE];
        for (int i = 0; i < TESTING_DATA_SIZE; i++) {
            probability[i] = Math.log(prob);
            for (int j = 0; j < TOTAL_PIXELS_IN_AN_IMAGE; j++) {
                if (fTest[i][j] == ' ') {
                    probability[i] = probability[i] + Math.log(condSpaceProb[j]);
                } else if (fTest[i][j] == '#') {
                    probability[i] = probability[i] + Math.log(condHashProb[j]);
                }
            }
        }
        return probability;
    }


    private char[][] readData(String fileName) throws IOException {
        int face = 0;
        int f = 0;
        char[][] faceData = new char[TRAINING_DATA_SIZE][TOTAL_PIXELS_IN_AN_IMAGE];
        String sCurrentLine;
        BufferedReader br = new BufferedReader(new InputStreamReader(getClass().getResourceAsStream(fileName)));
        while ((sCurrentLine = br.readLine()) != null) {
            if (f == TOTAL_PIXELS_IN_AN_IMAGE) {
                f = 0;
                face++;
            }

            for (int j = 0; j < sCurrentLine.length(); j++) {
                faceData[face][f] = sCurrentLine.charAt(j);
                f++;
            }
        }
        br.close();
        return faceData;
    }

    private void printConfusionMatrixAndAccuracy(int[] actualOutput, int[] predictedOutput, float smoothk) {
        int i;
        int tn = 0;
        int fp = 0;
        int fn = 0;
        int tp = 0;
        for (i = 0; i < TESTING_DATA_SIZE; i++) {
            if (actualOutput[i] == 0 && predictedOutput[i] == 0) {
                tn++;
            } else if (actualOutput[i] == 0 && predictedOutput[i] == 1) {
                fp++;
                log.info("false positive occurring at i= " + i);
            } else if (actualOutput[i] == 1 && predictedOutput[i] == 0) {
                fn++;
                log.info("false negative occurring at i= " + i);
            } else if (actualOutput[i] == 1 && predictedOutput[i] == 1) {
                tp++;
            }
        }
        log.info(
                "false positive " + fp + " true positive " + tp + " false negative " + fn + " true negative" + tn);
        double accuracy = (double) (tp + tn) / (double) (fp + tp + fn + tn);
        log.info("smoothing constant " + smoothk + " accuracy " + accuracy);
    }


    void detectFace() throws IOException {

        // reading training data file
        // declaring array containing information about pixels of each face
        char[][] ft = readData(TRAINING_DATA);

        // reading label file to calculate probability of being a face and not being a face
        int[] train = readOutput(TRAINING_DATA_OUTPUT, TRAINING_DATA_SIZE);
        int faceCount = 0;
        int nonFaceCount = 0;

        for (int i = 0; i < TRAINING_DATA_SIZE; i++) {
            if (train[i] == 0) {
                nonFaceCount++;
            } else {
                faceCount++;
            }
        }

        float pf = (float) faceCount / (faceCount + nonFaceCount);
        float pnf = (float) nonFaceCount / (faceCount + nonFaceCount);

        // now calculating conditional probabilities
        float[] condHashFaceProb = new float[TOTAL_PIXELS_IN_AN_IMAGE];
        float[] condSpaceFaceProb = new float[TOTAL_PIXELS_IN_AN_IMAGE];
        float[] condHashNonProb = new float[TOTAL_PIXELS_IN_AN_IMAGE];
        float[] condSpaceNonProb = new float[TOTAL_PIXELS_IN_AN_IMAGE];
        int hashFaceCount;
        int spaceFaceCount;
        int hashNonFaceCount;
        int spaceNonFaceCount;
        for (int i = 0; i < TOTAL_PIXELS_IN_AN_IMAGE; i++) {
            hashFaceCount = 0;
            spaceFaceCount = 0;
            hashNonFaceCount = 0;
            spaceNonFaceCount = 0;
            for (int j = 0; j < TRAINING_DATA_SIZE; j++) {
                if (train[j] == 1) {
                    if (ft[j][i] == '#') {
                        hashFaceCount++;
                    } else {
                        spaceFaceCount++;
                    }

                } else if (train[j] == 0) {
                    if (ft[j][i] == '#') {
                        hashNonFaceCount++;
                    } else {
                        spaceNonFaceCount++;
                    }
                }
            }

            condHashFaceProb[i] = (hashFaceCount + SMOOTH_K) / (hashFaceCount + SMOOTH_K + spaceFaceCount + SMOOTH_K);
            condSpaceFaceProb[i] = (spaceFaceCount + SMOOTH_K) / (hashFaceCount + SMOOTH_K + spaceFaceCount + SMOOTH_K);
            condHashNonProb[i] = (hashNonFaceCount + SMOOTH_K) / (hashNonFaceCount + SMOOTH_K + spaceNonFaceCount + SMOOTH_K);
            condSpaceNonProb[i] = (spaceNonFaceCount + SMOOTH_K) / (hashNonFaceCount + SMOOTH_K + spaceNonFaceCount + SMOOTH_K);
        }

        // for testing data

        // reading data from testing file
        char[][] fTest = readData(TESTING_DATA);
        // test labels
        int[] actualOutput = readOutput(TESTING_DATA_OUTPUT, TESTING_DATA_SIZE);
        int[] predictedOutput = getPredictedOutput(pf, pnf, fTest, condSpaceFaceProb, condHashFaceProb,
                condSpaceNonProb, condHashNonProb);

        // confusion matrix
        printConfusionMatrixAndAccuracy(actualOutput, predictedOutput, SMOOTH_K);

    }

}
