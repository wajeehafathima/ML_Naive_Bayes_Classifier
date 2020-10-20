package algorithm;

import lombok.extern.slf4j.Slf4j;

import java.io.IOException;

@Slf4j
public class FaceDetector {
    public static void main(String[] args) throws IOException {

        NaiveBayesClassifier naiveBayesClassifier = new NaiveBayesClassifier();
        naiveBayesClassifier.detectFace();

    }
}
