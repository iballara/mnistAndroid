package ignasi.mnistandroid;

import android.content.res.AssetManager;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

public class Classifier {

    // Only returns if at least this confidence
    private static final float THRESHOLD = 0.1f;

    private TensorFlowInferenceInterface tfHelper;

    private String inputName;
    private String outputName;
    private int inputSize;

    private List<String> labels;
    private float[] output;
    private String[] outputNames;

    private static List<String> readLabels(AssetManager am, String fileName) throws IOException {

        final BufferedReader br = new BufferedReader(new InputStreamReader(am.open(fileName)));

        String line;
        final List<String> labels = new ArrayList<>();
        while ((line = br.readLine()) != null) {
            labels.add(line);
        }

        br.close();
        return labels;
    }


    public static Classifier create(AssetManager assetManager, String modelPath, String labelPath,
                             int inputSize, String inputName, String outputName)
                             throws IOException {

        final Classifier classifier = new Classifier();

        classifier.inputName = inputName;
        classifier.outputName = outputName;

        // Read labels
        final String labelFile = labelPath.split("file:///android_asset/")[1];
        classifier.labels = readLabels(assetManager, labelFile);

        classifier.tfHelper = new TensorFlowInferenceInterface();
        if (classifier.tfHelper.initializeTensorFlow(assetManager, modelPath) != 0) {
            throw new RuntimeException("TF initialization failed");
        }

        int numClasses = 10;

        classifier.inputSize = inputSize;
        // Pre-allocate buffer.
        classifier.outputNames = new String[]{ outputName };
        classifier.outputName = outputName;
        classifier.output = new float[numClasses];

        return classifier;
    }

    public Classification recognize(final float[] pixels) {

        tfHelper.fillNodeFloat(inputName, new int[]{inputSize * inputSize}, pixels);
        tfHelper.runInference(outputNames);

        tfHelper.readNodeFloat(outputName, output);

        // Find the best classification
        Classification result = new Classification();
        for (int i = 0; i < output.length; ++i) {
            System.out.println(output[i]);
            System.out.println(labels.get(i));
            if (output[i] > THRESHOLD && output[i] > result.getConf()) {
                result.update(output[i], labels.get(i));
            }
        }

        return result;
    }
}
