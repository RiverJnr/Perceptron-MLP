package model;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class Text {

    private List<List<Double>> inputs = new ArrayList<>();
    private List<List<Double>> outputs = new ArrayList<>();

    public Text(String fileName, List<Integer> indexOutput) {
        try {
            FileReader file = new FileReader(fileName);
            BufferedReader readFile = new BufferedReader(file);

            String row;
            String[] data;
            while ((row = readFile.readLine()) != null) {
                data = row.split(",", 0);
                List<Double> inDouble = new ArrayList<>();
                List<Double> outDouble = new ArrayList<>();
                for (int i = 0; i < data.length; i++) {
                    if (indexOutput.contains(i)) {
                        if (data[i].equals("N")) {
                            outDouble.add(1.0);
                        } else if (data[i].equals("O")) {
                            outDouble.add(0.0);
                        } else {
                            outDouble.add(Double.parseDouble(data[i]));
                        }

                    } else {
                        inDouble.add(Double.parseDouble(data[i]));
                    }
                }

                this.inputs.add(inDouble);
                this.outputs.add(outDouble);
            }

            file.close();
        } catch (IOException e) {
            System.err.printf("Erro na abertura do arquivo: %s.\n",
                    e.getMessage());
        }
    }

    public List<List<Double>> getInputs() {
        return inputs;
    }

    public List<List<Double>> getOutputs() {
        return outputs;
    }

}
