import java.util.ArrayList;
import java.util.List;
import model.PerceptronMLP;
import model.Text;

public class PerceptronRunner {

    public static void main(String[] args) {
        List<Integer> indexOutput = new ArrayList<>();
        // And
        // indexOutput.add(2);
        // Text dataBase = new Text("And.txt", indexOutput);
        // Or
        // indexOutput.add(2);
        // Text dataBase = new Text("Or.txt", indexOutput);
        // Xor
        // indexOutput.add(2);
        // Text dataBase = new Text("Xor.txt", indexOutput);
        // Robo
        // indexOutput.add(3);
        // indexOutput.add(4);
        // Text dataBase = new Text("Robo.txt", indexOutput);
        // Fertility
        indexOutput.add(9);
        Text dataBase = new Text("fertility_Diagnosis.txt", indexOutput);

        int entradasDoPerceptron = dataBase.getInputs().get(0).size();
        int arrayTeta = dataBase.getOutputs().get(0).size();
        double ni = 0.3;
        int qtdNeuronioIntermediario = 8;
        PerceptronMLP neuronio = new PerceptronMLP(entradasDoPerceptron, qtdNeuronioIntermediario, arrayTeta, ni);

        double erroEpoca;
        double erroAmostra;
        // Percorre cada época
        for (int i = 0; i < 10000; i++) {
            erroEpoca = 0;
            // Percorre as amostras
            for (int j = 0; j < dataBase.getInputs().size(); j++) {
                erroAmostra = 0;
                Double[] Xamostras = dataBase.getInputs().get(j).toArray(new Double[dataBase.getInputs().get(0).size()]);
                Double[] Yamostras = dataBase.getOutputs().get(j).toArray(new Double[dataBase.getOutputs().get(0).size()]);
                double[] teta = neuronio.learn(Xamostras, Yamostras);
                for (int k = 0; k < teta.length; k++) {
                    erroAmostra += Math.abs(Yamostras[k] - teta[k]);
                }
                erroEpoca += erroAmostra;
            }
            System.out.println("Época: " + (i + 1) + " - Erro: " + erroEpoca);
        }
        //pesosFinal(neuronio);
    }

    public static void pesosFinal(PerceptronMLP perceptron) {
        double[][] pesos = perceptron.getPesosSaida();
        int linha = 0;
        int coluna = 0;
        for (int j = 0; j < pesos[0].length; j++) {
            System.out.printf("\nColuna %d\n", coluna + 1);
            for (int i = 0; i < pesos.length; i++) {
                System.out.printf("W%d%d: %f\n", linha++, coluna, pesos[i][j]);
            }
            linha = 0;
            coluna++;
        }
    }

}
