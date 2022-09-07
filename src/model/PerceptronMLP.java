package model;

import java.util.Random;

public class PerceptronMLP {

    private double ni;
    private double[][] pesosSaida;
    private double[][] pesosIntermediario;
    private int qtdNeuronioIntermediario;

    public PerceptronMLP(int qtdAmostra, int qtdNeuronioIntermediario, int arrayTeta, Double ni) {
        this.ni = ni;
        this.qtdNeuronioIntermediario = qtdNeuronioIntermediario;
        this.pesosIntermediario = new double[qtdAmostra + 1][qtdNeuronioIntermediario];
        this.pesosSaida = new double[qtdNeuronioIntermediario + 1][arrayTeta];

        Random rand = new Random();
        for (int j = 0; j < this.qtdNeuronioIntermediario; j++) {
            for (int i = 0; i < qtdAmostra + 1; i++) {
                double w = rand.nextDouble() * 0.3 * 2 - 0.3;
                this.pesosIntermediario[i][j] = w;
            }
        }

        for (int j = 0; j < arrayTeta; j++) {
            for (int i = 0; i < this.qtdNeuronioIntermediario + 1; i++) {
                double w = rand.nextDouble() * 0.3 * 2 - 0.3;
                this.pesosSaida[i][j] = w;
            }
        }
    }

    public double[] learn(Double[] Xamostras, Double[] Yamostras) {
        double sum;
        // Vetor x é o que guarda os valores da amostra
        double[] x = new double[Xamostras.length + 1];
        x[x.length - 1] = 1;
        for (int i = 0; i < x.length - 1; i++) {
            x[i] = Xamostras[i];
        }

        // Vetor H é o que guarda o somatorio dos pesos * amostras
        double[] hidden = new double[this.qtdNeuronioIntermediario + 1];

        for (int j = 0; j < this.qtdNeuronioIntermediario; j++) {
            sum = 0;
            for (int i = 0; i < x.length; i++) { // Somatorio dos pesos
                sum += x[i] * pesosIntermediario[i][j];
            }
            hidden[j] = this.sigmoidal(sum);
        }
        hidden[this.qtdNeuronioIntermediario] = 1;

        double[] teta = new double[Yamostras.length];
        // Percorre cada coluna da saida Y
        for (int j = 0; j < teta.length; j++) {
            sum = 0;
            // Percorre cada linha da saida Y
            for (int i = 0; i < hidden.length; i++) {
                sum += hidden[i] * pesosSaida[i][j];
            }
            teta[j] = this.sigmoidal(sum);
        }

        double[] deltaTeta = new double[Yamostras.length];
        for (int j = 0; j < deltaTeta.length; j++) {
            deltaTeta[j] = teta[j] * (1 - teta[j]) * (Yamostras[j] - teta[j]);
        }

        double[] deltaHidden = new double[this.qtdNeuronioIntermediario];
        for (int j = 0; j < deltaHidden.length; j++) {
            sum = 0;
            for (int i = 0; i < Yamostras.length; i++) {
                sum += deltaTeta[i] * pesosSaida[j][i];
            }
            deltaHidden[j] = hidden[j] * (1 - hidden[j]) * sum;
        }

        for (int j = 0; j < this.pesosIntermediario.length; j++) {
            for (int i = 0; i < this.pesosIntermediario[0].length; i++) {
                pesosIntermediario[j][i] += ni * deltaHidden[i] * x[j];
            }
        }

        for (int j = 0; j < this.pesosSaida.length; j++) {
            for (int i = 0; i < pesosSaida[0].length; i++) {
                pesosSaida[j][i] += ni * deltaTeta[i] * hidden[j];
            }
        }

        return teta;
    }

    public double[][] getPesosSaida() {
        return this.pesosSaida;
    }

    public double[][] getPesosIntermediario() {
        return this.pesosIntermediario;
    }

    private double sigmoidal(double sum) {
        return 1 / (1 + Math.exp(-sum));
    }
}