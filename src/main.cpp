/*
 * NeuralNet++ - A Simple Feed-Forward Neural Network in C++
 * 
 * This is a learning-focused implementation showing OOP composition:
 *   Network → Layers → Neurons
 * 
 * Features: Training with backpropagation, XOR demo, save/load weights
 */

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace std;

// ============================================================================
// MathUtils - Helper functions for neural network math
// ============================================================================
class MathUtils {
public:
    // Sigmoid squashes any value to range (0, 1)
    static double sigmoid(double x) {
        return 1.0 / (1.0 + exp(-x));
    }

    // Derivative of sigmoid (used in backpropagation)
    // Note: takes the OUTPUT of sigmoid, not the input
    static double sigmoidDerivative(double sigmoidOutput) {
        return sigmoidOutput * (1.0 - sigmoidOutput);
    }

    // Random weight between -1 and 1
    static double randomWeight() {
        static mt19937 rng(random_device{}());
        static uniform_real_distribution<double> dist(-1.0, 1.0);
        return dist(rng);
    }
};

// ============================================================================
// Sample - One training example (inputs and expected outputs)
// ============================================================================
struct Sample {
    vector<double> inputs;
    vector<double> targets;
};

// ============================================================================
// Neuron - A single node in the network
// ============================================================================
class Neuron {
public:
    double value = 0.0;                      // Current activation value
    double bias = 0.0;                       // Bias term (added before activation)
    double gradient = 0.0;                   // Error gradient (for training)
    vector<double> weights;             // Weights to next layer neurons

    // Create neuron with random weights connecting to next layer
    Neuron(int numOutputs, bool hasBias = true) {
        bias = hasBias ? MathUtils::randomWeight() : 0.0;
        weights.resize(numOutputs);
        for (double& w : weights) {
            w = MathUtils::randomWeight();
        }
    }
};

// ============================================================================
// Layer - A collection of neurons at the same depth
// ============================================================================
class Layer {
public:
    vector<Neuron> neurons;

    // Create a layer with specified neurons, each connected to next layer
    Layer(int numNeurons, int numOutputs, bool isInputLayer = false) {
        for (int i = 0; i < numNeurons; i++) {
            neurons.emplace_back(numOutputs, !isInputLayer);
        }
    }
};

// ============================================================================
// NeuralNetwork - The main engine (holds layers, does training)
// ============================================================================
class NeuralNetwork {
public:
    double learningRate;

    // Create network from topology (e.g., {2, 3, 1} = 2 inputs, 3 hidden, 1 output)
    NeuralNetwork(const vector<int>& topology, double lr = 0.5) 
        : learningRate(lr) {
        for (size_t i = 0; i < topology.size(); i++) {
            int numOutputs = (i < topology.size() - 1) ? topology[i + 1] : 0;
            bool isInput = (i == 0);
            layers.emplace_back(topology[i], numOutputs, isInput);
        }
    }

    // Forward pass: compute output from inputs
    vector<double> predict(const vector<double>& inputs) {
        // Set input layer values
        for (size_t i = 0; i < inputs.size(); i++) {
            layers[0].neurons[i].value = inputs[i];
        }

        // Propagate through each layer
        for (size_t L = 1; L < layers.size(); L++) {
            Layer& prevLayer = layers[L - 1];
            Layer& currLayer = layers[L];

            for (size_t j = 0; j < currLayer.neurons.size(); j++) {
                double sum = currLayer.neurons[j].bias;
                
                // Sum weighted inputs from previous layer
                for (size_t i = 0; i < prevLayer.neurons.size(); i++) {
                    sum += prevLayer.neurons[i].value * prevLayer.neurons[i].weights[j];
                }
                
                currLayer.neurons[j].value = MathUtils::sigmoid(sum);
            }
        }

        // Collect output values
        vector<double> outputs;
        for (const Neuron& n : layers.back().neurons) {
            outputs.push_back(n.value);
        }
        return outputs;
    }

    // Backward pass: compute gradients and update weights
    void backpropagate(const vector<double>& targets) {
        Layer& outputLayer = layers.back();

        // Step 1: Calculate output layer gradients
        for (size_t i = 0; i < outputLayer.neurons.size(); i++) {
            double output = outputLayer.neurons[i].value;
            double error = targets[i] - output;
            outputLayer.neurons[i].gradient = error * MathUtils::sigmoidDerivative(output);
        }

        // Step 2: Calculate hidden layer gradients (back to front)
        for (int L = layers.size() - 2; L > 0; L--) {
            Layer& currLayer = layers[L];
            Layer& nextLayer = layers[L + 1];

            for (size_t i = 0; i < currLayer.neurons.size(); i++) {
                double errorSum = 0.0;
                for (size_t j = 0; j < nextLayer.neurons.size(); j++) {
                    errorSum += currLayer.neurons[i].weights[j] * nextLayer.neurons[j].gradient;
                }
                double output = currLayer.neurons[i].value;
                currLayer.neurons[i].gradient = errorSum * MathUtils::sigmoidDerivative(output);
            }
        }

        // Step 3: Update all weights and biases
        for (size_t L = 0; L < layers.size() - 1; L++) {
            Layer& currLayer = layers[L];
            Layer& nextLayer = layers[L + 1];

            for (size_t i = 0; i < currLayer.neurons.size(); i++) {
                for (size_t j = 0; j < nextLayer.neurons.size(); j++) {
                    double delta = learningRate * nextLayer.neurons[j].gradient * currLayer.neurons[i].value;
                    currLayer.neurons[i].weights[j] += delta;
                }
            }
            for (Neuron& n : nextLayer.neurons) {
                n.bias += learningRate * n.gradient;
            }
        }
    }

    // Train on dataset for specified epochs
    void train(const vector<Sample>& data, int epochs, bool showProgress = true) {
        int reportInterval = max(1, epochs / 10);

        for (int epoch = 1; epoch <= epochs; epoch++) {
            double totalLoss = 0.0;

            for (const Sample& sample : data) {
                vector<double> output = predict(sample.inputs);
                totalLoss += computeLoss(output, sample.targets);
                backpropagate(sample.targets);
            }

            if (showProgress && epoch % reportInterval == 0) {
                double avgLoss = totalLoss / data.size();
                cout << "  Epoch " << setw(5) << epoch 
                          << " | Loss: " << fixed << setprecision(6) << avgLoss << "\n";
            }
        }
    }

    // Compute mean squared error
    double computeLoss(const vector<double>& output, const vector<double>& target) {
        double sum = 0.0;
        for (size_t i = 0; i < output.size(); i++) {
            double diff = target[i] - output[i];
            sum += diff * diff;
        }
        return sum / 2.0;
    }

    // Total loss over entire dataset
    double totalLoss(const vector<Sample>& data) {
        double sum = 0.0;
        for (const Sample& s : data) {
            sum += computeLoss(predict(s.inputs), s.targets);
        }
        return sum / data.size();
    }

    // Save weights to file
    void saveWeights(const string& filename) {
        ofstream file(filename);
        if (!file) {
            cerr << "Error: Cannot save to " << filename << "\n";
            return;
        }

        // Write topology
        file << layers.size() << "\n";
        for (const Layer& layer : layers) {
            file << layer.neurons.size() << " ";
        }
        file << "\n";

        // Write weights and biases
        for (const Layer& layer : layers) {
            for (const Neuron& n : layer.neurons) {
                file << n.bias;
                for (double w : n.weights) {
                    file << " " << w;
                }
                file << "\n";
            }
        }
        cout << "Saved to " << filename << "\n";
    }

    // Load weights from file
    bool loadWeights(const string& filename) {
        ifstream file(filename);
        if (!file) {
            cerr << "Error: Cannot open " << filename << "\n";
            return false;
        }

        size_t numLayers;
        file >> numLayers;

        vector<int> topology(numLayers);
        for (size_t i = 0; i < numLayers; i++) {
            file >> topology[i];
        }

        // Rebuild network with loaded topology
        layers.clear();
        for (size_t i = 0; i < topology.size(); i++) {
            int numOutputs = (i < topology.size() - 1) ? topology[i + 1] : 0;
            layers.emplace_back(topology[i], numOutputs, i == 0);
        }

        // Load weights and biases
        for (Layer& layer : layers) {
            for (Neuron& n : layer.neurons) {
                file >> n.bias;
                for (double& w : n.weights) {
                    file >> w;
                }
            }
        }
        cout << "Loaded from " << filename << "\n";
        return true;
    }

private:
    vector<Layer> layers;
};

// ============================================================================
// Dataset - Provides training data
// ============================================================================
class Dataset {
public:
    // Classic XOR problem - a simple non-linear classification task
    static vector<Sample> XOR() {
        return {
            {{0, 0}, {0}},
            {{0, 1}, {1}},
            {{1, 0}, {1}},
            {{1, 1}, {0}}
        };
    }
};

// ============================================================================
// Helper functions for CLI
// ============================================================================
void printPredictions(NeuralNetwork& net, const vector<Sample>& data) {
    cout << fixed << setprecision(4);
    cout << "\n  Input     | Predicted | Target\n";
    cout << "  ----------|-----------|-------\n";
    for (const Sample& s : data) {
        auto out = net.predict(s.inputs);
        cout << "  [" << s.inputs[0] << ", " << s.inputs[1] << "]"
                  << "  |   " << out[0] 
                  << "   |  " << s.targets[0] << "\n";
    }
}

void printMenu() {
    cout << "\n========================================\n";
    cout << "       NeuralNet++ Interactive Menu\n";
    cout << "========================================\n";
    cout << "  1. Train network on XOR\n";
    cout << "  2. Test current network\n";
    cout << "  3. Predict custom input\n";
    cout << "  4. Save weights to file\n";
    cout << "  5. Load weights from file\n";
    cout << "  6. Run full demo\n";
    cout << "  0. Exit\n";
    cout << "----------------------------------------\n";
    cout << "  Choice: ";
}

void runDemo() {
    cout << "\n=== XOR Demo ===\n";
    cout << "Creating network with topology: 2 -> 3 -> 1\n";
    
    NeuralNetwork net({2, 3, 1}, 0.5);
    auto data = Dataset::XOR();

    cout << "\nBEFORE training (random weights):";
    printPredictions(net, data);
    cout << "\nLoss: " << fixed << setprecision(6) << net.totalLoss(data) << "\n";

    cout << "\nTraining for 5000 epochs...\n";
    net.train(data, 5000);

    cout << "\nAFTER training:";
    printPredictions(net, data);
    cout << "\nFinal Loss: " << net.totalLoss(data) << "\n";
}

// ============================================================================
// Main - Interactive CLI
// ============================================================================
int main(int argc, char* argv[]) {
    cout << "\n";
    cout << "  _   _                      _  _   _      _   \n";
    cout << " | \\ | | ___ _   _ _ __ __ _| || \\ | | ___| |_ \n";
    cout << " |  \\| |/ _ \\ | | | '__/ _` | ||  \\| |/ _ \\ __|\n";
    cout << " | |\\  |  __/ |_| | | | (_| | || |\\  |  __/ |_ \n";
    cout << " |_| \\_|\\___|\\__,_|_|  \\__,_|_||_| \\_|\\___|\\__| ++\n";
    cout << "\n A Simple Neural Network Engine in C++\n";

    // Quick command-line mode
    if (argc > 1 && string(argv[1]) == "demo") {
        runDemo();
        return 0;
    }

    // Interactive mode
    NeuralNetwork net({2, 3, 1}, 0.5);
    auto data = Dataset::XOR();
    bool trained = false;

    while (true) {
        printMenu();
        
        int choice;
        cin >> choice;

        if (choice == 0) {
            cout << "\nGoodbye!\n";
            break;
        }

        switch (choice) {
            case 1: {  // Train
                int epochs;
                double lr;
                cout << "  Epochs (e.g., 5000): ";
                cin >> epochs;
                cout << "  Learning rate (e.g., 0.5): ";
                cin >> lr;
                
                net = NeuralNetwork({2, 3, 1}, lr);
                cout << "\nTraining...\n";
                net.train(data, epochs);
                trained = true;
                
                cout << "\nResults:";
                printPredictions(net, data);
                cout << "\nFinal Loss: " << fixed << setprecision(6) 
                          << net.totalLoss(data) << "\n";
                break;
            }
            case 2: {  // Test
                if (!trained) {
                    cout << "\n  Note: Network has random weights (not trained yet)\n";
                }
                printPredictions(net, data);
                cout << "\nLoss: " << fixed << setprecision(6) 
                          << net.totalLoss(data) << "\n";
                break;
            }
            case 3: {  // Predict
                double x, y;
                cout << "  Enter input x (0 or 1): ";
                cin >> x;
                cout << "  Enter input y (0 or 1): ";
                cin >> y;
                
                auto result = net.predict({x, y});
                cout << "\n  Prediction for [" << x << ", " << y << "]: " 
                          << fixed << setprecision(4) << result[0] << "\n";
                break;
            }
            case 4: {  // Save
                string filename;
                cout << "  Filename: ";
                cin >> filename;
                net.saveWeights(filename);
                break;
            }
            case 5: {  // Load
                string filename;
                cout << "  Filename: ";
                cin >> filename;
                if (net.loadWeights(filename)) {
                    trained = true;
                }
                break;
            }
            case 6: {  // Demo
                runDemo();
                break;
            }
            default:
                cout << "  Invalid choice. Try again.\n";
        }
    }

    return 0;
}
