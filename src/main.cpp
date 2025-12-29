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

// ============================================================================
// MathUtils - Helper functions for neural network math
// ============================================================================
class MathUtils {
public:
    // Sigmoid squashes any value to range (0, 1)
    static double sigmoid(double x) {
        return 1.0 / (1.0 + std::exp(-x));
    }

    // Derivative of sigmoid (used in backpropagation)
    // Note: takes the OUTPUT of sigmoid, not the input
    static double sigmoidDerivative(double sigmoidOutput) {
        return sigmoidOutput * (1.0 - sigmoidOutput);
    }

    // Random weight between -1 and 1
    static double randomWeight() {
        static std::mt19937 rng(std::random_device{}());
        static std::uniform_real_distribution<double> dist(-1.0, 1.0);
        return dist(rng);
    }
};

// ============================================================================
// Sample - One training example (inputs and expected outputs)
// ============================================================================
struct Sample {
    std::vector<double> inputs;
    std::vector<double> targets;
};

// ============================================================================
// Neuron - A single node in the network
// ============================================================================
class Neuron {
public:
    double value = 0.0;                      // Current activation value
    double bias = 0.0;                       // Bias term (added before activation)
    double gradient = 0.0;                   // Error gradient (for training)
    std::vector<double> weights;             // Weights to next layer neurons

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
    std::vector<Neuron> neurons;

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
    NeuralNetwork(const std::vector<int>& topology, double lr = 0.5) 
        : learningRate(lr) {
        for (size_t i = 0; i < topology.size(); i++) {
            int numOutputs = (i < topology.size() - 1) ? topology[i + 1] : 0;
            bool isInput = (i == 0);
            layers.emplace_back(topology[i], numOutputs, isInput);
        }
    }

    // Forward pass: compute output from inputs
    std::vector<double> predict(const std::vector<double>& inputs) {
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
        std::vector<double> outputs;
        for (const Neuron& n : layers.back().neurons) {
            outputs.push_back(n.value);
        }
        return outputs;
    }

    // Backward pass: compute gradients and update weights
    void backpropagate(const std::vector<double>& targets) {
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
    void train(const std::vector<Sample>& data, int epochs, bool showProgress = true) {
        int reportInterval = std::max(1, epochs / 10);

        for (int epoch = 1; epoch <= epochs; epoch++) {
            double totalLoss = 0.0;

            for (const Sample& sample : data) {
                std::vector<double> output = predict(sample.inputs);
                totalLoss += computeLoss(output, sample.targets);
                backpropagate(sample.targets);
            }

            if (showProgress && epoch % reportInterval == 0) {
                double avgLoss = totalLoss / data.size();
                std::cout << "  Epoch " << std::setw(5) << epoch 
                          << " | Loss: " << std::fixed << std::setprecision(6) << avgLoss << "\n";
            }
        }
    }

    // Compute mean squared error
    double computeLoss(const std::vector<double>& output, const std::vector<double>& target) {
        double sum = 0.0;
        for (size_t i = 0; i < output.size(); i++) {
            double diff = target[i] - output[i];
            sum += diff * diff;
        }
        return sum / 2.0;
    }

    // Total loss over entire dataset
    double totalLoss(const std::vector<Sample>& data) {
        double sum = 0.0;
        for (const Sample& s : data) {
            sum += computeLoss(predict(s.inputs), s.targets);
        }
        return sum / data.size();
    }

    // Save weights to file
    void saveWeights(const std::string& filename) {
        std::ofstream file(filename);
        if (!file) {
            std::cerr << "Error: Cannot save to " << filename << "\n";
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
        std::cout << "Saved to " << filename << "\n";
    }

    // Load weights from file
    bool loadWeights(const std::string& filename) {
        std::ifstream file(filename);
        if (!file) {
            std::cerr << "Error: Cannot open " << filename << "\n";
            return false;
        }

        size_t numLayers;
        file >> numLayers;

        std::vector<int> topology(numLayers);
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
        std::cout << "Loaded from " << filename << "\n";
        return true;
    }

private:
    std::vector<Layer> layers;
};

// ============================================================================
// Dataset - Provides training data
// ============================================================================
class Dataset {
public:
    // Classic XOR problem - a simple non-linear classification task
    static std::vector<Sample> XOR() {
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
void printPredictions(NeuralNetwork& net, const std::vector<Sample>& data) {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "\n  Input     | Predicted | Target\n";
    std::cout << "  ----------|-----------|-------\n";
    for (const Sample& s : data) {
        auto out = net.predict(s.inputs);
        std::cout << "  [" << s.inputs[0] << ", " << s.inputs[1] << "]"
                  << "  |   " << out[0] 
                  << "   |  " << s.targets[0] << "\n";
    }
}

void printMenu() {
    std::cout << "\n========================================\n";
    std::cout << "       NeuralNet++ Interactive Menu\n";
    std::cout << "========================================\n";
    std::cout << "  1. Train network on XOR\n";
    std::cout << "  2. Test current network\n";
    std::cout << "  3. Predict custom input\n";
    std::cout << "  4. Save weights to file\n";
    std::cout << "  5. Load weights from file\n";
    std::cout << "  6. Run full demo\n";
    std::cout << "  0. Exit\n";
    std::cout << "----------------------------------------\n";
    std::cout << "  Choice: ";
}

void runDemo() {
    std::cout << "\n=== XOR Demo ===\n";
    std::cout << "Creating network with topology: 2 -> 3 -> 1\n";
    
    NeuralNetwork net({2, 3, 1}, 0.5);
    auto data = Dataset::XOR();

    std::cout << "\nBEFORE training (random weights):";
    printPredictions(net, data);
    std::cout << "\nLoss: " << std::fixed << std::setprecision(6) << net.totalLoss(data) << "\n";

    std::cout << "\nTraining for 5000 epochs...\n";
    net.train(data, 5000);

    std::cout << "\nAFTER training:";
    printPredictions(net, data);
    std::cout << "\nFinal Loss: " << net.totalLoss(data) << "\n";
}

// ============================================================================
// Main - Interactive CLI
// ============================================================================
int main(int argc, char* argv[]) {
    std::cout << "\n";
    std::cout << "  _   _                      _  _   _      _   \n";
    std::cout << " | \\ | | ___ _   _ _ __ __ _| || \\ | | ___| |_ \n";
    std::cout << " |  \\| |/ _ \\ | | | '__/ _` | ||  \\| |/ _ \\ __|\n";
    std::cout << " | |\\  |  __/ |_| | | | (_| | || |\\  |  __/ |_ \n";
    std::cout << " |_| \\_|\\___|\\__,_|_|  \\__,_|_||_| \\_|\\___|\\__| ++\n";
    std::cout << "\n A Simple Neural Network Engine in C++\n";

    // Quick command-line mode
    if (argc > 1 && std::string(argv[1]) == "demo") {
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
        std::cin >> choice;

        if (choice == 0) {
            std::cout << "\nGoodbye!\n";
            break;
        }

        switch (choice) {
            case 1: {  // Train
                int epochs;
                double lr;
                std::cout << "  Epochs (e.g., 5000): ";
                std::cin >> epochs;
                std::cout << "  Learning rate (e.g., 0.5): ";
                std::cin >> lr;
                
                net = NeuralNetwork({2, 3, 1}, lr);
                std::cout << "\nTraining...\n";
                net.train(data, epochs);
                trained = true;
                
                std::cout << "\nResults:";
                printPredictions(net, data);
                std::cout << "\nFinal Loss: " << std::fixed << std::setprecision(6) 
                          << net.totalLoss(data) << "\n";
                break;
            }
            case 2: {  // Test
                if (!trained) {
                    std::cout << "\n  Note: Network has random weights (not trained yet)\n";
                }
                printPredictions(net, data);
                std::cout << "\nLoss: " << std::fixed << std::setprecision(6) 
                          << net.totalLoss(data) << "\n";
                break;
            }
            case 3: {  // Predict
                double x, y;
                std::cout << "  Enter input x (0 or 1): ";
                std::cin >> x;
                std::cout << "  Enter input y (0 or 1): ";
                std::cin >> y;
                
                auto result = net.predict({x, y});
                std::cout << "\n  Prediction for [" << x << ", " << y << "]: " 
                          << std::fixed << std::setprecision(4) << result[0] << "\n";
                break;
            }
            case 4: {  // Save
                std::string filename;
                std::cout << "  Filename: ";
                std::cin >> filename;
                net.saveWeights(filename);
                break;
            }
            case 5: {  // Load
                std::string filename;
                std::cout << "  Filename: ";
                std::cin >> filename;
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
                std::cout << "  Invalid choice. Try again.\n";
        }
    }

    return 0;
}
