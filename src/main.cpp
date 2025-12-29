#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <sstream>
#include <vector>

class MathUtils {
public:
    static double sigmoid(double x) {
        return 1.0 / (1.0 + std::exp(-x));
    }

    static double sigmoidDerivativeFromOutput(double output) {
        return output * (1.0 - output);
    }

    static double randomWeight() {
        static std::mt19937 rng(std::random_device{}());
        static std::uniform_real_distribution<double> dist(-1.0, 1.0);
        return dist(rng);
    }
};

struct Sample {
    std::vector<double> inputs;
    std::vector<double> targets;
};

class Neuron {
public:
    explicit Neuron(size_t outgoingCount = 0, bool withBias = true)
        : value(0.0), bias(withBias ? MathUtils::randomWeight() : 0.0), gradient(0.0), outgoingWeights(outgoingCount) {
        for (double &w : outgoingWeights) {
            w = MathUtils::randomWeight();
        }
    }

    double value;
    double bias;
    double gradient;
    std::vector<double> outgoingWeights;
};

class Layer {
public:
    Layer(size_t neuronCount = 0, size_t outgoingCount = 0, bool isInput = false) {
        neurons.reserve(neuronCount);
        for (size_t i = 0; i < neuronCount; ++i) {
            neurons.emplace_back(outgoingCount, !isInput);
        }
    }

    std::vector<Neuron> neurons;
};

class NeuralNetwork {
public:
    NeuralNetwork() = default;

    NeuralNetwork(const std::vector<size_t> &topology, double learningRate)
        : learningRate(learningRate) {
        if (topology.size() < 2) {
            throw std::invalid_argument("Topology must include at least input and output layers.");
        }
        layers.reserve(topology.size());
        for (size_t i = 0; i < topology.size(); ++i) {
            size_t nextCount = (i + 1 < topology.size()) ? topology[i + 1] : 0;
            bool isInput = i == 0;
            layers.emplace_back(topology[i], nextCount, isInput);
        }
    }

    std::vector<double> feedForward(const std::vector<double> &inputs) {
        if (inputs.size() != layers.front().neurons.size()) {
            throw std::invalid_argument("Input size does not match input layer.");
        }
        for (size_t i = 0; i < inputs.size(); ++i) {
            layers.front().neurons[i].value = inputs[i];
        }
        for (size_t layerIdx = 1; layerIdx < layers.size(); ++layerIdx) {
            Layer &prev = layers[layerIdx - 1];
            Layer &curr = layers[layerIdx];
            for (size_t j = 0; j < curr.neurons.size(); ++j) {
                double sum = 0.0;
                for (size_t i = 0; i < prev.neurons.size(); ++i) {
                    sum += prev.neurons[i].value * prev.neurons[i].outgoingWeights[j];
                }
                sum += curr.neurons[j].bias;
                curr.neurons[j].value = MathUtils::sigmoid(sum);
            }
        }
        std::vector<double> outputs;
        Layer &outputLayer = layers.back();
        outputs.reserve(outputLayer.neurons.size());
        for (const auto &n : outputLayer.neurons) {
            outputs.push_back(n.value);
        }
        return outputs;
    }

    void backPropagate(const std::vector<double> &targets) {
        Layer &outputLayer = layers.back();
        if (targets.size() != outputLayer.neurons.size()) {
            throw std::invalid_argument("Target size does not match output layer.");
        }

        // Output gradients
        for (size_t i = 0; i < outputLayer.neurons.size(); ++i) {
            double output = outputLayer.neurons[i].value;
            double error = targets[i] - output;
            outputLayer.neurons[i].gradient = error * MathUtils::sigmoidDerivativeFromOutput(output);
        }

        // Hidden gradients
        for (size_t layerIdx = layers.size() - 2; layerIdx > 0; --layerIdx) {
            Layer &curr = layers[layerIdx];
            Layer &next = layers[layerIdx + 1];
            for (size_t i = 0; i < curr.neurons.size(); ++i) {
                double downstream = 0.0;
                for (size_t j = 0; j < next.neurons.size(); ++j) {
                    downstream += curr.neurons[i].outgoingWeights[j] * next.neurons[j].gradient;
                }
                curr.neurons[i].gradient = MathUtils::sigmoidDerivativeFromOutput(curr.neurons[i].value) * downstream;
            }
        }

        // Update weights and biases
        for (size_t layerIdx = 0; layerIdx < layers.size() - 1; ++layerIdx) {
            Layer &curr = layers[layerIdx];
            Layer &next = layers[layerIdx + 1];
            for (size_t i = 0; i < curr.neurons.size(); ++i) {
                for (size_t j = 0; j < next.neurons.size(); ++j) {
                    double delta = learningRate * next.neurons[j].gradient * curr.neurons[i].value;
                    curr.neurons[i].outgoingWeights[j] += delta;
                }
            }
            for (auto &neuron : next.neurons) {
                neuron.bias += learningRate * neuron.gradient;
            }
        }
    }

    double train(const std::vector<Sample> &dataset, size_t epochs, size_t reportEvery = 0) {
        double lastLoss = 0.0;
        for (size_t epoch = 1; epoch <= epochs; ++epoch) {
            double epochLoss = 0.0;
            for (const auto &sample : dataset) {
                auto outputs = feedForward(sample.inputs);
                epochLoss += computeSampleLoss(outputs, sample.targets);
                backPropagate(sample.targets);
            }
            epochLoss /= static_cast<double>(dataset.size());
            lastLoss = epochLoss;
            if (reportEvery != 0 && epoch % reportEvery == 0) {
                std::cout << "Epoch " << epoch << " loss: " << epochLoss << "\n";
            }
        }
        return lastLoss;
    }

    std::vector<double> predict(const std::vector<double> &inputs) {
        return feedForward(inputs);
    }

    double computeLoss(const std::vector<Sample> &dataset) {
        double total = 0.0;
        for (const auto &sample : dataset) {
            auto outputs = feedForward(sample.inputs);
            total += computeSampleLoss(outputs, sample.targets);
        }
        return total / static_cast<double>(dataset.size());
    }

    void saveWeights(const std::string &filename) const;
    void loadWeights(const std::string &filename);

    std::vector<size_t> topology() const {
        std::vector<size_t> topo;
        topo.reserve(layers.size());
        for (const auto &layer : layers) {
            topo.push_back(layer.neurons.size());
        }
        return topo;
    }

    const std::vector<Layer> &getLayers() const { return layers; }
    std::vector<Layer> &getLayers() { return layers; }
    double getLearningRate() const { return learningRate; }

private:
    double computeSampleLoss(const std::vector<double> &outputs, const std::vector<double> &targets) {
        double sum = 0.0;
        for (size_t i = 0; i < outputs.size(); ++i) {
            double diff = targets[i] - outputs[i];
            sum += diff * diff;
        }
        return sum / 2.0;
    }

    std::vector<Layer> layers;
    double learningRate = 0.1;
};

class ModelIO {
public:
    static void save(const NeuralNetwork &net, const std::string &filename) {
        std::ofstream out(filename);
        if (!out) {
            throw std::runtime_error("Failed to open file for saving: " + filename);
        }
        auto topo = net.topology();
        out << topo.size() << "\n";
        for (size_t size : topo) {
            out << size << " ";
        }
        out << "\n";

        const auto &layers = net.getLayers();
        for (size_t layerIdx = 0; layerIdx < layers.size(); ++layerIdx) {
            const auto &layer = layers[layerIdx];
            for (size_t nIdx = 0; nIdx < layer.neurons.size(); ++nIdx) {
                const auto &n = layer.neurons[nIdx];
                out << n.bias << " " << n.outgoingWeights.size();
                for (double w : n.outgoingWeights) {
                    out << " " << w;
                }
                out << "\n";
            }
        }
    }

    static NeuralNetwork load(const std::string &filename, double learningRate) {
        std::ifstream in(filename);
        if (!in) {
            throw std::runtime_error("Failed to open file for loading: " + filename);
        }
        size_t layerCount = 0;
        in >> layerCount;
        if (layerCount < 2) {
            throw std::runtime_error("Invalid topology in file.");
        }
        std::vector<size_t> topo(layerCount);
        for (size_t i = 0; i < layerCount; ++i) {
            in >> topo[i];
        }
        NeuralNetwork net(topo, learningRate);
        auto &layers = net.getLayers();
        for (size_t layerIdx = 0; layerIdx < layers.size(); ++layerIdx) {
            auto &layer = layers[layerIdx];
            for (size_t nIdx = 0; nIdx < layer.neurons.size(); ++nIdx) {
                auto &n = layer.neurons[nIdx];
                size_t outgoingSize = 0;
                in >> n.bias >> outgoingSize;
                n.outgoingWeights.resize(outgoingSize);
                for (size_t wIdx = 0; wIdx < outgoingSize; ++wIdx) {
                    in >> n.outgoingWeights[wIdx];
                }
            }
        }
        return net;
    }
};

void NeuralNetwork::saveWeights(const std::string &filename) const {
    ModelIO::save(*this, filename);
}

void NeuralNetwork::loadWeights(const std::string &filename) {
    NeuralNetwork loaded = ModelIO::load(filename, learningRate);
    *this = loaded;
}

class Dataset {
public:
    static std::vector<Sample> xorDataset() {
        return {
            {{0.0, 0.0}, {0.0}},
            {{0.0, 1.0}, {1.0}},
            {{1.0, 0.0}, {1.0}},
            {{1.0, 1.0}, {0.0}},
        };
    }
};

static void printPredictions(NeuralNetwork &net, const std::vector<Sample> &data) {
    std::cout << std::fixed << std::setprecision(4);
    for (const auto &sample : data) {
        auto out = net.predict(sample.inputs);
        std::cout << "Input [" << sample.inputs[0] << ", " << sample.inputs[1]
                  << "] -> Predicted: " << out[0] << " Target: " << sample.targets[0] << "\n";
    }
}

static void runDemo(size_t epochs, double lr) {
    std::vector<size_t> topo{2, 3, 1};
    NeuralNetwork net(topo, lr);
    auto data = Dataset::xorDataset();

    std::cout << "Before training (random weights):\n";
    printPredictions(net, data);

    net.train(data, epochs, epochs / 10);
    double finalLoss = net.computeLoss(data);

    std::cout << "\nAfter training:\n";
    printPredictions(net, data);
    std::cout << "Final loss: " << finalLoss << "\n";
}

static void printUsage() {
    std::cout << "NeuralNet++ CLI\n"
                 "Usage:\n"
                 "  neuralnet demo [--epochs N] [--lr V]\n"
                 "  neuralnet train [--epochs N] [--lr V] [--save file]\n"
                 "  neuralnet test [--weights file]\n"
                 "  neuralnet predict x y [--weights file]\n"
                 "  neuralnet save file\n"
                 "  neuralnet load file\n";
}

int main(int argc, char **argv) {
    if (argc == 1) {
        printUsage();
        std::cout << "\nRunning demo with defaults...\n";
        runDemo(5000, 0.5);
        return 0;
    }

    std::string cmd = argv[1];
    size_t epochs = 5000;
    double lr = 0.5;
    std::string filename;
    std::vector<size_t> topo{2, 3, 1};
    NeuralNetwork net(topo, lr);
    auto data = Dataset::xorDataset();

    auto parseOption = [&](const std::string &flag, const std::string &value) {
        if (flag == "--epochs") {
            epochs = static_cast<size_t>(std::stoul(value));
        } else if (flag == "--lr") {
            lr = std::stod(value);
        } else if (flag == "--save" || flag == "--weights") {
            filename = value;
        }
    };

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--epochs" || arg == "--lr" || arg == "--save" || arg == "--weights") {
            if (i + 1 >= argc) {
                throw std::invalid_argument("Missing value for option " + arg);
            }
            parseOption(arg, argv[++i]);
        }
    }

    net = NeuralNetwork(topo, lr);

    if (cmd == "demo") {
        runDemo(epochs, lr);
    } else if (cmd == "train") {
        double initialLoss = net.computeLoss(data);
        std::cout << "Initial loss: " << initialLoss << "\n";
        double finalLoss = net.train(data, epochs, epochs / 10);
        finalLoss = net.computeLoss(data);
        std::cout << "Final loss: " << finalLoss << "\n";
        printPredictions(net, data);
        if (!filename.empty()) {
            net.saveWeights(filename);
            std::cout << "Saved weights to " << filename << "\n";
        }
    } else if (cmd == "test") {
        if (!filename.empty()) {
            net.loadWeights(filename);
            std::cout << "Loaded weights from " << filename << "\n";
        }
        printPredictions(net, data);
        std::cout << "Loss: " << net.computeLoss(data) << "\n";
    } else if (cmd == "predict") {
        if (argc < 4) {
            throw std::invalid_argument("predict requires two inputs (x y).");
        }
        double x = std::stod(argv[2]);
        double y = std::stod(argv[3]);
        if (!filename.empty()) {
            net.loadWeights(filename);
            std::cout << "Loaded weights from " << filename << "\n";
        }
        auto out = net.predict({x, y});
        std::cout << "Prediction for [" << x << ", " << y << "]: " << out[0] << "\n";
    } else if (cmd == "save") {
        if (argc < 3) {
            throw std::invalid_argument("save requires filename.");
        }
        net.saveWeights(argv[2]);
        std::cout << "Saved weights to " << argv[2] << "\n";
    } else if (cmd == "load") {
        if (argc < 3) {
            throw std::invalid_argument("load requires filename.");
        }
        net.loadWeights(argv[2]);
        std::cout << "Loaded weights from " << argv[2] << "\n";
        printPredictions(net, data);
    } else {
        printUsage();
        throw std::invalid_argument("Unknown command: " + cmd);
    }

    return 0;
}
