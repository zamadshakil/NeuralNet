/*
 * NeuralNet++ - A Powerful Feed-Forward Neural Network in C++
 *
 * OOP composition: Network -> Layers -> Neurons
 *
 * Features:
 *   - Training with backpropagation
 *   - CSV dataset loading with header detection & quoted fields
 *   - Data normalization (Min-Max, Z-Score)
 *   - Train/test split with reproducible shuffling
 *   - Multiple activation functions (Sigmoid, ReLU, Leaky ReLU, Tanh)
 *   - Save/load weights, XOR demo, interactive CLI
 */

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

// ============================================================================
// ActivationType - Supported activation functions
// ============================================================================
enum class ActivationType { SIGMOID, RELU, LEAKY_RELU, TANH, SOFTMAX };

enum class OptimizerType { SGD, ADAM };

string activationName(ActivationType t) {
  switch (t) {
  case ActivationType::SIGMOID:
    return "Sigmoid";
  case ActivationType::RELU:
    return "ReLU";
  case ActivationType::LEAKY_RELU:
    return "Leaky ReLU";
  case ActivationType::TANH:
    return "Tanh";
  case ActivationType::SOFTMAX:
    return "Softmax";
  }
  return "Unknown";
}

// ============================================================================
// MathUtils - Helper functions for neural network math
// ============================================================================
class MathUtils {
public:
  static double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }

  static double sigmoidDerivative(double sigmoidOutput) {
    return sigmoidOutput * (1.0 - sigmoidOutput);
  }

  static double relu(double x) { return x > 0.0 ? x : 0.0; }

  static double reluDerivative(double x) { return x > 0.0 ? 1.0 : 0.0; }

  static double leakyRelu(double x, double alpha = 0.01) {
    return x > 0.0 ? x : alpha * x;
  }

  static double leakyReluDerivative(double x, double alpha = 0.01) {
    return x > 0.0 ? 1.0 : alpha;
  }

  static double tanhActivation(double x) { return tanh(x); }

  static double tanhDerivative(double tanhOutput) {
    return 1.0 - tanhOutput * tanhOutput;
  }

  // Apply activation function
  static double activate(double x, ActivationType type) {
    switch (type) {
    case ActivationType::SIGMOID:
      return sigmoid(x);
    case ActivationType::RELU:
      return relu(x);
    case ActivationType::LEAKY_RELU:
      return leakyRelu(x);
    case ActivationType::TANH:
      return tanhActivation(x);
    case ActivationType::SOFTMAX:
      return x; // Softmax is applied at the layer level, not per-neuron
    }
    return sigmoid(x);
  }

  // Apply activation derivative (takes output of activation, except ReLU
  // variants take raw sum)
  static double activateDerivative(double output, ActivationType type) {
    switch (type) {
    case ActivationType::SIGMOID:
      return sigmoidDerivative(output);
    case ActivationType::RELU:
      return reluDerivative(output);
    case ActivationType::LEAKY_RELU:
      return leakyReluDerivative(output);
    case ActivationType::TANH:
      return tanhDerivative(output);
    case ActivationType::SOFTMAX:
      return 1.0; // Derivative handled in backpropagation (softmax + CE
                  // combined)
    }
    return sigmoidDerivative(output);
  }

  static double randomWeight() {
    static mt19937 rng(random_device{}());
    static uniform_real_distribution<double> dist(
        -0.5, 0.5); // Smaller initialization for stability
    return dist(rng);
  }

  static void softmax(vector<double> &values) {
    double maxVal = -1e9;
    for (double v : values)
      if (v > maxVal)
        maxVal = v;

    double sum = 0.0;
    for (double &v : values) {
      v = exp(v - maxVal);
      sum += v;
    }
    for (double &v : values)
      v /= sum;
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
  double value = 0.0;
  double rawSum = 0.0; // Pre-activation sum (needed for ReLU derivatives)
  double bias = 0.0;
  double gradient = 0.0;
  vector<double> weights;

  // Adam Optimizer Cache
  vector<double> m_weights, v_weights;
  double m_bias = 0.0, v_bias = 0.0;

  Neuron(int numOutputs, bool hasBias = true) {
    bias = hasBias ? MathUtils::randomWeight() : 0.0;
    weights.resize(numOutputs);
    m_weights.resize(numOutputs, 0.0);
    v_weights.resize(numOutputs, 0.0);

    for (double &w : weights) {
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

  Layer(int numNeurons, int numOutputs, bool isInputLayer = false) {
    for (int i = 0; i < numNeurons; i++) {
      neurons.emplace_back(numOutputs, !isInputLayer);
    }
  }
};

// ============================================================================
// CSVLoader - Parse CSV files into Samples
// ============================================================================
class CSVLoader {
public:
  vector<string> headers;
  char delimiter;

  CSVLoader(char delim = ',') : delimiter(delim) {}

  // Parse a single CSV line respecting quoted fields
  vector<string> parseLine(const string &line) const {
    vector<string> fields;
    string field;
    bool inQuotes = false;

    for (size_t i = 0; i < line.size(); i++) {
      char c = line[i];
      if (inQuotes) {
        if (c == '"') {
          if (i + 1 < line.size() && line[i + 1] == '"') {
            field += '"';
            i++; // skip escaped quote
          } else {
            inQuotes = false;
          }
        } else {
          field += c;
        }
      } else {
        if (c == '"') {
          inQuotes = true;
        } else if (c == delimiter) {
          fields.push_back(field);
          field.clear();
        } else if (c != '\r') {
          field += c;
        }
      }
    }
    fields.push_back(field);
    return fields;
  }

  // Check if a string is numeric
  static bool isNumeric(const string &s) {
    if (s.empty())
      return false;
    try {
      size_t pos;
      stod(s, &pos);
      return pos == s.size();
    } catch (...) {
      return false;
    }
  }

  // Trim whitespace
  static string trim(const string &s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start == string::npos)
      return "";
    size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
  }

  // Load CSV file and return samples
  // inputCols: indices of input columns
  // targetCols: indices of target columns
  vector<Sample> load(const string &filename, const vector<int> &inputCols,
                      const vector<int> &targetCols) {
    ifstream file(filename);
    if (!file.is_open()) {
      cerr << "\n  Error: Cannot open file '" << filename << "'\n";
      return {};
    }

    vector<Sample> samples;
    string line;
    int lineNum = 0;
    bool hasHeader = false;

    while (getline(file, line)) {
      lineNum++;
      line = trim(line);

      // Skip empty lines and comments
      if (line.empty() || line[0] == '#')
        continue;

      vector<string> fields = parseLine(line);

      // Auto-detect header: if first data line has non-numeric fields
      if (samples.empty() && !hasHeader) {
        bool allNumeric = true;
        for (const string &f : fields) {
          if (!isNumeric(trim(f))) {
            allNumeric = false;
            break;
          }
        }
        if (!allNumeric) {
          headers = fields;
          for (string &h : headers)
            h = trim(h);
          hasHeader = true;
          continue;
        }
      }

      // Validate column count
      int maxCol = 0;
      for (int c : inputCols)
        maxCol = max(maxCol, c);
      for (int c : targetCols)
        maxCol = max(maxCol, c);

      if ((int)fields.size() <= maxCol) {
        cerr << "  Warning: Line " << lineNum << " has only " << fields.size()
             << " columns (expected at least " << (maxCol + 1)
             << "), skipping.\n";
        continue;
      }

      // Build sample
      Sample s;
      bool valid = true;
      for (int c : inputCols) {
        string val = trim(fields[c]);
        if (!isNumeric(val)) {
          cerr << "  Warning: Non-numeric value '" << val << "' at line "
               << lineNum << " col " << c << ", skipping row.\n";
          valid = false;
          break;
        }
        s.inputs.push_back(stod(val));
      }
      if (!valid)
        continue;

      for (int c : targetCols) {
        string val = trim(fields[c]);
        if (!isNumeric(val)) {
          cerr << "  Warning: Non-numeric value '" << val << "' at line "
               << lineNum << " col " << c << ", skipping row.\n";
          valid = false;
          break;
        }
        s.targets.push_back(stod(val));
      }
      if (!valid)
        continue;

      samples.push_back(s);
    }

    // Print summary
    cout << "\n  ╔══════════════════════════════════════╗\n";
    cout << "  ║        CSV Dataset Loaded!            ║\n";
    cout << "  ╚══════════════════════════════════════╝\n";
    cout << "  File:    " << filename << "\n";
    cout << "  Rows:    " << samples.size() << "\n";
    cout << "  Inputs:  " << inputCols.size() << " columns\n";
    cout << "  Targets: " << targetCols.size() << " columns\n";

    if (!headers.empty()) {
      cout << "  Headers: ";
      for (size_t i = 0; i < headers.size(); i++) {
        cout << headers[i];
        if (i < headers.size() - 1)
          cout << ", ";
      }
      cout << "\n";
    }

    // Preview first 3 rows
    if (!samples.empty()) {
      cout << "\n  Preview (first " << min((int)samples.size(), 3)
           << " rows):\n";
      for (int i = 0; i < min((int)samples.size(), 3); i++) {
        cout << "    [";
        for (size_t j = 0; j < samples[i].inputs.size(); j++) {
          cout << fixed << setprecision(2) << samples[i].inputs[j];
          if (j < samples[i].inputs.size() - 1)
            cout << ", ";
        }
        cout << "] -> [";
        for (size_t j = 0; j < samples[i].targets.size(); j++) {
          cout << fixed << setprecision(2) << samples[i].targets[j];
          if (j < samples[i].targets.size() - 1)
            cout << ", ";
        }
        cout << "]\n";
      }
    }

    return samples;
  }
};

// ============================================================================
// DataNormalizer - Feature scaling utilities
// ============================================================================
class DataNormalizer {
public:
  enum Method { MIN_MAX, Z_SCORE };

  struct Stats {
    vector<double> mins, maxs;     // For min-max
    vector<double> means, stddevs; // For z-score
    Method method;
  };

  // Compute statistics and normalize in-place
  static Stats normalize(vector<Sample> &data, Method method) {
    if (data.empty())
      return {};

    int numInputs = data[0].inputs.size();
    Stats stats;
    stats.method = method;

    if (method == MIN_MAX) {
      stats.mins.assign(numInputs, 1e18);
      stats.maxs.assign(numInputs, -1e18);

      for (const auto &s : data) {
        for (int i = 0; i < numInputs; i++) {
          stats.mins[i] = min(stats.mins[i], s.inputs[i]);
          stats.maxs[i] = max(stats.maxs[i], s.inputs[i]);
        }
      }

      for (auto &s : data) {
        for (int i = 0; i < numInputs; i++) {
          double range = stats.maxs[i] - stats.mins[i];
          s.inputs[i] =
              (range > 1e-10) ? (s.inputs[i] - stats.mins[i]) / range : 0.0;
        }
      }

      cout << "  Applied Min-Max normalization (scaled to [0, 1])\n";

    } else { // Z_SCORE
      stats.means.assign(numInputs, 0.0);
      stats.stddevs.assign(numInputs, 0.0);

      for (const auto &s : data) {
        for (int i = 0; i < numInputs; i++) {
          stats.means[i] += s.inputs[i];
        }
      }
      for (int i = 0; i < numInputs; i++) {
        stats.means[i] /= data.size();
      }

      for (const auto &s : data) {
        for (int i = 0; i < numInputs; i++) {
          double diff = s.inputs[i] - stats.means[i];
          stats.stddevs[i] += diff * diff;
        }
      }
      for (int i = 0; i < numInputs; i++) {
        stats.stddevs[i] = sqrt(stats.stddevs[i] / data.size());
      }

      for (auto &s : data) {
        for (int i = 0; i < numInputs; i++) {
          s.inputs[i] = (stats.stddevs[i] > 1e-10)
                            ? (s.inputs[i] - stats.means[i]) / stats.stddevs[i]
                            : 0.0;
        }
      }

      cout << "  Applied Z-Score standardization (mean=0, std=1)\n";
    }

    return stats;
  }

  // Denormalize a single sample's inputs
  static vector<double> denormalize(const vector<double> &inputs,
                                    const Stats &stats) {
    vector<double> result = inputs;
    if (stats.method == MIN_MAX) {
      for (size_t i = 0; i < result.size() && i < stats.mins.size(); i++) {
        double range = stats.maxs[i] - stats.mins[i];
        result[i] = result[i] * range + stats.mins[i];
      }
    } else {
      for (size_t i = 0; i < result.size() && i < stats.means.size(); i++) {
        result[i] = result[i] * stats.stddevs[i] + stats.means[i];
      }
    }
    return result;
  }
};

// ============================================================================
// DataSplitter - Train/test split with shuffling
// ============================================================================
class DataSplitter {
public:
  struct SplitResult {
    vector<Sample> train;
    vector<Sample> test;
  };

  // Split data into train/test sets
  // trainRatio: fraction for training (e.g., 0.8 = 80% train, 20% test)
  // seed: for reproducible shuffling (0 = random)
  static SplitResult split(const vector<Sample> &data, double trainRatio,
                           unsigned int seed = 0) {
    vector<size_t> indices(data.size());
    iota(indices.begin(), indices.end(), 0);

    // Fisher-Yates shuffle
    mt19937 rng(seed == 0 ? random_device{}() : seed);
    for (size_t i = indices.size() - 1; i > 0; i--) {
      uniform_int_distribution<size_t> dist(0, i);
      swap(indices[i], indices[dist(rng)]);
    }

    size_t trainSize = static_cast<size_t>(data.size() * trainRatio);
    SplitResult result;

    for (size_t i = 0; i < indices.size(); i++) {
      if (i < trainSize) {
        result.train.push_back(data[indices[i]]);
      } else {
        result.test.push_back(data[indices[i]]);
      }
    }

    cout << "  Split: " << result.train.size() << " train / "
         << result.test.size() << " test (" << fixed << setprecision(0)
         << (trainRatio * 100) << "/" << ((1.0 - trainRatio) * 100) << "%)\n";

    return result;
  }
};

// ============================================================================
// NeuralNetwork - The main engine
// ============================================================================
class NeuralNetwork {
public:
  double learningRate;
  ActivationType activation;
  OptimizerType optimizer;

  // Adam specific parameters
  double beta1 = 0.9;
  double beta2 = 0.999;
  double epsilon = 1e-8;
  int timeStep = 0;

  NeuralNetwork(const vector<int> &topology, double lr = 0.01,
                ActivationType act = ActivationType::SIGMOID,
                OptimizerType opt = OptimizerType::SGD)
      : learningRate(lr), activation(act), optimizer(opt) {
    for (size_t i = 0; i < topology.size(); i++) {
      int numOutputs = (i < topology.size() - 1) ? topology[i + 1] : 0;
      bool isInput = (i == 0);
      layers.emplace_back(topology[i], numOutputs, isInput);
    }
  }

  int getInputSize() const {
    return layers.empty() ? 0 : layers.front().neurons.size();
  }
  int getOutputSize() const {
    return layers.empty() ? 0 : layers.back().neurons.size();
  }

  vector<double> predict(const vector<double> &inputs) {
    if (layers.empty() || inputs.size() != layers[0].neurons.size()) {
      // Return zero vector of checking for mismatch (or handle gracefully)
      // For now, we return zeroes to prevent crash, but this usually indicates
      // logic error
      return vector<double>(getOutputSize(), 0.0);
    }

    for (size_t i = 0; i < inputs.size(); i++) {
      layers[0].neurons[i].value = inputs[i];
    }

    for (size_t L = 1; L < layers.size(); L++) {
      Layer &prevLayer = layers[L - 1];
      Layer &currLayer = layers[L];

      for (size_t j = 0; j < currLayer.neurons.size(); j++) {
        double sum = currLayer.neurons[j].bias;

        for (size_t i = 0; i < prevLayer.neurons.size(); i++) {
          sum += prevLayer.neurons[i].value * prevLayer.neurons[i].weights[j];
        }

        currLayer.neurons[j].rawSum = sum;
        currLayer.neurons[j].value = MathUtils::activate(sum, activation);
      }

      // Apply Softmax on the output layer if selected
      if (L == layers.size() - 1 && activation == ActivationType::SOFTMAX) {
        vector<double> values;
        for (const auto &n : currLayer.neurons)
          values.push_back(n.rawSum);
        MathUtils::softmax(values);
        for (size_t i = 0; i < currLayer.neurons.size(); i++) {
          currLayer.neurons[i].value = values[i];
        }
      }
    }

    vector<double> outputs;
    for (const Neuron &n : layers.back().neurons) {
      outputs.push_back(n.value);
    }
    return outputs;
  }

  void backpropagate(const vector<double> &targets) {
    Layer &outputLayer = layers.back();

    // Auto-detect sparse targets (e.g. class index [2.0]) when output is vector
    bool sparseTarget = (targets.size() == 1 && outputLayer.neurons.size() > 1);
    int targetClass = sparseTarget ? static_cast<int>(targets[0]) : -1;

    for (size_t i = 0; i < outputLayer.neurons.size(); i++) {
      double output = outputLayer.neurons[i].value;

      double targetVal;
      if (sparseTarget) {
        targetVal = (i == (size_t)targetClass) ? 1.0 : 0.0;
      } else {
        // Safety check for bounds
        targetVal = (i < targets.size()) ? targets[i] : 0.0;
      }

      double error = targetVal - output;

      if (activation == ActivationType::SOFTMAX) {
        // For Softmax + Cross Entropy, gradient is just (target - output)
        // Since we add to gradient in backprop, and we defined error = target -
        // output The direction is correct. MathUtils::activateDerivative not
        // needed or set to 1.
        outputLayer.neurons[i].gradient = error;
      } else {
        outputLayer.neurons[i].gradient =
            error * MathUtils::activateDerivative(output, activation);
      }
    }

    for (int L = layers.size() - 2; L > 0; L--) {
      Layer &currLayer = layers[L];
      Layer &nextLayer = layers[L + 1];

      for (size_t i = 0; i < currLayer.neurons.size(); i++) {
        double errorSum = 0.0;
        for (size_t j = 0; j < nextLayer.neurons.size(); j++) {
          errorSum +=
              currLayer.neurons[i].weights[j] * nextLayer.neurons[j].gradient;
        }
        double output = currLayer.neurons[i].value;
        currLayer.neurons[i].gradient =
            errorSum * MathUtils::activateDerivative(output, activation);
      }
    }

    for (size_t L = 0; L < layers.size() - 1; L++) {
      Layer &currLayer = layers[L];
      Layer &nextLayer = layers[L + 1];

      for (size_t i = 0; i < currLayer.neurons.size(); i++) {
        for (size_t j = 0; j < nextLayer.neurons.size(); j++) {
          double gradient =
              nextLayer.neurons[j].gradient * currLayer.neurons[i].value;

          if (optimizer == OptimizerType::ADAM) {
            // Adam Update for Weight
            double &m = currLayer.neurons[i].m_weights[j];
            double &v = currLayer.neurons[i].v_weights[j];

            // Gradients are calculated for gradient ASCENT in this
            // implementation (target - output) So we use (+) gradient.

            m = beta1 * m + (1 - beta1) * gradient;
            v = beta2 * v + (1 - beta2) * gradient * gradient;

            double m_hat = m / (1 - pow(beta1, timeStep));
            double v_hat = v / (1 - pow(beta2, timeStep));

            currLayer.neurons[i].weights[j] +=
                learningRate * m_hat / (sqrt(v_hat) + epsilon);
          } else {
            // SGD
            currLayer.neurons[i].weights[j] += learningRate * gradient;
          }
        }
      }

      // Bias updates
      for (Neuron &n : nextLayer.neurons) {
        double gradient = n.gradient;

        if (optimizer == OptimizerType::ADAM) {
          double &m = n.m_bias;
          double &v = n.v_bias;

          m = beta1 * m + (1 - beta1) * gradient;
          v = beta2 * v + (1 - beta2) * gradient * gradient;

          double m_hat = m / (1 - pow(beta1, timeStep));
          double v_hat = v / (1 - pow(beta2, timeStep));

          n.bias += learningRate * m_hat / (sqrt(v_hat) + epsilon);
        } else {
          n.bias += learningRate * gradient;
        }
      }
    }
  }

  void train(const vector<Sample> &data, int epochs, bool showProgress = true) {
    int reportInterval = max(1, epochs / 10);

    for (int epoch = 1; epoch <= epochs; epoch++) {
      double totalLoss = 0.0;

      for (const Sample &sample : data) {
        timeStep++; // Increment for Adam
        vector<double> output = predict(sample.inputs);
        totalLoss += computeLoss(output, sample.targets);
        backpropagate(sample.targets);
      }

      if (showProgress && epoch % reportInterval == 0) {
        double avgLoss = totalLoss / data.size();
        cout << "  Epoch " << setw(5) << epoch << " | Loss: " << fixed
             << setprecision(6) << avgLoss << "\n";
      }
    }
  }

  double computeLoss(const vector<double> &output,
                     const vector<double> &target) {
    bool sparseTarget = (target.size() == 1 && output.size() > 1);
    int targetClass = sparseTarget ? static_cast<int>(target[0]) : -1;

    if (activation == ActivationType::SOFTMAX) {
      // Cross-Entropy Loss
      double sum = 0.0;
      for (size_t i = 0; i < output.size(); i++) {
        // Avoid log(0)
        double val = max(output[i], 1e-15);
        double t = sparseTarget ? ((i == (size_t)targetClass) ? 1.0 : 0.0)
                                : ((i < target.size()) ? target[i] : 0.0);
        sum += t * log(val);
      }
      return -sum;
    } else {
      // MSE
      double sum = 0.0;
      for (size_t i = 0; i < output.size(); i++) {
        double t = sparseTarget ? ((i == (size_t)targetClass) ? 1.0 : 0.0)
                                : ((i < target.size()) ? target[i] : 0.0);
        double diff = t - output[i];
        sum += diff * diff;
      }
      return sum / 2.0;
    }
  }

  double totalLoss(const vector<Sample> &data) {
    double sum = 0.0;
    for (const Sample &s : data) {
      sum += computeLoss(predict(s.inputs), s.targets);
    }
    return sum / data.size();
  }

  // Calculate accuracy (for classification tasks)
  double accuracy(const vector<Sample> &data) {
    if (data.empty())
      return 0.0;
    int correct = 0;
    for (const Sample &s : data) {
      auto out = predict(s.inputs);
      if (out.size() == 1) {
        // Binary classification: threshold at 0.5
        int predicted = (out[0] >= 0.5) ? 1 : 0;
        int actual = (s.targets[0] >= 0.5) ? 1 : 0;
        if (predicted == actual)
          correct++;
      } else {
        // Multi-class: argmax
        int predIdx = max_element(out.begin(), out.end()) - out.begin();
        int targIdx;
        if (s.targets.size() == 1 && out.size() > 1) {
          targIdx = static_cast<int>(s.targets[0]);
        } else {
          targIdx = max_element(s.targets.begin(), s.targets.end()) -
                    s.targets.begin();
        }

        if (predIdx == targIdx)
          correct++;
      }
    }
    return 100.0 * correct / data.size();
  }

  void saveWeights(const string &filename) {
    ofstream file(filename);
    if (!file) {
      cerr << "Error: Cannot save to " << filename << "\n";
      return;
    }

    file << layers.size() << "\n";
    for (const Layer &layer : layers) {
      file << layer.neurons.size() << " ";
    }
    file << "\n";
    file << static_cast<int>(activation) << "\n";

    for (const Layer &layer : layers) {
      for (const Neuron &n : layer.neurons) {
        file << n.bias;
        for (double w : n.weights) {
          file << " " << w;
        }
        file << "\n";
      }
    }
    cout << "Saved to " << filename << "\n";
  }

  bool loadWeights(const string &filename) {
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

    int actInt;
    file >> actInt;
    activation = static_cast<ActivationType>(actInt);

    layers.clear();
    for (size_t i = 0; i < topology.size(); i++) {
      int numOutputs = (i < topology.size() - 1) ? topology[i + 1] : 0;
      layers.emplace_back(topology[i], numOutputs, i == 0);
    }

    for (Layer &layer : layers) {
      for (Neuron &n : layer.neurons) {
        file >> n.bias;
        for (double &w : n.weights) {
          file >> w;
        }
      }
    }
    cout << "Loaded from " << filename
         << " (activation: " << activationName(activation) << ")\n";
    return true;
  }

private:
  vector<Layer> layers;
};

// ============================================================================
// Dataset - Provides built-in training data
// ============================================================================
class Dataset {
public:
  static vector<Sample> XOR() {
    return {{{0, 0}, {0}}, {{0, 1}, {1}}, {{1, 0}, {1}}, {{1, 1}, {0}}};
  }
};

// ============================================================================
// Helper functions for CLI
// ============================================================================
void printPredictions(NeuralNetwork &net, const vector<Sample> &data,
                      int maxRows = 0) {
  cout << fixed << setprecision(4);
  int numInputs = data[0].inputs.size();
  int numTargets = data[0].targets.size();

  // Header
  cout << "\n  Input" << string(numInputs * 8, ' ') << "| Predicted"
       << string(numTargets * 8, ' ') << "| Target\n";
  cout << "  " << string(numInputs * 8 + 5, '-') << "|"
       << string(numTargets * 8 + 9, '-') << "|"
       << string(numTargets * 8 + 6, '-') << "\n";

  int rows =
      (maxRows > 0 && maxRows < (int)data.size()) ? maxRows : data.size();
  for (int idx = 0; idx < rows; idx++) {
    const Sample &s = data[idx];
    auto out = net.predict(s.inputs);
    cout << "  [";
    for (size_t i = 0; i < s.inputs.size(); i++) {
      cout << s.inputs[i];
      if (i < s.inputs.size() - 1)
        cout << ", ";
    }
    cout << "]  |   [";
    for (size_t i = 0; i < out.size(); i++) {
      cout << out[i];
      if (i < out.size() - 1)
        cout << ", ";
    }
    cout << "]   |  [";
    for (size_t i = 0; i < s.targets.size(); i++) {
      cout << s.targets[i];
      if (i < s.targets.size() - 1)
        cout << ", ";
    }
    cout << "]\n";
  }
  if (maxRows > 0 && maxRows < (int)data.size()) {
    cout << "  ... (" << (data.size() - maxRows) << " more rows)\n";
  }
}

void printMenu() {
  cout << "\n╔══════════════════════════════════════════╗\n";
  cout << "║       NeuralNet++ Interactive Menu        ║\n";
  cout << "╠══════════════════════════════════════════╣\n";
  cout << "║  1.  Train network on XOR                ║\n";
  cout << "║  2.  Test current network                ║\n";
  cout << "║  3.  Predict custom input                ║\n";
  cout << "║  4.  Save weights to file                ║\n";
  cout << "║  5.  Load weights from file              ║\n";
  cout << "║  6.  Run full demo                       ║\n";
  cout << "║  ──────────────────────────────────────  ║\n";
  cout << "║  7.  Load CSV dataset                    ║\n";
  cout << "║  8.  Normalize loaded dataset            ║\n";
  cout << "║  9.  Split into train/test sets          ║\n";
  cout << "║  10. Train on loaded dataset             ║\n";
  cout << "║  11. Test on test set                    ║\n";
  cout << "║  ──────────────────────────────────────  ║\n";
  cout << "║  0.  Exit                                ║\n";
  cout << "╚══════════════════════════════════════════╝\n";
  cout << "  Choice: ";
}

void runDemo() {
  cout << "\n=== XOR Demo (Adam Optimizer) ===\n";
  cout << "Creating network with topology: 2 -> 4 -> 1\n";

  // Sigmoid + Adam: fast, reliable convergence on XOR
  NeuralNetwork net({2, 4, 1}, 0.05, ActivationType::SIGMOID,
                    OptimizerType::ADAM);
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

// Parse a comma-separated list of ints like "0,1,2,3"
vector<int> parseIntList(const string &s) {
  vector<int> result;
  stringstream ss(s);
  string token;
  while (getline(ss, token, ',')) {
    try {
      result.push_back(stoi(token));
    } catch (...) {
      cerr << "  Warning: '" << token
           << "' is not a valid integer, skipping.\n";
    }
  }
  return result;
}

// Parse topology string like "4,8,3"
vector<int> parseTopology(const string &s) { return parseIntList(s); }

// ============================================================================
// Main - Interactive CLI
// ============================================================================
int main(int argc, char *argv[]) {
  cout << "\n";
  cout << "  _   _                      _  _   _      _   \n";
  cout << " | \\ | | ___ _   _ _ __ __ _| || \\ | | ___| |_ \n";
  cout << " |  \\| |/ _ \\ | | | '__/ _` | ||  \\| |/ _ \\ __|\n";
  cout << " | |\\  |  __/ |_| | | | (_| | || |\\  |  __/ |_ \n";
  cout << " |_| \\_|\\___|\\__,_|_|  \\__,_|_||_| \\_|\\___|\\__| ++\n";
  cout << "\n A Powerful Neural Network Engine in C++\n";
  cout << " Now with CSV loading, normalization, and multiple activations!\n";

  // Quick command-line mode
  if (argc > 1 && string(argv[1]) == "demo") {
    runDemo();
    return 0;
  }

  // State: current network and datasets
  NeuralNetwork net({2, 3, 1}, 0.5);
  auto xorData = Dataset::XOR();
  vector<Sample> csvData;
  vector<Sample> trainData;
  vector<Sample> testData;
  vector<Sample> *activeData = &xorData;
  DataNormalizer::Stats normStats;
  bool trained = false;
  bool csvLoaded = false;
  bool dataSplit = false;
  bool normalized = false;

  while (true) {
    printMenu();

    int choice;
    if (!(cin >> choice)) {
      cin.clear();
      cin.ignore(10000, '\n');
      cout << "  Invalid input. Please enter a number.\n";
      continue;
    }

    if (choice == 0) {
      cout << "\nGoodbye!\n";
      break;
    }

    switch (choice) {
    case 1: { // Train on XOR
      int epochs;
      double lr;
      cout << "  Epochs (e.g., 5000): ";
      cin >> epochs;
      cout << "  Learning rate (e.g., 0.5): ";
      cin >> lr;

      cout << "  Activation [0=Sigmoid, 1=ReLU, 2=LeakyReLU, 3=Tanh, "
              "4=Softmax]: ";
      int actChoice;
      cin >> actChoice;
      ActivationType act = static_cast<ActivationType>(
          (actChoice >= 0 && actChoice <= 4) ? actChoice : 0);

      cout << "  Optimizer [0=SGD, 1=Adam]: ";
      int optChoice;
      cin >> optChoice;
      OptimizerType opt =
          (optChoice == 1) ? OptimizerType::ADAM : OptimizerType::SGD;

      net = NeuralNetwork({2, 3, 1}, lr, act, opt);
      activeData = &xorData;
      cout << "\nTraining with " << activationName(act) << " and "
           << (opt == OptimizerType::ADAM ? "Adam" : "SGD") << "...\n";
      net.train(*activeData, epochs);
      trained = true;

      cout << "\nResults:";
      printPredictions(net, *activeData);
      cout << "\nFinal Loss: " << fixed << setprecision(6)
           << net.totalLoss(*activeData) << "\n";
      cout << "Accuracy: " << fixed << setprecision(1)
           << net.accuracy(*activeData) << "%\n";
      break;
    }
    case 2: { // Test
      if (!trained) {
        cout << "\n  Note: Network has random weights (not trained yet)\n";
      }
      printPredictions(net, *activeData, 20);
      cout << "\nLoss: " << fixed << setprecision(6)
           << net.totalLoss(*activeData) << "\n";
      cout << "Accuracy: " << fixed << setprecision(1)
           << net.accuracy(*activeData) << "%\n";
      break;
    }
    case 3: { // Predict custom input
      cout << "  How many input values? (Expected: " << net.getInputSize()
           << "): ";
      int n;
      cin >> n;
      vector<double> inputs(n);
      for (int i = 0; i < n; i++) {
        cout << "  Input[" << i << "]: ";
        cin >> inputs[i];
      }

      auto result = net.predict(inputs);
      cout << "\n  Prediction: [";
      for (size_t i = 0; i < result.size(); i++) {
        cout << fixed << setprecision(4) << result[i];
        if (i < result.size() - 1)
          cout << ", ";
      }
      cout << "]\n";
      break;
    }
    case 4: { // Save
      string filename;
      cout << "  Filename: ";
      cin >> filename;
      net.saveWeights(filename);
      break;
    }
    case 5: { // Load
      string filename;
      cout << "  Filename: ";
      cin >> filename;
      if (net.loadWeights(filename)) {
        trained = true;
      }
      break;
    }
    case 6: { // Demo
      runDemo();
      break;
    }
    case 7: { // Load CSV
      string filename;
      cout << "  CSV file path: ";
      cin >> filename;

      cout << "  Delimiter (comma/semicolon/tab) [default: ,]: ";
      string delimStr;
      cin.ignore();
      getline(cin, delimStr);
      char delim = ',';
      if (delimStr == "semicolon" || delimStr == ";")
        delim = ';';
      else if (delimStr == "tab" || delimStr == "\\t")
        delim = '\t';

      cout << "  Input column indices (comma-separated, e.g., 0,1,2,3): ";
      string inputColStr;
      getline(cin, inputColStr);
      vector<int> inputCols = parseIntList(inputColStr);

      cout << "  Target column indices (comma-separated, e.g., 4): ";
      string targetColStr;
      getline(cin, targetColStr);
      vector<int> targetCols = parseIntList(targetColStr);

      if (inputCols.empty() || targetCols.empty()) {
        cout << "  Error: Must specify at least one input and one target "
                "column.\n";
        break;
      }

      CSVLoader loader(delim);
      csvData = loader.load(filename, inputCols, targetCols);

      if (!csvData.empty()) {
        csvLoaded = true;
        activeData = &csvData;
        dataSplit = false;
        normalized = false;
        cout << "\n  Dataset loaded successfully! Use options 8-10 to "
                "normalize, split, and train.\n";
      }
      break;
    }
    case 8: { // Normalize
      if (!csvLoaded) {
        cout << "\n  Error: Load a CSV dataset first (option 7).\n";
        break;
      }
      if (normalized) {
        cout
            << "\n  Warning: Data is already normalized. Normalizing again may "
               "distort values.\n";
        cout << "  Continue? [y/n]: ";
        char confirm;
        cin >> confirm;
        if (confirm != 'y' && confirm != 'Y')
          break;
      }
      cout << "  Normalization method [1=Min-Max, 2=Z-Score]: ";
      int method;
      cin >> method;

      DataNormalizer::Method m =
          (method == 2) ? DataNormalizer::Z_SCORE : DataNormalizer::MIN_MAX;

      normStats = DataNormalizer::normalize(csvData, m);
      normalized = true;

      // Re-apply split if already split
      if (dataSplit) {
        cout << "  Note: Re-split needed after normalization.\n";
        dataSplit = false;
      }
      break;
    }
    case 9: { // Train/test split
      if (!csvLoaded) {
        cout << "\n  Error: Load a CSV dataset first (option 7).\n";
        break;
      }
      double ratio;
      unsigned int seed;
      cout << "  Train ratio (e.g., 0.8 for 80%): ";
      cin >> ratio;
      cout << "  Random seed (0 for random): ";
      cin >> seed;

      auto result = DataSplitter::split(csvData, ratio, seed);
      trainData = result.train;
      testData = result.test;
      dataSplit = true;
      activeData = &trainData;
      cout << "  Active dataset set to training set.\n";
      break;
    }
    case 10: { // Train on loaded data
      if (!csvLoaded) {
        cout << "\n  Error: Load a CSV dataset first (option 7).\n";
        break;
      }

      cout << "  Network topology (comma-separated, e.g., 4,8,3): ";
      string topoStr;
      cin >> topoStr;
      vector<int> topology = parseTopology(topoStr);

      if (topology.size() < 2) {
        cout << "  Error: Topology must have at least 2 layers.\n";
        break;
      }

      int epochs;
      double lr;
      cout << "  Epochs: ";
      cin >> epochs;
      cout << "  Learning rate: ";
      cin >> lr;

      cout << "  Activation [0=Sigmoid, 1=ReLU, 2=LeakyReLU, 3=Tanh, "
              "4=Softmax]: ";
      int actChoice;
      cin >> actChoice;
      ActivationType act = static_cast<ActivationType>(
          (actChoice >= 0 && actChoice <= 4) ? actChoice : 0);

      cout << "  Optimizer [0=SGD, 1=Adam]: ";
      int optChoice;
      cin >> optChoice;
      OptimizerType opt =
          (optChoice == 1) ? OptimizerType::ADAM : OptimizerType::SGD;

      net = NeuralNetwork(topology, lr, act, opt);

      vector<Sample> &trainingSet = dataSplit ? trainData : csvData;
      cout << "\n  Training with " << activationName(act) << " on "
           << trainingSet.size() << " samples...\n\n";
      net.train(trainingSet, epochs);
      trained = true;

      cout << "\n  Training Results:";
      printPredictions(net, trainingSet, 10);
      cout << "\n  Train Loss: " << fixed << setprecision(6)
           << net.totalLoss(trainingSet) << "\n";
      cout << "  Train Accuracy: " << fixed << setprecision(1)
           << net.accuracy(trainingSet) << "%\n";

      if (dataSplit && !testData.empty()) {
        cout << "\n  Test Loss: " << fixed << setprecision(6)
             << net.totalLoss(testData) << "\n";
        cout << "  Test Accuracy: " << fixed << setprecision(1)
             << net.accuracy(testData) << "%\n";
      }
      break;
    }
    case 11: { // Test on test set
      if (!dataSplit || testData.empty()) {
        cout << "\n  Error: Split the dataset first (option 9).\n";
        break;
      }
      if (!trained) {
        cout << "\n  Warning: Network not trained yet.\n";
      }
      cout << "\n  Test Set Results:";
      printPredictions(net, testData, 20);
      cout << "\n  Test Loss: " << fixed << setprecision(6)
           << net.totalLoss(testData) << "\n";
      cout << "  Test Accuracy: " << fixed << setprecision(1)
           << net.accuracy(testData) << "%\n";
      break;
    }
    default:
      cout << "  Invalid choice. Try again.\n";
    }
  }

  return 0;
}
