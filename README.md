# NeuralNet++

A powerful, single-file C++ neural network engine.
It demonstrates OOP composition, memory management, backpropagation, and now includes a full CSV data pipeline.

## 🚀 Features

- **Clean OOP design**: `NeuralNetwork` → `Layer` → `Neuron`
- **CSV Loading**: Load headers, handle quotes, select input/target columns
- **Preprocessing**: Min-Max & Z-Score normalization, randomized Train/Test splitting
- **Activations**: Sigmoid, ReLU, Leaky ReLU, Tanh, **Softmax**
- **Loss Functions**: MSE (regression) and **Cross-Entropy** (classification)
- **Optimizers**: SGD and **Adam** (adaptive learning rate)
- **Training**: Backpropagation with configurable topology and learning rate
- **Persistence**: Save and load model weights
- **Interactive CLI**: Menu-driven workflow for the entire pipeline

## 📦 Getting Started

### 1. Compilation

Requires a C++17 compiler (GCC, Clang, MSVC).

```bash
g++ -std=c++17 -o neuralnet src/main.cpp
```

### 2. Running

**Interactive Mode:**

```bash
./neuralnet
```

**Quick Demo (XOR):**

```bash
./neuralnet demo
```

## 🧠 Workflow Example (Iris Dataset)

1. **Load CSV**: Select option **7** and enter `data/iris.csv`.
   - Columns: Inputs `0,1,2,3`, Target `4`
2. **Normalize**: Select option **8** (Min-Max recommended).
3. **Split**: Select option **9** (e.g., 0.8 split).
4. **Train**: Select option **10**.
   - Topology: `4,8,3` (4 inputs, hidden layer of 8, 3 outputs)
   - Activation: `Softmax` (recommended for classification) or `Tanh`
   - Optimizer: `Adam` (recommended) or `SGD`
5. **Test**: Select option **11** to verify accuracy on the test set.

## 🏗️ Architecture

- **`NeuralNetwork`**: Manages layers and training loop.
- **`Layer` / `Neuron`**: Core structural components with Adam optimizer caches.
- **`MathUtils`**: Activation functions, derivatives, and Softmax.
- **`CSVLoader`**: Parses text files into `Sample` vectors.
- **`DataNormalizer`**: Scaling utilities.
- **`DataSplitter`**: Random shuffling and splitting.

## 💾 File Format

Weights are saved as plain text:

1. Topology (layer sizes)
2. Activation function ID
3. Weights & Biases for every neuron

## 🤝 Contributing

Feel free to open issues or PRs for new features like:

- Dropout regularization
- Real-time loss visualization
- Convolutional layers

---
**Author**: Zamad Shakeel
