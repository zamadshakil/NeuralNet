# NeuralNet++

A simple, self-contained feed-forward neural network engine in C++.

Built for learning — focuses on clarity and OOP composition:
```
NeuralNetwork → Layers → Neurons
```

## Features

- **Dense layers** with sigmoid activation  
- **Backpropagation** with configurable learning rate & epochs  
- **XOR demo** showing before/after predictions  
- **Save/Load** trained weights to file  
- **Interactive CLI** menu for easy use  

## Build

```bash
g++ -std=c++17 src/main.cpp -o neuralnet
```

## Usage

### Interactive Mode (recommended)
```bash
./neuralnet
```
Opens a menu where you can train, test, predict, save, and load:
```
========================================
       NeuralNet++ Interactive Menu
========================================
  1. Train network on XOR
  2. Test current network
  3. Predict custom input
  4. Save weights to file
  5. Load weights from file
  6. Run full demo
  0. Exit
----------------------------------------
```

### Quick Demo
```bash
./neuralnet demo
```
Trains on XOR and shows before/after predictions.

## Example Output

```
BEFORE training (random weights):
  Input     | Predicted | Target
  ----------|-----------|-------
  [0, 0]    |   0.65    |  0
  [0, 1]    |   0.62    |  1
  [1, 0]    |   0.64    |  1
  [1, 1]    |   0.61    |  0

Training for 5000 epochs...
  Epoch  1000 | Loss: 0.100156
  Epoch  2000 | Loss: 0.002947
  ...

AFTER training:
  Input     | Predicted | Target
  ----------|-----------|-------
  [0, 0]    |   0.01    |  0
  [0, 1]    |   0.97    |  1
  [1, 0]    |   0.97    |  1
  [1, 1]    |   0.04    |  0
```

## OOP Design

| Class | Purpose |
|-------|---------|
| `MathUtils` | Sigmoid, derivative, random weights |
| `Neuron` | Holds value, bias, gradient, weights |
| `Layer` | Collection of neurons at same depth |
| `NeuralNetwork` | Engine: predict, train, save, load |
| `Dataset` | Provides XOR training samples |

## Notes

- Topology: 2 inputs → 3 hidden → 1 output (for XOR)
- Loss: Mean Squared Error
- Weights file stores topology + all weights/biases
