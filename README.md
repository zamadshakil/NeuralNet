# NeuralNet++

NeuralNet++ is a tiny, self-contained feed-forward neural network engine written in modern C++. It focuses on clarity and OOP composition (Network → Layers → Neurons) while still supporting training with backpropagation, persistence, and a simple CLI for demos.

## Features
- Dense fully-connected layers with sigmoid activation
- Backpropagation + gradient descent with configurable learning rate and epochs
- Built-in XOR dataset demo with before/after predictions and loss
- Save/load topology and weights to reuse trained models
- CLI commands for train, test, predict, demo, save, and load

## Build
```
g++ -std=c++17 src/main.cpp -o neuralnet
```

## CLI usage
```
./neuralnet demo [--epochs N] [--lr V]
./neuralnet train [--epochs N] [--lr V] [--save file]
./neuralnet test [--weights file]
./neuralnet predict x y [--weights file]
./neuralnet save file
./neuralnet load file
```

### Quick demo
Runs XOR training with defaults (epochs=5000, lr=0.5) and prints predictions before/after:
```
./neuralnet demo
```

### Train and save
```
./neuralnet train --epochs 8000 --lr 0.4 --save xor.weights
```

### Test a saved model
```
./neuralnet test --weights xor.weights
```

### Predict custom input
```
./neuralnet predict 1 0 --weights xor.weights
```

## Notes
- Model files store the topology, weights, and biases so training progress is preserved across runs.
- Loss is Mean Squared Error; sigmoid derivative is used for backpropagation.
- Default topology for the XOR task is 2-3-1; adjust in code if experimenting with other shapes.
