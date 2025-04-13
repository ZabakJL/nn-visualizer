# Neural Network Visualizer

**Neural Network Visualizer** is a simple and customizable tool to visualize feedforward neural networks built with Keras `Sequential` models. It helps understand the model architecture by plotting neurons, layers, and connections in either **horizontal (left-to-right)** or **vertical (top-down)** orientation.

---

## ✨ Features

- Visualizes number of neurons per layer  
- Differentiates input, hidden, and output layers with color coding  
- Annotates each neuron with its ID  
- Displays model information such as layer name, input/output shape, and number of parameters  
- Supports both horizontal and vertical diagrams  

---

## 📦 Installation

You can install directly from GitHub:

```bash
pip install git+https://github.com/ZabakJL/nn-visualizer.git 
```

Or, if you cloned this repository:

```bash
cd nn-visualizer
pip install -e .
```

---

## 🧠 Example Usage

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from nn_visualizer import plot_neural_network

model = Sequential([
    Input(shape=(8,)),
    Dense(12, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

plot_neural_network(model, orientation='horizontal')  # or 'vertical'
```

---

## 📁 Repository Structure

```
nn-visualizer/
├── nn_visualizer/
│   └── __init__.py
├── setup.py
└── README.md
```

---

## 📄 License

This project is open source and free to use under the MIT License.
