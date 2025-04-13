# Neural Network Visualizer

A lightweight and flexible visualizer for Keras feedforward neural networks. This tool generates clean and intuitive diagrams (both horizontal and vertical) showing the architecture of your model, including:

- Number of neurons per layer
- Layer type (input, hidden, output)
- Technical information per layer (name, input/output shape, parameters)
- Activation functions (if defined)

---

## 📦 Installation

You can install this module directly from GitHub.

### 🔖 Latest tagged version (`v0.1.0-beta`)

```bash
pip install git+https://github.com/ZabakJL/nn-visualizer.git@v0.1.0-beta
```

> This version is a beta release intended for testing and feedback.

### 🛠 Manual installation (editable mode)

```bash
git clone https://github.com/ZabakJL/nn-visualizer.git
cd nn-visualizer
pip install -e .
```

---

## 🚀 Usage

```python
from nn_visualizer import plot_neural_network
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# Define model
model = Sequential([
    Input(shape=(8,)),
    Dense(12, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Plot architecture (vertical or horizontal)
plot_neural_network(model, orientation="vertical")
plot_neural_network(model, orientation="horizontal")
```

---

## 🖼️ Output Example

The visualizer generates a plot like this:

- Nodes for each neuron
- Layer-specific color backgrounds
- Arrows for inputs/outputs
- Layer info block with input/output shape, parameter count, and activation

---

## 📦 Versiones

### v0.1.0-beta – Versión de prueba

Esta es la primera versión de prueba del módulo `nn_visualizer`, pensada para exploración y retroalimentación.

#### Características incluidas:
- Soporte para orientación **horizontal** (izquierda a derecha) y **vertical** (arriba a abajo).
- Visualización de:
  - Nodos numerados
  - Flechas de entrada (`xᵢ`) y salida (`yᵢ`)
  - Información técnica de cada capa:
    - Nombre de la capa
    - Tipo
    - Forma de entrada y salida
    - Número de parámetros
    - Función de activación (si está definida)
- Ajuste automático del tamaño de la figura.
- Colores diferenciados para capas de entrada, ocultas y salida.

---

> ⚠️ Esta es una versión beta. Puede estar sujeta a cambios antes del primer release estable.

---

## 📄 License

This project is licensed under the MIT License.

Created with 💡 by [ZabakJL](https://github.com/ZabakJL)
