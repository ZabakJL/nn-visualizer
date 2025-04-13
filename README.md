# Neural Network Visualizer

A lightweight and flexible visualizer for Keras feedforward neural networks. This tool generates clean and intuitive diagrams (both horizontal and vertical) showing the architecture of your model, including:

- Number of neurons per layer
- Layer type (input, hidden, output)
- Technical information per layer (name, input/output shape, parameters)
- Activation functions (if defined)

---

## 📦 Installation

You can install this module directly from GitHub:

```bash
pip install git+https://github.com/ZabakJL/nn-visualizer.git
```

Or clone the repo and install it manually:

```bash
git clone https://github.com/ZabakJL/nn-visualizer.git
cd nn-visualizer
pip install .
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

## ✨ Features

- Orientation: horizontal or vertical
- Automatic layout scaling
- Color-coded layer blocks
- Annotation of activation functions and technical info
- Ready for Jupyter notebooks or saving as images

---

## 🛠️ Requirements

- Python 3.7+
- TensorFlow (for `keras.models`)
- Matplotlib

---

## 📦 Versiones

### v1.0.0 – Primera versión estable

Esta versión introduce la funcionalidad principal del módulo `nn_visualizer`, permitiendo visualizar modelos `Sequential` de Keras como diagramas de red neuronal con información técnica por capa.

#### Características destacadas:
- Soporte para orientación **horizontal** (izquierda a derecha) y **vertical** (arriba a abajo).
- Visualización de:
  - Número de neurona por nodo
  - Etiquetas de entrada `xᵢ` y salida `yᵢ` con flechas
  - Información técnica por capa:
    - Nombre de la capa
    - Tipo de capa
    - Forma de entrada y salida
    - Número de parámetros
    - Función de activación (si está definida)
- Colores diferenciados por tipo de capa: entrada, oculta, salida.
- Ajuste automático del tamaño de la figura según el modelo.

#### Ejemplo de uso:
```python
from nn_visualizer import plot_neural_network

plot_neural_network(model, orientation="v")  # Vertical
plot_neural_network(model, orientation="h")  # Horizontal
```

---

## 📄 License

This project is licensed under the MIT License.

Created with 💡 by [ZabakJL](https://github.com/ZabakJL)
