# Neural Network Visualizer

A lightweight and flexible visualizer for Keras feedforward neural networks. This tool generates clean and intuitive diagrams (both horizontal and vertical) showing the architecture of your model, including:

- Number of neurons per layer
- Layer type (input, hidden, output)
- Technical information per layer (name, input/output shape, parameters)
- Activation functions (if defined)

---

## 📦 Installation

You can install this module directly from GitHub.

### 🔖 Latest stable version (`v0.2.0`)

```bash
pip install git+https://github.com/ZabakJL/nn-visualizer.git@v0.2.0
```

> This version is the first modular and semi-stable release including activation function display.

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
    Dense(64, activation='relu'),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Plot architecture with options
plot_neural_network(
    model,
    orientation="horizontal",        # or "vertical"
    summarized=True,                 # show condensed neuron layout
    max_neurons_display=19,          # max neurons per layer to display
    show_layer_info=True             # show technical layer metadata
)
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

### v0.2.0 – Summarized layers and configurable display

- Added support for summarized visualizations of large layers using ellipsis ("...") markers.
- New arguments in `draw_feedforward_network` and `plot_neural_network`:
  - `summarized`: toggle compact mode (default: `True`)
  - `max_neurons_display`: limit number of neurons per layer (default: `19`)
  - `show_layer_info`: toggle layer metadata display (default: `True`)
- Improved documentation and parameter forwarding.
- Backward compatible with earlier usage.

### v0.1.1 – Modular organization and activation display

- Modularization into `core.py`, `visualizer.py`, and `utils.py`
- Added display of layer activation functions
- Improved labeling and figure scaling
- Enhanced vertical/horizontal layout support

### v0.1.0-beta – Versión de prueba

- First beta version with basic horizontal and vertical layout
- Displays layer names, shapes, parameter counts, and neuron IDs
- Color-coded background per layer type

---

## 📄 License

This project is licensed under the MIT License.

Created with 💡 by [ZabakJL](https://github.com/ZabakJL)
