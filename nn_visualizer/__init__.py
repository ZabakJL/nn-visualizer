# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 16:24:59 2025

@author: jpmog
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def extract_layer_sizes_from_model(model):
    """
    Extracts the number of neurons per layer and technical info from a Keras Sequential model.

    Returns:
    --------
    - layer_sizes: List[int]
    - layer_infos: List[dict]
    """
    from tensorflow.keras.layers import Dense

    layer_sizes = []
    layer_infos = []

    # Ensure model is built
    if not model.built:
        try:
            first_layer = model.layers[0]
            input_dim = first_layer.input_shape[-1]
            model.build(input_shape=(None, input_dim))
        except Exception as e:
            print("Error building model:", e)
            return [], []

    # Get input dimension
    try:
        input_dim = model.input_shape[-1]
    except:
        input_dim = model.layers[0].input_shape[-1]

    layer_sizes.append(input_dim)

    for layer in model.layers:
        if isinstance(layer, Dense):
            layer_sizes.append(layer.units)

            try:
                input_shape = tuple(layer.input.shape)
                output_shape = tuple(layer.output.shape)
            except:
                input_shape = output_shape = "(unknown)"

            info = {
                "name": layer.name,
                "type": layer.__class__.__name__,
                "input_shape": input_shape,
                "output_shape": output_shape,
                "params": layer.count_params()
            }

            layer_infos.append(info)

    return layer_sizes, layer_infos


def draw_feedforward_network(layer_sizes, layer_infos=None, orientation="horizontal"):
    """
    Draws a feedforward neural network in horizontal or vertical orientation.

    Parameters:
    -----------
    layer_sizes : list of int
        Number of neurons per layer. Example: [4, 5, 3, 1]
    layer_infos : list of dict, optional
        Technical layer info (except input layer). Each dict contains:
            - name
            - type
            - input_shape
            - output_shape
            - params
    orientation : str
        "horizontal" (left to right) or "vertical" (top to bottom)
    """
    colors = {
        "input":  {"fill": "#27ae60", "box": "#d4efdf"},
        "hidden": {"fill": "#2e86c1", "box": "#d4e6f1"},
        "output": {"fill": "#c0392b", "box": "#f9e1e0"}
    }

    n_layers = len(layer_sizes)
    max_neurons = max(layer_sizes)

    h_spacing = 3.0
    v_spacing = 1.5
    node_radius = 0.45

    if orientation.lower() in ["vertical", "v"]:
        fig_width = max(6, max_neurons * 1.0)
        fig_height = max(12, n_layers * 3.5)
    else:
        fig_width = max(12, n_layers * 3.5)
        fig_height = max(6, max_neurons * 1.0)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('off')

    positions = {}
    neuron_id = 1

    for layer_idx, layer_size in enumerate(layer_sizes):
        kind = "input" if layer_idx == 0 else "output" if layer_idx == len(layer_sizes) - 1 else "hidden"

        if orientation.lower() in ["vertical", "v"]:
            y = (n_layers - 1 - layer_idx) * h_spacing
            x_positions = [0] if layer_size == 1 else [i * v_spacing - (layer_size - 1) * v_spacing / 2 for i in range(layer_size)]
        else:
            x = layer_idx * h_spacing
            y_positions = [0] if layer_size == 1 else [-i * v_spacing + (layer_size - 1) * v_spacing / 2 for i in range(layer_size)]

        for i in range(layer_size):
            if orientation.lower() in ["vertical", "v"]:
                x = x_positions[i]
                pos = (x, y)
            else:
                y = y_positions[i]
                pos = (x, y)

            node_id = (layer_idx, i)
            positions[node_id] = pos

            ax.add_patch(plt.Circle(pos, radius=node_radius, color=colors[kind]["fill"], zorder=3))
            ax.text(*pos, str(neuron_id), fontsize=8, ha='center', va='center', color='white')
            neuron_id += 1

            # x_i and y_i arrows
            if layer_idx == 0:
                if orientation.lower() in ["vertical", "v"]:
                    ax.annotate(f'$x_{{{i+1}}}$', xy=(x, y + node_radius), xytext=(x, y + 0.9),
                                arrowprops=dict(arrowstyle='->'), ha='center', va='bottom', fontsize=10)
                else:
                    ax.annotate(f'$x_{{{i+1}}}$', xy=(x - node_radius, y), xytext=(x - 0.9, y),
                                arrowprops=dict(arrowstyle='->'), ha='right', va='center', fontsize=10)
            if layer_idx == len(layer_sizes) - 1:
                if orientation.lower() in ["vertical", "v"]:
                    ax.annotate(f'$y_{{{i+1}}}$', xy=(x, y - node_radius - 0.05), xytext=(x, y - 0.9),
                                arrowprops=dict(arrowstyle='<-', lw=0.8), ha='center', va='top', fontsize=10)
                else:
                    ax.annotate(f'$y_{{{i+1}}}$', xy=(x + node_radius + 0.1, y), xytext=(x + node_radius + 0.5, y),
                                arrowprops=dict(arrowstyle='<-', lw=0.8), ha='left', va='center', fontsize=10)

        # Background box
        if orientation.lower() in ["vertical", "v"]:
            box_width = max_neurons * v_spacing
            box_height = h_spacing - 0.5
            x0 = -box_width / 2
            y0 = y - box_height / 2
        else:
            box_width = h_spacing - 0.5
            box_height = (max_neurons - 1) * v_spacing + 2.0
            x0 = x - box_width / 2
            y0 = -box_height / 2

        ax.add_patch(patches.FancyBboxPatch(
            (x0, y0), box_width, box_height,
            boxstyle="round,pad=0.02", linewidth=0,
            facecolor=colors[kind]["box"], zorder=0
        ))

        # Layer label
        label = "Input\nLayer" if layer_idx == 0 else "Output\nLayer" if layer_idx == len(layer_sizes) - 1 else f"Hidden\nLayer {layer_idx}"
        if orientation.lower() in ["vertical", "v"]:
            ax.text(x0 - 0.5, y, label, ha='right', va='center', fontsize=9, style='italic')
        else:
            ax.text(x, y0 + box_height + 0.6, label, ha='center', fontsize=9, style='italic')

        # Technical info
        if layer_infos and layer_idx > 0 and (layer_idx - 1) < len(layer_infos):
            info_dict = layer_infos[layer_idx - 1]
            info = (f"{info_dict['name']} ({info_dict['type']})\n"
                    f"In: {info_dict['input_shape']}\n"
                    f"Out: {info_dict['output_shape']}\n"
                    f"Params: {info_dict['params']}")
            if orientation.lower() in ["vertical", "v"]:
                ax.text(x0 + box_width + 0.3, y, info, ha='left', va='center', fontsize=8, family='monospace')
            else:
                ax.text(x, y0 - 0.5, info, ha='center', va='top', fontsize=8, family='monospace')

    # Connections
    for l in range(len(layer_sizes) - 1):
        for i in range(layer_sizes[l]):
            for j in range(layer_sizes[l + 1]):
                x0, y0 = positions[(l, i)]
                x1, y1 = positions[(l + 1, j)]
                if orientation.lower() in ["vertical", "v"]:
                    ax.plot([x0, x1], [y0 - node_radius, y1 + node_radius], 'k-', lw=0.5, zorder=1)
                else:
                    ax.plot([x0 + node_radius, x1 - node_radius], [y0, y1], 'k-', lw=0.5, zorder=1)

    ax.set_aspect('equal')
    plt.subplots_adjust(top=0.92, bottom=0.08)
    plt.show()


def plot_neural_network(model, orientation="vertical"):
    """
    Plots a Keras Sequential model as a neural network diagram.

    Parameters:
    -----------
    model : keras.Sequential
        Keras neural network model.
    orientation : str
        "horizontal" or "h" → left to right
        "vertical" or "v" → top to bottom

    Raises:
    -------
    ValueError if orientation is invalid.
    """
    layer_sizes, layer_infos = extract_layer_sizes_from_model(model)
    orientation = orientation.lower()
    if orientation not in ["horizontal", "h", "vertical", "v"]:
        raise ValueError("Orientation must be 'horizontal' (or 'h') or 'vertical' (or 'v')")
    draw_feedforward_network(layer_sizes, layer_infos, orientation=orientation)
