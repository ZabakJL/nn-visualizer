import matplotlib.pyplot as plt
import matplotlib.patches as patches
from .core import extract_layer_sizes_from_model
from .utils import get_colors_by_layer_type, format_layer_info

def draw_feedforward_network(layer_sizes, layer_infos=None, orientation="horizontal"):
    colors = get_colors_by_layer_type()
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
            x_positions = [0] if layer_size == 1 else [i * v_spacing - ((layer_size - 1) * v_spacing / 2) for i in range(layer_size)]
        else:
            x = layer_idx * h_spacing
            y_positions = [0] if layer_size == 1 else [-i * v_spacing + ((layer_size - 1) * v_spacing / 2) for i in range(layer_size)]

        for i in range(layer_size):
            pos = (x_positions[i], y) if orientation.lower() in ["vertical", "v"] else (x, y_positions[i])
            positions[(layer_idx, i)] = pos

            circle = plt.Circle(pos, radius=node_radius, color=colors[kind]["fill"], zorder=3)
            ax.add_patch(circle)
            ax.text(*pos, str(neuron_id), fontsize=8, ha='center', va='center', color='white')
            neuron_id += 1

            if layer_idx == 0:
                label_pos = (pos[0], pos[1] + 0.9) if orientation.lower() in ["vertical", "v"] else (pos[0] - 0.9, pos[1])
                ax.annotate(f'$x_{{{i+1}}}$', xy=pos, xytext=label_pos, arrowprops=dict(arrowstyle='->'), ha='center' if orientation == 'v' else 'right', va='bottom' if orientation == 'v' else 'center', fontsize=10)

            if layer_idx == len(layer_sizes) - 1:
                label_pos = (pos[0], pos[1] - 0.9) if orientation.lower() in ["vertical", "v"] else (pos[0] + 0.5, pos[1])
                ax.annotate(f'$y_{{{i+1}}}$', xy=pos, xytext=label_pos, arrowprops=dict(arrowstyle='<-', lw=0.8), ha='center' if orientation == 'v' else 'left', va='top' if orientation == 'v' else 'center', fontsize=10)

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

        rect = patches.FancyBboxPatch((x0, y0), box_width, box_height, boxstyle="round,pad=0.02", linewidth=0, facecolor=colors[kind]["box"], zorder=0)
        ax.add_patch(rect)

        label = "Capa\nentrada" if layer_idx == 0 else "Capa\nsalida" if layer_idx == len(layer_sizes) - 1 else f"Capa\noculta {layer_idx}"
        ax.text(x0 - 0.5 if orientation == 'v' else x, y if orientation == 'v' else y0 + box_height + 0.6, label, ha='right' if orientation == 'v' else 'center', va='center', fontsize=9, style='italic')

        if layer_infos and layer_idx > 0 and (layer_idx - 1) < len(layer_infos):
            info = format_layer_info(layer_infos[layer_idx - 1])
            if orientation.lower() in ["vertical", "v"]:
                ax.text(x0 + box_width + 0.3, y, info, ha='left', va='center', fontsize=8, family='monospace')
            else:
                ax.text(x, y0 - 0.5, info, ha='center', va='top', fontsize=8, family='monospace')

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

def plot_neural_network(model, orientation="horizontal"):
    layer_sizes, layer_infos = extract_layer_sizes_from_model(model)
    draw_feedforward_network(layer_sizes, layer_infos, orientation=orientation)