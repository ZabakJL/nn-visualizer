import matplotlib.pyplot as plt
import matplotlib.patches as patches
from .core import extract_layer_sizes_from_model
from .utils import get_colors_by_layer_type, format_layer_info

def draw_feedforward_network(
    layer_sizes,
    layer_infos=None,
    orientation="horizontal",
    summarized=True,
    max_neurons_display=19,
    show_layer_info=True
):
    """
    Draws a feedforward neural network architecture with optional summarization.

    Parameters:
    -----------
    layer_sizes : list of int
        A list indicating the number of neurons in each layer.
    layer_infos : list of dict, optional
        Technical information for each layer, shown in monospaced text.
    orientation : str, default="horizontal"
        Layout direction of the network: "horizontal" or "vertical".
    summarized : bool, default=True
        Whether to limit the number of neurons drawn per layer to keep the diagram concise.
    max_neurons_display : int, default=19
        Maximum number of neurons to draw per layer in summarized mode.
    show_layer_info : bool, default=True
        Whether to show the technical information associated with each layer.
    """
    colors = get_colors_by_layer_type()
    n_layers = len(layer_sizes)
    max_neurons = min(max(layer_sizes), max_neurons_display)

    # Set layout spacing and node appearance
    h_spacing = 3.0
    v_spacing = 1.5
    node_radius = 0.45

    # Adjust figure dimensions based on orientation
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
        # Determine the type of layer
        kind = "input" if layer_idx == 0 else "output" if layer_idx == len(layer_sizes) - 1 else "hidden"

        actual_size = layer_size
        indices_to_draw = list(range(layer_size))

        # Summarize if the layer has too many neurons
        if summarized and layer_size > max_neurons_display:
            half = max_neurons_display // 2
            indices_to_draw = list(range(half)) + ["..."] + list(range(layer_size - half, layer_size))
            actual_size = max_neurons_display

        # Compute positions for neurons
        if orientation.lower() in ["vertical", "v"]:
            y = (n_layers - 1 - layer_idx) * h_spacing
            x_positions = [0] if actual_size == 1 else [i * v_spacing - ((actual_size - 1) * v_spacing / 2) for i in range(actual_size)]
        else:
            x = layer_idx * h_spacing
            y_positions = [0] if actual_size == 1 else [-i * v_spacing + ((actual_size - 1) * v_spacing / 2) for i in range(actual_size)]

        draw_idx = 0
        for idx in indices_to_draw:
            # If skipping neurons, show ellipsis
            if idx == "...":
                label = "..."
                pos = (x_positions[draw_idx], y) if orientation.lower() in ["vertical", "v"] else (x, y_positions[draw_idx])
                ax.text(*pos, label, fontsize=10, ha='center', va='center', color='black')
                draw_idx += 1
                continue

            # Calculate position and store it
            pos = (x_positions[draw_idx], y) if orientation.lower() in ["vertical", "v"] else (x, y_positions[draw_idx])
            positions[(layer_idx, idx)] = pos

            # Draw neuron as a circle
            circle = plt.Circle(pos, radius=node_radius, color=colors[kind]["fill"], zorder=3)
            ax.add_patch(circle)
            ax.text(*pos, str(neuron_id), fontsize=8, ha='center', va='center', color='white')
            neuron_id += 1

            # Draw input labels
            if layer_idx == 0:
                label_pos = (pos[0], pos[1] + 0.9) if orientation.lower() in ["vertical", "v"] else (pos[0] - 0.9, pos[1])
                ax.annotate(f'$x_{{{idx+1}}}$', xy=pos, xytext=label_pos, arrowprops=dict(arrowstyle='->'),
                            ha='center' if orientation == 'v' else 'right',
                            va='bottom' if orientation == 'v' else 'center', fontsize=10)

            # Draw output labels
            if layer_idx == len(layer_sizes) - 1:
                label_pos = (pos[0], pos[1] - 0.9) if orientation.lower() in ["vertical", "v"] else (pos[0] + 0.5, pos[1])
                ax.annotate(f'$y_{{{idx+1}}}$', xy=pos, xytext=label_pos, arrowprops=dict(arrowstyle='<-', lw=0.8),
                            ha='center' if orientation == 'v' else 'left',
                            va='top' if orientation == 'v' else 'center', fontsize=10)

            draw_idx += 1

        # Draw layer box
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

        # Draw layer label
        label = "Input\nlayer" if layer_idx == 0 else "Output\nlayer" if layer_idx == len(layer_sizes) - 1 else f"Hidden\nlayer {layer_idx}"
        ax.text(x0 - 0.5 if orientation == 'v' else x, y if orientation == 'v' else y0 + box_height + 0.6, label, ha='right' if orientation == 'v' else 'center', va='center', fontsize=9, style='italic')

        # Draw layer info if available
        if show_layer_info and layer_infos and layer_idx > 0 and (layer_idx - 1) < len(layer_infos):
            info = format_layer_info(layer_infos[layer_idx - 1])
            if orientation.lower() in ["vertical", "v"]:
                ax.text(x0 + box_width + 0.3, y, info, ha='left', va='center', fontsize=8, family='monospace')
            else:
                ax.text(x, y0 - 0.5, info, ha='center', va='top', fontsize=8, family='monospace')

    # Draw connections between neurons
    for l in range(len(layer_sizes) - 1):
        layer_from = layer_sizes[l]
        layer_to = layer_sizes[l + 1]

        from_indices = list(range(layer_from))
        to_indices = list(range(layer_to))

        if summarized and layer_from > max_neurons_display:
            half = max_neurons_display // 2
            from_indices = list(range(half)) + list(range(layer_from - half, layer_from))
        if summarized and layer_to > max_neurons_display:
            half = max_neurons_display // 2
            to_indices = list(range(half)) + list(range(layer_to - half, layer_to))

        for i in from_indices:
            for j in to_indices:
                if (l, i) in positions and (l + 1, j) in positions:
                    x0, y0 = positions[(l, i)]
                    x1, y1 = positions[(l + 1, j)]
                    if orientation.lower() in ["vertical", "v"]:
                        ax.plot([x0, x1], [y0 - node_radius, y1 + node_radius], 'k-', lw=0.5, zorder=1)
                    else:
                        ax.plot([x0 + node_radius, x1 - node_radius], [y0, y1], 'k-', lw=0.5, zorder=1)

    # Final layout adjustments
    ax.set_aspect('equal')
    plt.subplots_adjust(top=0.92, bottom=0.08)
    plt.show()


def plot_neural_network(
    model,
    orientation="vertical",
    summarized=True,
    max_neurons_display=19,
    show_layer_info=True
):
    """
    Plots a feedforward neural network diagram from a Keras sequential model.

    Parameters:
    -----------
    model : keras.Model
        A sequential Keras model to be visualized.
    orientation : str, default="horizontal"
        Layout direction of the network: "horizontal" or "vertical".
    summarized : bool, default=True
        Whether to limit the number of neurons drawn per layer.
    max_neurons_display : int, default=19
        Maximum number of neurons to display in each layer.
    show_layer_info : bool, default=True
        Whether to display technical layer information.
    """
    layer_sizes, layer_infos = extract_layer_sizes_from_model(model)
    draw_feedforward_network(
        layer_sizes,
        layer_infos,
        orientation=orientation,
        summarized=summarized,
        max_neurons_display=max_neurons_display,
        show_layer_info=show_layer_info
    )
