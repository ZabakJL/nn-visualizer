import matplotlib.pyplot as plt
import matplotlib.patches as patches
from .core import extract_layer_sizes_from_model
from .utils import get_colors_by_layer_type, format_layer_info

ddef draw_feedforward_network(
    layer_sizes,
    layer_infos=None,
    orientation="vertical",
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
    max_neurons = min(max(layer_sizes), max_neurons_display) if summarized else max(layer_sizes)
    
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
    nueron_y = 0

    for layer_idx, layer_size in enumerate(layer_sizes):
        kind = "input" if layer_idx == 0 else "output" if layer_idx == len(layer_sizes) - 1 else "hidden"
        neuron_n = min(max_neurons_display, layer_size) if summarized and max_neurons_display < layer_size else layer_size
        final_neuron_id = neuron_id + layer_size
        
        if orientation.lower() in ["vertical", "v"]:
            y = (n_layers - 1 - layer_idx) * h_spacing
            x_positions = [0] if neuron_n == 1 else [i * v_spacing - (neuron_n - 1) * v_spacing / 2 for i in range(neuron_n)]
        else:
            x = layer_idx * h_spacing
            y_positions = [0] if neuron_n == 1 else [-i * v_spacing + (neuron_n - 1) * v_spacing / 2 for i in range(neuron_n)]

        #neuron_n = min(max_neurons_display, layer_size) if summarized and max_neurons_display < layer_size else layer_size
        #final_neuron_id = neuron_id + layer_size

        if summarized and max_neurons_display < layer_size:
            half = neuron_n // 2
            neuron_txt = [str(n) for n in range(neuron_id, neuron_id + half)] + ["..."] + \
                 [str(n) for n in range(neuron_id + layer_size - half, neuron_id + layer_size)]
        else:
            neuron_txt = [str(n) for n in range(neuron_id, neuron_id + layer_size)]
        
        for i in range(neuron_n):
            if orientation.lower() in ["vertical", "v"]:
                x = x_positions[i]
                pos = (x, y)
            else:
                y = y_positions[i]
                pos = (x, y)

            node_id = (layer_idx, i)
            positions[node_id] = pos

            ax.add_patch(plt.Circle(pos, radius=node_radius, color=colors[kind]["fill"], zorder=3))
            ax.text(*pos, str(neuron_txt[nueron_y]), fontsize=8, ha='center', va='center', color='white')
            neuron_id += 1
            nueron_y += 1
            
            # x_i and y_i arrows
            if layer_idx == 0:
                if orientation.lower() in ["vertical", "v"]:
                    ax.annotate(fr'$x_{{{neuron_txt[i]}}}$', xy=(x, y + node_radius), xytext=(x, y + 0.9),
                                arrowprops=dict(arrowstyle='->', lw=0.8), ha='center', va='bottom', fontsize=10)
                else:
                    ax.annotate(fr'$x_{{{neuron_txt[i]}}}$', xy=(x - node_radius, y), xytext=(x - 0.9, y),
                                arrowprops=dict(arrowstyle='->', lw=0.8), ha='right', va='center', fontsize=10)
            if layer_idx == len(layer_sizes) - 1:
                if orientation.lower() in ["vertical", "v"]:
                    ax.annotate(fr'$y_{{{neuron_txt[i]}}}$', xy=(x, y - node_radius - 0.05), xytext=(x, y - 0.9),
                                arrowprops=dict(arrowstyle='<-', lw=1.0), ha='center', va='top', fontsize=10)
                else:
                    ax.annotate(fr'$y_{{{neuron_txt[i]}}}$', xy=(x + node_radius + 0.1, y), xytext=(x + node_radius + 0.5, y),
                                arrowprops=dict(arrowstyle='<-', lw=0.8), ha='left', va='center', fontsize=10)

        if summarized==True and max_neurons_display < layer_size:
            neuron_id = int(neuron_txt[nueron_y-1])+1
            nueron_y = 0
        else:
            nueron_y = 0
        
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
        if show_layer_info==True:
            if layer_infos and layer_idx > 0 and (layer_idx - 1) < len(layer_infos):
                info = format_layer_info(layer_infos[layer_idx - 1])
                if orientation.lower() in ["vertical", "v"]:
                    ax.text(x0 + box_width + 0.3, y, info, ha='left', va='center', fontsize=8, family='monospace')
                else:
                    ax.text(x, y0 - 0.5, info, ha='center', va='top', fontsize=8, family='monospace')
        
    # Connections
    layer_sizes= [min(n, max_neurons_display) for n in layer_sizes]
    
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
def get_colors_by_layer_type():
    return {
        "input":  {"fill": "#27ae60", "box": "#d4efdf"},
        "hidden": {"fill": "#2e86c1", "box": "#d4e6f1"},
        "output": {"fill": "#c0392b", "box": "#f9e1e0"}
    }

def format_layer_info(info_dict):
    return (f"{info_dict['name']} ({info_dict['type']})\n"
            f"In: {info_dict['input_shape']}\n"
            f"Out: {info_dict['output_shape']}\n"
            f"Params: {info_dict['params']}\n"
            f"Act: {info_dict['activation']}")

def extract_layer_sizes_from_model(model):
    """
    Extracts the number of neurons per layer and metadata from a Keras Sequential model.

    Returns:
    --------
    - layer_sizes: List[int]
    - layer_infos: List[dict]
    """
    layer_sizes = []
    layer_infos = []

    if not model.built:
        try:
            input_dim = model.layers[0].input_shape[-1]
            model.build(input_shape=(None, input_dim))
        except Exception as e:
            raise RuntimeError("Model must be built or compiled first.") from e

    input_dim = model.input_shape[-1]
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
                "params": layer.count_params(),
                "activation": layer.activation.__name__
            }

            layer_infos.append(info)

    return layer_sizes, layer_infos


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
