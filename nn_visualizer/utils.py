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