from tensorflow.keras.layers import Dense

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