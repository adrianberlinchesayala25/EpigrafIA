"""
Script to convert Keras 3 models to TensorFlow.js format
Compatible with TensorFlow.js 4.x
"""
import os
import json
import numpy as np

def convert_keras3_to_tfjs(model_path: str, output_dir: str):
    """Convert a Keras 3 model to TensorFlow.js layers format"""
    
    import keras
    
    print(f"🔄 Loading model from {model_path}...")
    model = keras.models.load_model(model_path)
    model.summary()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Build a simplified layer topology that TensorFlow.js can understand
    print("📦 Building TensorFlow.js compatible topology...")
    
    layers_config = []
    
    for i, layer in enumerate(model.layers):
        layer_class = layer.__class__.__name__
        layer_config = {
            "class_name": layer_class,
            "config": {
                "name": layer.name,
                "trainable": layer.trainable,
                "dtype": "float32"
            }
        }
        
        # Add layer-specific configs
        if layer_class == "InputLayer":
            # Handle both single input and multiple inputs
            if hasattr(layer, 'input') and hasattr(layer.input, 'shape'):
                layer_config["config"]["batch_input_shape"] = list(layer.input.shape)
            elif hasattr(layer, 'output') and hasattr(layer.output, 'shape'):
                layer_config["config"]["batch_input_shape"] = list(layer.output.shape)
            else:
                # Get from layer config
                cfg = layer.get_config()
                layer_config["config"]["batch_input_shape"] = cfg.get('batch_shape', [None, 94, 120])
            layer_config["config"]["sparse"] = False
        elif layer_class == "Conv1D":
            layer_config["config"].update({
                "filters": layer.filters,
                "kernel_size": list(layer.kernel_size),
                "strides": list(layer.strides),
                "padding": layer.padding,
                "data_format": layer.data_format,
                "dilation_rate": list(layer.dilation_rate),
                "activation": "linear",
                "use_bias": layer.use_bias
            })
        elif layer_class == "BatchNormalization":
            # TensorFlow.js expects axis as integer, not list
            axis_val = layer.axis
            if isinstance(axis_val, (list, tuple)):
                axis_val = axis_val[0] if len(axis_val) == 1 else -1
            layer_config["config"].update({
                "axis": axis_val,  # Integer, not list
                "momentum": float(layer.momentum),
                "epsilon": float(layer.epsilon),
                "center": layer.center,
                "scale": layer.scale
            })
        elif layer_class == "ReLU":
            layer_config["class_name"] = "Activation"
            layer_config["config"]["activation"] = "relu"
        elif layer_class == "MaxPooling1D":
            layer_config["config"].update({
                "pool_size": list(layer.pool_size),
                "strides": list(layer.strides),
                "padding": layer.padding
            })
        elif layer_class == "Dropout":
            layer_config["config"]["rate"] = float(layer.rate)
        elif layer_class == "GlobalAveragePooling1D":
            layer_config["config"]["data_format"] = "channels_last"
        elif layer_class == "Dense":
            activation_name = "linear"
            if hasattr(layer.activation, '__name__'):
                activation_name = layer.activation.__name__
            elif hasattr(layer.activation, 'name'):
                activation_name = layer.activation.name
            layer_config["config"].update({
                "units": layer.units,
                "activation": activation_name,
                "use_bias": layer.use_bias
            })
        
        layers_config.append(layer_config)
    
    # Build Sequential-style model topology
    model_topology = {
        "class_name": "Sequential",
        "config": {
            "name": model.name,
            "layers": layers_config
        },
        "keras_version": "2.15.0",  # TF.js compatible version
        "backend": "tensorflow"
    }
    
    # Build weights manifest
    print("⚖️ Extracting weights...")
    weights_data = []
    weights_manifest = []
    
    for layer in model.layers:
        layer_weights = layer.get_weights()
        layer_name = layer.name
        
        # Get weight types based on layer class
        layer_class = layer.__class__.__name__
        
        if layer_class == "Conv1D" or layer_class == "Dense":
            weight_types = ["kernel", "bias"]
        elif layer_class == "BatchNormalization":
            weight_types = ["gamma", "beta", "moving_mean", "moving_variance"]
        else:
            weight_types = [f"weight_{i}" for i in range(len(layer_weights))]
        
        for i, w in enumerate(layer_weights):
            # Create unique TF.js compatible weight name
            if i < len(weight_types):
                weight_name = f"{layer_name}/{weight_types[i]}"
            else:
                weight_name = f"{layer_name}/weight_{i}"
            
            weights_manifest.append({
                "name": weight_name,
                "shape": list(w.shape),
                "dtype": "float32"
            })
            weights_data.append(w.astype(np.float32))
            print(f"   {weight_name}: {w.shape}")
    
    # Serialize weights to binary
    print("💾 Serializing weights...")
    all_weights = np.concatenate([w.flatten() for w in weights_data])
    weights_bytes = all_weights.tobytes()
    
    # Build model.json
    model_json = {
        "format": "layers-model",
        "generatedBy": "EpigrafIA tfjs converter v2.0",
        "convertedBy": "keras2tfjs-manual",
        "modelTopology": model_topology,
        "weightsManifest": [{
            "paths": ["group1-shard1of1.bin"],
            "weights": weights_manifest
        }]
    }
    
    # Save model.json
    model_json_path = os.path.join(output_dir, "model.json")
    with open(model_json_path, 'w', encoding='utf-8') as f:
        json.dump(model_json, f)
    print(f"✅ Model config saved: {model_json_path}")
    
    # Save weights
    weights_path = os.path.join(output_dir, "group1-shard1of1.bin")
    with open(weights_path, 'wb') as f:
        f.write(weights_bytes)
    print(f"✅ Weights saved: {weights_path} ({len(weights_bytes) / 1024:.1f} KB)")
    
    # Save labels
    labels = ["Español", "Inglés", "Francés", "Alemán"]
    labels_path = os.path.join(output_dir, "labels.json")
    with open(labels_path, 'w', encoding='utf-8') as f:
        json.dump(labels, f, ensure_ascii=False)
    print(f"✅ Labels saved: {labels_path}")
    
    print(f"\n🎉 Conversion complete!")
    print(f"   Total weights: {len(all_weights):,} parameters")
    print(f"   Output: {output_dir}/")
    
    return True


if __name__ == "__main__":
    model_path = "outputs/models_trained/language_model_best.keras"
    output_dir = "frontend/public/models/language"
    
    if os.path.exists(model_path):
        convert_keras3_to_tfjs(model_path, output_dir)
    else:
        print(f"❌ Model not found: {model_path}")
