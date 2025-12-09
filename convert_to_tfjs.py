"""
Script to convert Keras models to TensorFlow.js format
Manual conversion to avoid tensorflowjs package compatibility issues
"""
import os
import json
import struct
import numpy as np

def serialize_weights(weights_list):
    """Serialize a list of weight arrays to bytes"""
    buffer = bytearray()
    for w in weights_list:
        # Ensure float32
        w_f32 = w.astype(np.float32)
        # Append raw bytes
        buffer.extend(w_f32.tobytes())
    return bytes(buffer)


def get_dtype_size(dtype):
    """Get size in bytes for a dtype"""
    sizes = {'float32': 4, 'float16': 2, 'int32': 4, 'int8': 1, 'uint8': 1}
    return sizes.get(dtype, 4)


def convert_keras_to_tfjs(model_path: str, output_dir: str):
    """Convert a Keras model to TensorFlow.js layers format"""
    
    import keras
    
    print(f"🔄 Loading model from {model_path}...")
    model = keras.models.load_model(model_path)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Build layers model topology
    print("📦 Building model topology...")
    
    # Get the model config in Keras 3 format
    model_config = model.get_config()
    
    # Build TensorFlow.js layers-model format
    model_topology = {
        "class_name": "Sequential" if model.__class__.__name__ == "Sequential" else "Functional",
        "config": {
            "name": model.name,
            "trainable": True,
            "layers": []
        }
    }
    
    # For Functional models
    if hasattr(model_config, 'get') and 'layers' in model_config:
        for layer_config in model_config['layers']:
            model_topology["config"]["layers"].append({
                "class_name": layer_config['class_name'],
                "config": layer_config['config'],
                "name": layer_config['config'].get('name', ''),
                "inbound_nodes": layer_config.get('inbound_nodes', [])
            })
    
    # Build weights manifest
    print("⚖️ Extracting weights...")
    weights_data = []
    weights_manifest = []
    
    for layer in model.layers:
        layer_weights = layer.get_weights()
        weight_names = layer.weights  # Gets actual weight names
        
        for i, w in enumerate(layer_weights):
            if i < len(weight_names):
                weight_name = weight_names[i].name
            else:
                weight_name = f"{layer.name}/weight_{i}"
            
            # Store weight info
            weights_manifest.append({
                "name": weight_name,
                "shape": list(w.shape),
                "dtype": "float32"
            })
            weights_data.append(w.astype(np.float32))
    
    # Serialize weights
    print("💾 Serializing weights...")
    weights_bytes = serialize_weights(weights_data)
    
    # Calculate number of shards (split if > 4MB)
    shard_size = 4 * 1024 * 1024  # 4MB per shard
    num_shards = max(1, (len(weights_bytes) + shard_size - 1) // shard_size)
    
    # For simplicity, use single shard if small enough
    if len(weights_bytes) < shard_size:
        shard_paths = ["group1-shard1of1.bin"]
        shards = [weights_bytes]
    else:
        shard_paths = [f"group1-shard{i+1}of{num_shards}.bin" for i in range(num_shards)]
        shards = []
        for i in range(num_shards):
            start = i * shard_size
            end = min((i + 1) * shard_size, len(weights_bytes))
            shards.append(weights_bytes[start:end])
    
    # Build model.json
    model_json = {
        "format": "layers-model",
        "generatedBy": f"keras {keras.__version__}",
        "convertedBy": "EpigrafIA manual converter v1.0",
        "modelTopology": {
            "keras_version": keras.__version__,
            "backend": "tensorflow",
            "model_config": {
                "class_name": model.__class__.__name__,
                "config": model_config
            }
        },
        "weightsManifest": [{
            "paths": shard_paths,
            "weights": weights_manifest
        }]
    }
    
    # Save model.json
    model_json_path = os.path.join(output_dir, "model.json")
    with open(model_json_path, 'w', encoding='utf-8') as f:
        json.dump(model_json, f, indent=2)
    print(f"✅ Model config saved: {model_json_path}")
    
    # Save weight shards
    for shard_path, shard_data in zip(shard_paths, shards):
        shard_full_path = os.path.join(output_dir, shard_path)
        with open(shard_full_path, 'wb') as f:
            f.write(shard_data)
        print(f"✅ Weights saved: {shard_full_path} ({len(shard_data) / 1024:.1f} KB)")
    
    # Create a simple labels file
    labels = ["Español", "Inglés", "Francés", "Alemán"]
    labels_path = os.path.join(output_dir, "labels.json")
    with open(labels_path, 'w', encoding='utf-8') as f:
        json.dump(labels, f, indent=2)
    print(f"✅ Labels saved: {labels_path}")
    
    print(f"\n🎉 Conversion complete!")
    print(f"   Total weights: {sum(np.prod(w['shape']) for w in weights_manifest):,} parameters")
    print(f"   Weight files: {len(shard_paths)} shard(s)")
    print(f"   Output: {output_dir}/")
    
    return True


if __name__ == "__main__":
    # Convert language model
    model_path = "outputs/models_trained/language_model_best.keras"
    output_dir = "frontend/public/models/language"
    
    if os.path.exists(model_path):
        convert_keras_to_tfjs(model_path, output_dir)
    else:
        print(f"❌ Model not found: {model_path}")
