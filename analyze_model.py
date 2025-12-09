"""
Analyze model weights and biases to detect issues
"""
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('outputs/models_trained/language_model_best.keras')

# Get output layer weights
output_layer = model.get_layer('language_output')
weights, biases = output_layer.get_weights()

print('='*60)
print('ANÁLISIS DE LA CAPA DE SALIDA')
print('='*60)
print(f'Pesos shape: {weights.shape}')  # (128, 4)
print(f'Biases shape: {biases.shape}')  # (4,)

labels = ['Español', 'Inglés', 'Francés', 'Alemán']

print('\n📊 BIASES (sesgo inicial por clase):')
for i, label in enumerate(labels):
    print(f'   {label}: {biases[i]:.4f}')

print('\n📊 SUMA DE PESOS ABSOLUTOS por clase:')
for i, label in enumerate(labels):
    weight_sum = np.abs(weights[:, i]).sum()
    print(f'   {label}: {weight_sum:.2f}')

print('\n📊 MEDIA DE PESOS por clase:')
for i, label in enumerate(labels):
    weight_mean = weights[:, i].mean()
    print(f'   {label}: {weight_mean:.4f}')

print('\n📊 DESV. ESTÁNDAR DE PESOS por clase:')
for i, label in enumerate(labels):
    weight_std = weights[:, i].std()
    print(f'   {label}: {weight_std:.4f}')

# Check if German weights are significantly lower
print('\n' + '='*60)
print('DIAGNÓSTICO')
print('='*60)

german_bias = biases[3]
german_weight_sum = np.abs(weights[:, 3]).sum()
avg_weight_sum = np.mean([np.abs(weights[:, i]).sum() for i in range(4)])

if german_bias < -1:
    print(f'⚠️ Alemán tiene bias MUY NEGATIVO ({german_bias:.4f})')
    print('   Esto significa que el modelo está predispuesto CONTRA alemán')

if german_weight_sum < avg_weight_sum * 0.5:
    print(f'⚠️ Alemán tiene pesos MUY BAJOS ({german_weight_sum:.2f} vs promedio {avg_weight_sum:.2f})')
