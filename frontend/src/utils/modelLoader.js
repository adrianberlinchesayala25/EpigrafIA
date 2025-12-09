import * as tf from '@tensorflow/tfjs';

/**
 * 🧠 Model Loader for EpigrafIA
 * Loads TensorFlow.js models for language and accent detection
 */

let languageModel = null;
let accentModel = null;
let modelsLoaded = false;

const MODEL_PATHS = {
  language: '/models/language/model.json',
  accent: '/models/accent/model.json'
};

/**
 * Load both models (language and accent)
 * @returns {Promise<{languageModel: tf.LayersModel, accentModel: tf.LayersModel}>}
 */
export async function loadModels() {
  if (modelsLoaded) {
    console.log('✅ Models already loaded');
    return { languageModel, accentModel };
  }
  
  try {
    console.log('🔄 Loading TensorFlow.js models...');
    
    // Set TensorFlow.js backend (WebGL preferred)
    await tf.ready();
    console.log(`📊 TensorFlow.js backend: ${tf.getBackend()}`);
    
    // Load language model
    console.log('📥 Loading language detection model...');
    languageModel = await tf.loadLayersModel(MODEL_PATHS.language);
    console.log('✅ Language model loaded successfully');
    console.log(`   Input shape: ${JSON.stringify(languageModel.inputs[0].shape)}`);
    console.log(`   Output shape: ${JSON.stringify(languageModel.outputs[0].shape)}`);
    
    // Load accent model
    console.log('📥 Loading accent detection model...');
    accentModel = await tf.loadLayersModel(MODEL_PATHS.accent);
    console.log('✅ Accent model loaded successfully');
    console.log(`   Input shape: ${JSON.stringify(accentModel.inputs[0].shape)}`);
    console.log(`   Output shape: ${JSON.stringify(accentModel.outputs[0].shape)}`);
    
    modelsLoaded = true;
    
    // Log memory info
    const memInfo = tf.memory();
    console.log(`💾 TensorFlow.js memory: ${memInfo.numTensors} tensors, ${(memInfo.numBytes / 1024 / 1024).toFixed(2)} MB`);
    
    return { languageModel, accentModel };
    
  } catch (error) {
    console.error('❌ Error loading models:', error);
    
    if (error.message.includes('404')) {
      throw new Error(
        'No se encontraron los modelos. Asegúrate de haber entrenado y convertido los modelos a TensorFlow.js. ' +
        'Los archivos deben estar en /public/models/language/ y /public/models/accent/'
      );
    }
    
    throw new Error(`Error cargando modelos: ${error.message}`);
  }
}

/**
 * Get loaded models (singleton pattern)
 * @returns {{languageModel: tf.LayersModel|null, accentModel: tf.LayersModel|null, modelsLoaded: boolean}}
 */
export function getModels() {
  return { languageModel, accentModel, modelsLoaded };
}

/**
 * Unload models and free memory
 */
export function unloadModels() {
  if (languageModel) {
    languageModel.dispose();
    languageModel = null;
  }
  if (accentModel) {
    accentModel.dispose();
    accentModel = null;
  }
  modelsLoaded = false;
  console.log('🗑️ Models unloaded and memory freed');
}
