import * as tf from '@tensorflow/tfjs';

/**
 * 🧠 Model Loader for EpigrafIA
 * Loads TensorFlow.js models for language and accent detection
 */

let languageModel = null;
let accentModel = null;
let modelsLoaded = false;
let languageLabels = null;

const MODEL_PATHS = {
  language: '/models/language/model.json',
  accent: '/models/accent/model.json',
  languageLabels: '/models/language/labels.json'
};

/**
 * Load language labels
 */
async function loadLabels() {
  try {
    const response = await fetch(MODEL_PATHS.languageLabels);
    if (response.ok) {
      languageLabels = await response.json();
      console.log('✅ Language labels loaded:', languageLabels);
    }
  } catch (error) {
    console.warn('⚠️ Could not load language labels, using defaults');
    languageLabels = ["Español", "Inglés", "Francés", "Alemán"];
  }
}

/**
 * Load both models (language and accent)
 * @returns {Promise<{languageModel: tf.LayersModel, accentModel: tf.LayersModel}>}
 */
export async function loadModels() {
  if (modelsLoaded) {
    console.log('✅ Models already loaded');
    return { languageModel, accentModel, languageLabels };
  }
  
  try {
    console.log('🔄 Loading TensorFlow.js models...');
    
    // Set TensorFlow.js backend (WebGL preferred)
    await tf.ready();
    console.log(`📊 TensorFlow.js backend: ${tf.getBackend()}`);
    
    // Load language labels
    await loadLabels();
    
    // Load language model
    console.log('📥 Loading language detection model...');
    console.log('   Fetching from:', MODEL_PATHS.language);
    
    try {
      languageModel = await tf.loadLayersModel(MODEL_PATHS.language);
      console.log('✅ Language model loaded successfully');
      console.log(`   Input shape: ${JSON.stringify(languageModel.inputs[0].shape)}`);
      console.log(`   Output shape: ${JSON.stringify(languageModel.outputs[0].shape)}`);
    } catch (langError) {
      console.error('❌ Language model load error:', langError);
      throw new Error(`Error cargando modelo de idioma: ${langError.message}`);
    }
    // Try to load accent model (optional - may not exist yet)
    try {
      console.log('📥 Loading accent detection model...');
      accentModel = await tf.loadLayersModel(MODEL_PATHS.accent);
      console.log('✅ Accent model loaded successfully');
      console.log(`   Input shape: ${JSON.stringify(accentModel.inputs[0].shape)}`);
      console.log(`   Output shape: ${JSON.stringify(accentModel.outputs[0].shape)}`);
    } catch (accentError) {
      console.warn('⚠️ Accent model not available yet (will be added in future update)');
      accentModel = null;
    }
    
    modelsLoaded = true;
    
    // Log memory info
    const memInfo = tf.memory();
    console.log(`💾 TensorFlow.js memory: ${memInfo.numTensors} tensors, ${(memInfo.numBytes / 1024 / 1024).toFixed(2)} MB`);
    
    return { languageModel, accentModel, languageLabels };
    
  } catch (error) {
    console.error('❌ Error loading models:', error);
    
    if (error.message.includes('404')) {
      throw new Error(
        'No se encontraron los modelos. Asegúrate de haber entrenado y convertido los modelos a TensorFlow.js. ' +
        'Los archivos deben estar en /public/models/language/'
      );
    }
    
    throw new Error(`Error cargando modelos: ${error.message}`);
  }
}

/**
 * Get loaded models (singleton pattern)
 * @returns {{languageModel: tf.LayersModel|null, accentModel: tf.LayersModel|null, languageLabels: string[]|null, modelsLoaded: boolean}}
 */
export function getModels() {
  return { languageModel, accentModel, languageLabels, modelsLoaded };
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
  languageLabels = null;
  console.log('🗑️ Models unloaded and memory freed');
}
