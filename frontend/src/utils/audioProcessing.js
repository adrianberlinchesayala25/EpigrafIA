import * as tf from '@tensorflow/tfjs';

/**
 * 🎵 Audio Processing for EpigrafIA
 * Handles audio recording, file loading, and MFCC feature extraction
 */

// Configuration matching training parameters
const CONFIG = {
    sampleRate: 16000,
    duration: 3, // seconds
    nMfcc: 40,
    nMels: 128,
    fftSize: 2048,
    hopLength: 512,
    nFeatures: 120 // 40 MFCC + 40 delta + 40 delta²
};

/**
 * Record audio from microphone
 * @param {number} durationSeconds - Recording duration in seconds
 * @returns {Promise<AudioBuffer>}
 */
export async function recordAudio(durationSeconds = CONFIG.duration) {
    try {
        console.log('🎙️ Requesting microphone access...');
        const stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                channelCount: 1,
                sampleRate: CONFIG.sampleRate,
                echoCancellation: true,
                noiseSuppression: true
            }
        });

        const mediaRecorder = new MediaRecorder(stream, {
            mimeType: 'audio/webm;codecs=opus'
        });

        const chunks = [];

        return new Promise((resolve, reject) => {
            mediaRecorder.ondataavailable = (e) => {
                if (e.data.size > 0) {
                    chunks.push(e.data);
                }
            };

            mediaRecorder.onstop = async () => {
                try {
                    // Stop all tracks
                    stream.getTracks().forEach(track => track.stop());

                    // Convert to audio buffer
                    const blob = new Blob(chunks, { type: 'audio/webm' });
                    const arrayBuffer = await blob.arrayBuffer();

                    const audioContext = new (window.AudioContext || window.webkitAudioContext)({
                        sampleRate: CONFIG.sampleRate
                    });

                    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

                    console.log(`✅ Audio recorded: ${audioBuffer.duration.toFixed(2)}s, ${audioBuffer.sampleRate}Hz`);
                    resolve(audioBuffer);

                } catch (error) {
                    reject(new Error(`Error decodificando audio: ${error.message}`));
                }
            };

            mediaRecorder.onerror = (error) => {
                stream.getTracks().forEach(track => track.stop());
                reject(new Error(`Error en MediaRecorder: ${error}`));
            };

            // Start recording
            mediaRecorder.start();
            console.log(`⏺️ Recording for ${durationSeconds} seconds...`);

            // Stop after duration
            setTimeout(() => {
                if (mediaRecorder.state === 'recording') {
                    mediaRecorder.stop();
                }
            }, durationSeconds * 1000);
        });

    } catch (error) {
        if (error.name === 'NotAllowedError') {
            throw new Error('Permiso de micrófono denegado. Por favor, permite el acceso al micrófono.');
        } else if (error.name === 'NotFoundError') {
            throw new Error('No se encontró ningún micrófono. Por favor, conecta un micrófono.');
        }
        throw new Error(`Error accediendo al micrófono: ${error.message}`);
    }
}

/**
 * Load audio from file
 * @param {File} file - Audio file
 * @returns {Promise<AudioBuffer>}
 */
export async function loadAudioFile(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();

        reader.onload = async (e) => {
            try {
                const audioContext = new (window.AudioContext || window.webkitAudioContext)({
                    sampleRate: CONFIG.sampleRate
                });

                const arrayBuffer = e.target.result;
                const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

                console.log(`✅ Audio file loaded: ${audioBuffer.duration.toFixed(2)}s, ${audioBuffer.sampleRate}Hz`);
                resolve(audioBuffer);

            } catch (error) {
                reject(new Error(`Error decodificando archivo de audio: ${error.message}`));
            }
        };

        reader.onerror = () => reject(new Error('Error leyendo archivo'));
        reader.readAsArrayBuffer(file);
    });
}

/**
 * Convert AudioBuffer to normalized float32 array
 * @param {AudioBuffer} audioBuffer
 * @returns {Float32Array}
 */
export function audioBufferToFloat32(audioBuffer) {
    // Get mono channel
    let channelData = audioBuffer.getChannelData(0);

    // Resample if needed (simplified - just truncate or pad)
    const targetLength = CONFIG.sampleRate * CONFIG.duration;
    let processedData;

    if (channelData.length > targetLength) {
        // Truncate to target length
        processedData = channelData.slice(0, targetLength);
    } else if (channelData.length < targetLength) {
        // Pad with zeros
        processedData = new Float32Array(targetLength);
        processedData.set(channelData);
    } else {
        processedData = channelData;
    }

    return processedData;
}

/**
 * Extract MFCC features from audio buffer
 * NOTE: This is a SIMPLIFIED version for browser. In production, use a library like meyda.js
 * or pre-compute features server-side for better accuracy.
 * 
 * @param {AudioBuffer} audioBuffer
 * @returns {Promise<tf.Tensor>} Shape: [1, time_steps, features]
 */
export async function extractMFCC(audioBuffer) {
    return tf.tidy(() => {
        console.log('🔄 Extracting MFCC features...');

        // Convert to Float32Array
        const audioData = audioBufferToFloat32(audioBuffer);

        // For now, we'll create a simplified spectrogram-like representation
        // In production, this should use proper MFCC extraction
        const numFrames = Math.floor((audioData.length - CONFIG.fftSize) / CONFIG.hopLength) + 1;
        const features = [];

        for (let i = 0; i < numFrames; i++) {
            const frameStart = i * CONFIG.hopLength;
            const frameEnd = Math.min(frameStart + CONFIG.fftSize, audioData.length);
            const frame = audioData.slice(frameStart, frameEnd);

            // Simplified feature extraction (placeholder)
            // In production: use FFT, mel filterbank, DCT for real MFCC
            const frameFeatures = new Float32Array(CONFIG.nFeatures);

            // Compute simple spectral features
            for (let j = 0; j < CONFIG.nMfcc; j++) {
                let sum = 0;
                const binSize = Math.floor(frame.length / CONFIG.nMfcc);
                const start = j * binSize;
                const end = Math.min(start + binSize, frame.length);

                for (let k = start; k < end; k++) {
                    sum += Math.abs(frame[k]);
                }

                frameFeatures[j] = sum / binSize;
                frameFeatures[j + CONFIG.nMfcc] = j > 0 ? frameFeatures[j] - frameFeatures[j - 1] : 0; // delta
                frameFeatures[j + 2 * CONFIG.nMfcc] = j > 0 ? frameFeatures[j + CONFIG.nMfcc] - frameFeatures[j - 1 + CONFIG.nMfcc] : 0; // delta²
            }

            features.push(Array.from(frameFeatures));
        }

        // Model expects exactly 94 frames (matches Python training with 3s @ 16kHz, hop=512)
        const targetFrames = 94;
        while (features.length < targetFrames) {
            features.push(new Array(CONFIG.nFeatures).fill(0));
        }
        if (features.length > targetFrames) {
            features.length = targetFrames;
        }

        // Create tensor [1, time_steps, features]
        const tensor = tf.tensor3d([features]);

        // Normalize
        const mean = tensor.mean([1], true);
        const std = tf.moments(tensor, [1], true).variance.sqrt();
        const normalized = tensor.sub(mean).div(std.add(1e-8));

        console.log(`✅ MFCC extracted: shape ${normalized.shape}`);
        return normalized;
    });
}

/**
 * Draw waveform on canvas
 * @param {AudioBuffer} audioBuffer
 * @param {HTMLCanvasElement} canvas
 */
export function drawWaveform(audioBuffer, canvas) {
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    const data = audioBuffer.getChannelData(0);

    // Clear canvas
    ctx.fillStyle = '#0a0a0a';
    ctx.fillRect(0, 0, width, height);

    // Calculate step size
    const step = Math.ceil(data.length / width);
    const amp = height / 2;

    // Draw waveform
    ctx.strokeStyle = '#8b5cf6'; // Purple
    ctx.lineWidth = 2;
    ctx.beginPath();

    for (let i = 0; i < width; i++) {
        const min = Math.min(...data.slice(i * step, (i + 1) * step));
        const max = Math.max(...data.slice(i * step, (i + 1) * step));

        ctx.moveTo(i, (1 + min) * amp);
        ctx.lineTo(i, (1 + max) * amp);
    }

    ctx.stroke();
}

export { CONFIG };
