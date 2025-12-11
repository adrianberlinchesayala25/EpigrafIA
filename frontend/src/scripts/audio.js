/**
 * Audio Recording and Processing Module
 * Records audio directly as WAV at 16kHz for accurate ML processing
 */

let audioContext = null;
let mediaStream = null;
let scriptProcessor = null;
let audioData = [];
let isRecording = false;

const SAMPLE_RATE = 16000;

/**
 * Start recording audio from microphone
 * Records directly as PCM data at 16kHz
 * @param {number} duration - Recording duration in seconds
 * @returns {Promise<Blob>} - WAV blob
 */
export async function startRecording(duration = 3) {
    try {
        // Request microphone access
        mediaStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                sampleRate: SAMPLE_RATE,
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: true
            }
        });

        // Create audio context at 16kHz
        audioContext = new (window.AudioContext || window.webkitAudioContext)({
            sampleRate: SAMPLE_RATE
        });

        const source = audioContext.createMediaStreamSource(mediaStream);
        
        // Use ScriptProcessor to capture raw PCM data
        // Buffer size of 4096 is a good balance
        scriptProcessor = audioContext.createScriptProcessor(4096, 1, 1);
        
        audioData = [];
        isRecording = true;

        scriptProcessor.onaudioprocess = (e) => {
            if (isRecording) {
                const channelData = e.inputBuffer.getChannelData(0);
                // Clone the data since the buffer is reused
                audioData.push(new Float32Array(channelData));
            }
        };

        source.connect(scriptProcessor);
        scriptProcessor.connect(audioContext.destination);

        // Record for specified duration
        return new Promise((resolve, reject) => {
            setTimeout(async () => {
                try {
                    const wavBlob = await stopRecording();
                    resolve(wavBlob);
                } catch (error) {
                    reject(error);
                }
            }, duration * 1000);
        });

    } catch (error) {
        console.error('Error accessing microphone:', error);
        throw new Error('No se pudo acceder al micrófono. Verifica los permisos.');
    }
}

/**
 * Stop recording and return WAV blob
 * @returns {Promise<Blob>} - WAV audio blob
 */
export async function stopRecording() {
    isRecording = false;

    // Disconnect and cleanup
    if (scriptProcessor) {
        scriptProcessor.disconnect();
        scriptProcessor = null;
    }
    
    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
        mediaStream = null;
    }

    // Merge all audio chunks
    const totalLength = audioData.reduce((acc, chunk) => acc + chunk.length, 0);
    const mergedData = new Float32Array(totalLength);
    let offset = 0;
    for (const chunk of audioData) {
        mergedData.set(chunk, offset);
        offset += chunk.length;
    }
    audioData = [];

    // Get actual sample rate used
    const actualSampleRate = audioContext?.sampleRate || SAMPLE_RATE;
    
    if (audioContext) {
        await audioContext.close();
        audioContext = null;
    }

    // Convert to WAV
    const wavBlob = float32ToWav(mergedData, actualSampleRate);
    return wavBlob;
}

/**
 * Convert Float32Array PCM data to WAV blob
 * @param {Float32Array} samples - PCM audio samples
 * @param {number} sampleRate - Sample rate
 * @returns {Blob} - WAV file blob
 */
function float32ToWav(samples, sampleRate) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);

    // WAV header
    const writeString = (offset, string) => {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    };

    writeString(0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true); // Subchunk1Size
    view.setUint16(20, 1, true); // AudioFormat (PCM)
    view.setUint16(22, 1, true); // NumChannels (mono)
    view.setUint32(24, sampleRate, true); // SampleRate
    view.setUint32(28, sampleRate * 2, true); // ByteRate
    view.setUint16(32, 2, true); // BlockAlign
    view.setUint16(34, 16, true); // BitsPerSample
    writeString(36, 'data');
    view.setUint32(40, samples.length * 2, true); // Subchunk2Size

    // Convert Float32 to Int16
    let offset = 44;
    for (let i = 0; i < samples.length; i++) {
        const s = Math.max(-1, Math.min(1, samples[i]));
        view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
        offset += 2;
    }

    return new Blob([buffer], { type: 'audio/wav' });
}

/**
 * Load audio file from user upload
 * @param {File} file - Audio file
 * @returns {Promise<AudioBuffer>}
 */
export async function loadAudioFile(file) {
    try {
        const arrayBuffer = await file.arrayBuffer();
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
        return audioBuffer;
    } catch (error) {
        console.error('Error loading audio file:', error);
        throw new Error('No se pudo cargar el archivo de audio');
    }
}

/**
 * Check if browser supports audio recording
 * @returns {boolean}
 */
export function isRecordingSupported() {
    return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

/**
 * Send audio to backend for processing
 * @param {Blob} audioBlob - Audio data
 * @returns {Promise<Object>} - Server response
 */
export async function sendAudioToBackend(audioBlob) {
    const formData = new FormData();
    formData.append('audio', audioBlob, 'recording.wav');

  // 1. Leemos la URL de la variable de entorno (o usamos localhost si no existe)
const API_URL = import.meta.env.PUBLIC_API_URL || 'http://localhost:8000';

try {
    // 2. Usamos esa URL para hacer la petición
    const response = await fetch(`${API_URL}/api/analyze`, {
        method: 'POST',
        body: formData
    });
// ... resto de tu código

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        return await response.json();
    } catch (error) {
        console.error('Error sending audio:', error);
        throw new Error('Error al enviar audio al servidor');
    }
}
