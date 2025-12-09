/**
 * Audio Recording and Processing Module
 * Handles MediaRecorder API for capturing audio
 */

let mediaRecorder = null;
let audioChunks = [];
let stream = null;

/**
 * Start recording audio from microphone
 * @param {number} duration - Recording duration in seconds
 * @returns {Promise<AudioBuffer>}
 */
export async function startRecording(duration = 3) {
    try {
        // Request microphone access
        stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                sampleRate: 16000,
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: true
            }
        });

        audioChunks = [];

        // Create MediaRecorder
        const mimeType = MediaRecorder.isTypeSupported('audio/webm')
            ? 'audio/webm'
            : 'audio/ogg';

        mediaRecorder = new MediaRecorder(stream, { mimeType });

        // Collect audio data
        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };

        // Start recording
        mediaRecorder.start();

        // Stop after duration
        return new Promise((resolve, reject) => {
            setTimeout(async () => {
                try {
                    const audioBuffer = await stopRecording();
                    resolve(audioBuffer);
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
 * Stop recording and return audio buffer
 * @returns {Promise<AudioBuffer>}
 */
export async function stopRecording() {
    if (!mediaRecorder) {
        throw new Error('No hay grabación activa');
    }

    return new Promise((resolve, reject) => {
        mediaRecorder.onstop = async () => {
            try {
                // Create blob from chunks
                const audioBlob = new Blob(audioChunks, { type: mediaRecorder.mimeType });

                // Convert to AudioBuffer
                const arrayBuffer = await audioBlob.arrayBuffer();
                const audioContext = new (window.AudioContext || window.webkitAudioContext)();
                const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

                // Cleanup
                stream.getTracks().forEach(track => track.stop());
                stream = null;
                mediaRecorder = null;

                resolve(audioBuffer);
            } catch (error) {
                reject(error);
            }
        };

        mediaRecorder.stop();
    });
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
    formData.append('audio', audioBlob, 'recording.webm');

    try {
        const response = await fetch('/api/analyze', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        return await response.json();
    } catch (error) {
        console.error('Error sending audio:', error);
        throw new Error('Error al enviar audio al servidor');
    }
}
