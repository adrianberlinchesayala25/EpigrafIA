"""
🎤 EpigrafIA - Vercel Serverless Function
==========================================
Endpoint: /api/analyze
Endpoint principal de análisis de audio
Compatible con el backend existente FastAPI
"""

import json
import logging
import os
import tempfile
from http.server import BaseHTTPRequestHandler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================
# Configuration
# ============================================
SAMPLE_RATE = 16000
DURATION = 3
LANGUAGE_LABELS = ['Español', 'Inglés', 'Francés', 'Alemán']
ACCENT_LABELS = [
    'España', 'México', 'UK', 'USA',
    'Francia', 'Quebec', 'Alemania', 'Austria'
]

# ============================================
# Vercel Serverless Handler
# ============================================
class handler(BaseHTTPRequestHandler):
    """
    Vercel Python Serverless Function Handler
    Endpoint: /api/analyze
    """
    
    def do_OPTIONS(self):
        """Handle CORS preflight requests"""
        self.send_response(200)
        self._send_cors_headers()
        self.end_headers()
    
    def do_GET(self):
        """Health check / info endpoint"""
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self._send_cors_headers()
        self.end_headers()
        
        response = {
            "status": "healthy",
            "endpoint": "/api/analyze",
            "version": "1.0.0",
            "runtime": "vercel-python-3.11",
            "method": "POST",
            "accepts": "multipart/form-data with audio file"
        }
        self.wfile.write(json.dumps(response).encode())
    
    def do_POST(self):
        """
        Analyze audio file
        Compatible with the existing FastAPI endpoint
        """
        try:
            # Get content info
            content_type = self.headers.get('Content-Type', '')
            content_length = int(self.headers.get('Content-Length', 0))
            
            if content_length == 0:
                self._send_error(400, "No audio data received")
                return
            
            if content_length > 10 * 1024 * 1024:  # 10MB limit
                self._send_error(413, "File too large. Maximum size is 10MB")
                return
            
            # Read body
            body = self.rfile.read(content_length)
            
            # Parse multipart data
            audio_data = None
            filename = "audio.webm"
            
            if 'multipart/form-data' in content_type:
                audio_data, filename = self._parse_multipart(body, content_type)
            else:
                audio_data = body
            
            if not audio_data or len(audio_data) == 0:
                self._send_error(400, "Could not extract audio data")
                return
            
            logger.info(f"📥 Received audio: {filename} ({len(audio_data)} bytes)")
            
            # Get audio info
            audio_info = self._get_audio_info(audio_data, filename)
            
            # Return response compatible with frontend
            # NOTE: Real inference uses TensorFlow.js in the client
            result = {
                "success": True,
                "audio": {
                    "filename": filename,
                    "size_bytes": len(audio_data),
                    "duration": audio_info.get("duration", DURATION),
                    "sample_rate": audio_info.get("sample_rate", SAMPLE_RATE)
                },
                "language": {
                    "detected": "Analizando...",
                    "confidence": 0.0,
                    "probabilities": {label: 0.0 for label in LANGUAGE_LABELS}
                },
                "language_prediction": -1,
                "language_confidence": 0.0,
                "accent": {
                    "detected": "No disponible",
                    "confidence": 0.0,
                    "probabilities": {label: 0.0 for label in ACCENT_LABELS}
                },
                "accent_prediction": -1,
                "accent_confidence": 0.0,
                "inference_mode": "client-side-tfjs",
                "message": "Audio received. Use TensorFlow.js for ML inference."
            }
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
            
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            self._send_error(500, str(e))
    
    def _get_audio_info(self, audio_data: bytes, filename: str) -> dict:
        """Get audio file information"""
        info = {
            "duration": DURATION,
            "sample_rate": SAMPLE_RATE,
            "channels": 1
        }
        
        try:
            import soundfile as sf
            
            ext = '.webm'
            if '.' in filename:
                ext = '.' + filename.rsplit('.', 1)[1].lower()
            
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                tmp.write(audio_data)
                tmp_path = tmp.name
            
            try:
                with sf.SoundFile(tmp_path) as f:
                    info["duration"] = len(f) / f.samplerate
                    info["sample_rate"] = f.samplerate
                    info["channels"] = f.channels
            finally:
                os.unlink(tmp_path)
                
        except Exception as e:
            logger.warning(f"Could not read audio info: {e}")
        
        return info
    
    def _parse_multipart(self, body: bytes, content_type: str) -> tuple:
        """Parse multipart form data to extract audio file"""
        filename = "audio.webm"
        
        try:
            # Extract boundary
            boundary = None
            for part in content_type.split(';'):
                part = part.strip()
                if part.startswith('boundary='):
                    boundary = part[9:].strip('"')
                    break
            
            if not boundary:
                return body, filename
            
            # Split by boundary
            boundary_bytes = f'--{boundary}'.encode()
            parts = body.split(boundary_bytes)
            
            for part in parts:
                part_lower = part.lower()
                if b'audio' in part_lower or b'file' in part_lower or b'name="audio"' in part_lower:
                    # Try to extract filename
                    if b'filename="' in part:
                        start = part.find(b'filename="') + 10
                        end = part.find(b'"', start)
                        if end > start:
                            filename = part[start:end].decode('utf-8', errors='ignore')
                    
                    # Find data after headers
                    header_end = part.find(b'\r\n\r\n')
                    if header_end != -1:
                        data = part[header_end + 4:]
                        if data.endswith(b'--\r\n'):
                            data = data[:-4]
                        elif data.endswith(b'\r\n'):
                            data = data[:-2]
                        return data, filename
            
            return body, filename
            
        except Exception as e:
            logger.error(f"Error parsing multipart: {e}")
            return body, filename
    
    def _send_cors_headers(self):
        """Send CORS headers"""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-Requested-With')
        self.send_header('Access-Control-Max-Age', '86400')
    
    def _send_error(self, code: int, message: str):
        """Send error response"""
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps({
            "success": False,
            "error": message
        }).encode())
    
    def log_message(self, format, *args):
        """Override to use Python logger"""
        logger.info("%s - %s", self.address_string(), format % args)

