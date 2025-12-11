"""
🏥 EpigrafIA - Health Check Endpoint
=====================================
Endpoint: /api/health
Verificación del estado del servicio
"""

import json
from http.server import BaseHTTPRequestHandler


class handler(BaseHTTPRequestHandler):
    """Health check handler"""
    
    def do_GET(self):
        """Return health status"""
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        response = {
            "status": "healthy",
            "models_loaded": True,
            "version": "1.0.0",
            "runtime": "vercel-python-3.11"
        }
        
        self.wfile.write(json.dumps(response).encode())
    
    def do_OPTIONS(self):
        """Handle CORS preflight"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
