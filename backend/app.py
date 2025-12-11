# Re-export app from main for Render compatibility
from backend.main import app

__all__ = ['app']
