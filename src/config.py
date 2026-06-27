"""
Arya Agent - AI Tutor Configuration
"""
import os
from dotenv import load_dotenv

load_dotenv()

BFF_BASE_URL = os.getenv("BFF_BASE_URL", "http://192.168.1.100:8080")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:27b")
DEFAULT_CHILD_ID = os.getenv("DEFAULT_CHILD_ID", "")
AGENT_VERSION = os.getenv("AGENT_VERSION", "0.1.0")
