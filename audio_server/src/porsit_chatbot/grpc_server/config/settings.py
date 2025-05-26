# src/porsit_chatbot/grpc_server/config/settings.py
import os
import logging

# --- Load Configuration from Environment Variables ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- STT/TTS Configuration ---
STT_PROVIDER_MODEL_CONFIG = ("openai/whisper-1")
TTS_PROVIDER_MODEL_CONFIG = ("openai/tts-1")

# --- OpenAI Specific Keys ---
OPENAI_STT_API_KEY = os.getenv("OPENAI_STT_API_KEY", OPENAI_API_KEY)
OPENAI_TTS_API_KEY = os.getenv("OPENAI_TTS_API_KEY", OPENAI_API_KEY)
OPENAI_TTS_DEFAULT_VOICE = ("alloy")
OPENAI_TTS_DEFAULT_MODEL = ("tts-1")

# --- Speechmatics Specific ---
SPEECHMATICS_API_KEY = os.getenv("SPEECHMATICS_API_KEY")
SPEECHMATICS_BATCH_API_URL = os.getenv("SPEECHMATICS_BATCH_API_URL", "https://asr.api.speechmatics.com/v2")

# --- Network Configuration ---
SPEECH_GRPC_PORT = os.getenv("SPEECH_SERVICE_GRPC_PORT", "50052")
SPEECH_GRPC_LISTEN_ADDRESS = f"[::]:{SPEECH_GRPC_PORT}"

# --- Logging Configuration ---
LOGGING_LEVEL_NAME = os.getenv("LOGGING_LEVEL", "INFO").upper()
LOGGING_LEVEL = getattr(logging, LOGGING_LEVEL_NAME, logging.INFO)
LOGGING_FORMAT = "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
LOGGING_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

def setup_logging():
    """Configures root logger."""
    logging.basicConfig(
        level=LOGGING_LEVEL,
        format=LOGGING_FORMAT,
        datefmt=LOGGING_DATE_FORMAT
    )
    # Quieten noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

# --- Basic Key Validation ---
def validate_keys():
    """Logs warnings if essential API keys seem unset or default."""
    # Define your actual default/placeholder keys used in getenv calls above
    default_openai_key = "YOUR_DEFAULT_OR_PLACEHOLDER_OPENAI_KEY"
    default_gemini_key = "YOUR_DEFAULT_OR_PLACEHOLDER_GEMINI_KEY"

    if not OPENAI_API_KEY or OPENAI_API_KEY == default_openai_key:
        logging.warning("OPENAI_API_KEY environment variable not set or using default placeholder.")
    if not GEMINI_API_KEY or GEMINI_API_KEY == default_gemini_key:
        logging.warning("GEMINI_API_KEY environment variable not set or using default placeholder.")