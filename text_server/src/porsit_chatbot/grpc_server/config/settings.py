# src/porsit_chatbot/grpc_server/config/settings.py
import os
import json
import logging
from pathlib import Path

# --- Base Directory for Prompts ---
PROMPTS_DIR = Path(__file__).parent / "prompts"

def load_prompt_from_file(relative_path: str) -> str:
    """
    Loads a prompt string from a file relative to the PROMPTS_DIR.
    """
    filepath = PROMPTS_DIR / relative_path
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read().strip() # .strip() to remove leading/trailing whitespace
    except FileNotFoundError:
        logging.error(f"CRITICAL: Prompt file not found: {filepath}.")
        return "NO SYSTEM ROLE FOUND"
    except Exception as e:
        logging.error(f"CRITICAL: Error loading prompt file {filepath}: {e}", exc_info=True)
        return "NO SYSTEM ROLE FOUND"

# --- Load Configuration from Environment Variables ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
XAI_API_KEY = os.getenv("XAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# --- API Provider Configuration ---
API_PROVIDER_CONFIG = {
    "openai": {
        "api_key": OPENAI_API_KEY,
        "base_url": None,
        "langchain_class": "ChatOpenAI"
    },
    "gemini": {
        "api_key": GEMINI_API_KEY,
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "langchain_class": "ChatOpenAI"
    },
    "mistral": {
        "api_key": MISTRAL_API_KEY,
        "base_url": None,
        "langchain_class": "ChatMistralAI"
    },
    "xai": {
        "api_key": XAI_API_KEY,
        "base_url": "https://api.x.ai/v1",
        "langchain_class": "ChatOpenAI"
    },
    "deepseek": {
        "api_key": DEEPSEEK_API_KEY,
        "base_url": "https://api.deepseek.com",
        "langchain_class": "ChatOpenAI"
    },
    "anthropic": { 
        "api_key": ANTHROPIC_API_KEY,
        "base_url": None,
        "langchain_class": "ChatAnthropic"
    }
}

# --- Model Paths ---
INTENT_MODEL_PATH = os.getenv("INTENT_MODEL_PATH")
ENTITY_MODEL_PATH = os.getenv("ENTITY_MODEL_PATH")
ID2TAG_PATH = os.getenv("ID2TAG_PATH")
TOKENIZER_NAME = "bert-base-multilingual-cased"

# --- Labels (Used for validation, prompting, encoding) ---
INTENT_LABELS_ENGLISH = [
    "ask_shipping_duration", "cancel_order", "complaint_followup", "cost_estimation", "customer_support_hours",
    "define_jargon", "escalate", "feedback", "get_branch_info", "get_company_overview", "get_corporate_and_business_solutions",
    "get_international_services_info", "get_services_info", "off_topic", "other/FAQ", "recommend_company", "create_support_ticket_off_hours",
    "shipping_restrictions", "small_talk", "submit_complaint", "submit_order", "tracking_shipment", "ask_operating_regions",
    "Unclear"
]
ENTITY_LABELS_ENGLISH = [
    "address", "barcode_number", "branch_location", "branch_name", "company_name", "complaint_id",
    "complaint_type", "destination_location", "estimated_value", "item_name", "jargon", "last_of_contract_code",
    "package_size", "package_weight", "phone_number", "pickup_location", "service_type", "tracking_number",
    "O"
]

# --- Operational Hours Configuration (Client-Specific) ---
# Structure: client_name: { "start": hour, "end": hour, "timezone": "TZ", "days": [0,1,2,3,4] }
# If a client is not in this dict, they are considered 24/7 or time-agnostic for persona/NLU.
CLIENT_SPECIFIC_WORKING_HOURS = {
    "tipax": {
        "start_hour": 7,
        "end_hour": 21,
        "timezone": "Asia/Tehran",
        "days": [0, 1, 2, 3, 5, 6]
    },
    "zboom": {
        "start_hour": 8.5,
        "end_hour": 17,
        "timezone": "Asia/Tehran",
        "days": [0, 1, 2, 5, 6]
    }
}

# --- LangChain System Roles (Moved from server.py) ---
SYSTEM_ROLES = {
    "persona": {
        "porsit": load_prompt_from_file("personas/porsit.txt"),
        "tipax_working_hours": load_prompt_from_file("personas/tipax_working_hours.txt"),
        "tipax_off_hours": load_prompt_from_file("personas/tipax_off_hours.txt"),
        "sandbox": load_prompt_from_file("personas/sandbox.txt"),
        "tinext": load_prompt_from_file("personas/tinext.txt"),
        "tiexpress": load_prompt_from_file("personas/tiexpress.txt"),
        "jaabaar": load_prompt_from_file("personas/jaabaar.txt"),
        "nona": load_prompt_from_file("personas/nona.txt"),
        "fakher": load_prompt_from_file("personas/fakher.txt"),
        "zboom_working_hours": load_prompt_from_file("personas/zboom_working_hours.txt"),
        "zboom_off_hours": load_prompt_from_file("personas/zboom_off_hours.txt")
    },
    "nlu_instructions": {
        "porsit": load_prompt_from_file("nlu/porsit.txt"),
        "tipax_working_hours": load_prompt_from_file("nlu/tipax_working_hours.txt"),
        "tipax_off_hours": load_prompt_from_file("nlu/tipax_off_hours.txt"),
        "tinext": load_prompt_from_file("nlu/tinext.txt"),
        "tiexpress": load_prompt_from_file("nlu/tiexpress.txt"),
        "jaabaar": load_prompt_from_file("nlu/jaabaar.txt"),
        "nona": load_prompt_from_file("nlu/nona.txt"),
        "fakher": load_prompt_from_file("nlu/fakher.txt"),
        "zboom_working_hours": load_prompt_from_file("nlu/zboom_working_hours.txt"),
        "zboom_off_hours": load_prompt_from_file("nlu/zboom_off_hours.txt")
    },
    "cost_estimation_instructions": {
        "porsit": load_prompt_from_file("cost_estimation/porsit.txt"),
        "tipax": load_prompt_from_file("cost_estimation/tipax.txt"),
        "tinext": load_prompt_from_file("cost_estimation/tinext.txt"),
        "tiexpress": load_prompt_from_file("cost_estimation/tiexpress.txt"),
        "jaabaar": load_prompt_from_file("cost_estimation/jaabaar.txt"),
        "zboom": load_prompt_from_file("cost_estimation/zboom.txt")
    },
    "summary_instructions": {
        "porsit": load_prompt_from_file("summary/porsit.txt"),
        "tipax": load_prompt_from_file("summary/tipax.txt"),
        "tinext": load_prompt_from_file("summary/tinext.txt"),
        "tiexpress": load_prompt_from_file("summary/tiexpress.txt"),
        "jaabaar": load_prompt_from_file("summary/jaabaar.txt"),
        "zboom": load_prompt_from_file("summary/zboom.txt")
    }
}

# --- Network Configuration ---
GRPC_PORT = os.getenv("GRPC_PORT", "50051")
GRPC_LISTEN_ADDRESS = f"[::]:{GRPC_PORT}"

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

def load_id2tag(path: str) -> dict:
    """Loads the id2tag mapping from a JSON file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            id2tag_data = json.load(f)
            # Ensure keys are strings as expected by the prediction code
            return {str(k): v for k, v in id2tag_data.items()}
    except FileNotFoundError:
        logging.error(f"CRITICAL: id2tag file not found at {path}. Entity prediction will likely fail.")
        return {}
    except json.JSONDecodeError:
        logging.error(f"CRITICAL: Could not decode JSON from {path}. Entity prediction will likely fail.")
        return {}
    except Exception as e:
        logging.error(f"CRITICAL: Unexpected error loading id2tag from {path}: {e}", exc_info=True)
        return {}

# Load id2tag globally within settings
ID2TAG_MAP = load_id2tag(ID2TAG_PATH)

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