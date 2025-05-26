# src/porsit_chatbot/grpc_server/core/model_loader.py
import torch
import threading
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification
from sklearn.preprocessing import LabelEncoder

# Import config values from settings
from porsit_chatbot.grpc_server.config import settings

logger = logging.getLogger(__name__)

class ModelLoader:
    """Loads and holds ML models, tokenizer, and related resources."""
    def __init__(self):
        logger.info("Initializing ModelLoader...")
        self._load_tokenizer()
        self._load_models()
        self._setup_label_encoder()
        self.id2tag = settings.ID2TAG_MAP
        self.tokenizer_thread_lock = threading.Lock()
        logger.info("ModelLoader initialization complete.")

    def _load_tokenizer(self):
        logger.info(f"Loading tokenizer: {settings.TOKENIZER_NAME}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(settings.TOKENIZER_NAME)
            logger.info("Tokenizer loaded successfully.")
        except Exception as e:
            logger.critical(f"Failed to load tokenizer '{settings.TOKENIZER_NAME}': {e}", exc_info=True)
            raise  # Re-raise critical error

    def _load_models(self):
        logger.info(f"Loading intent model from: {settings.INTENT_MODEL_PATH}")
        try:
            # Load models onto CPU explicitly if needed, or let Transformers handle device placement
            self.intent_model = AutoModelForSequenceClassification.from_pretrained(settings.INTENT_MODEL_PATH)
            self.intent_model.eval() # Set to evaluation mode
            logger.info("Intent model loaded successfully.")
        except Exception as e:
            logger.critical(f"Failed to load intent model: {e}", exc_info=True)
            raise

        logger.info(f"Loading entity model from: {settings.ENTITY_MODEL_PATH}")
        try:
            self.entity_model = AutoModelForTokenClassification.from_pretrained(settings.ENTITY_MODEL_PATH)
            self.entity_model.eval() # Set to evaluation mode
            logger.info("Entity model loaded successfully.")
        except Exception as e:
            logger.critical(f"Failed to load entity model: {e}", exc_info=True)
            raise

    def _setup_label_encoder(self):
        logger.info("Setting up LabelEncoder for intents.")
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(settings.INTENT_LABELS_ENGLISH)
        logger.debug(f"LabelEncoder fitted with classes: {self.label_encoder.classes_}")


    # --- Prediction Functions (Synchronous CPU-bound code) ---
    # These now belong to the ModelLoader class

    def predict_intent_sync(self, prompt: str) -> tuple[str, float]:
        """Synchronous intent prediction."""
        if not hasattr(self, 'tokenizer') or not hasattr(self, 'intent_model'):
             logger.error("Intent prediction called before models were loaded.")
             return "Error", 0.0

        # Acquire lock specifically around tokenizer usage
        with self.tokenizer_thread_lock:
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=128)
            except Exception as e:
                logger.error(f"Tokenizer error during intent prediction: {e}", exc_info=True)
                return "Error", 0.0
        # Lock is released here

        try:
            with torch.no_grad():
                outputs = self.intent_model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                confidence = probabilities.max().item()
                predicted_class_idx = probabilities.argmax(dim=-1).item()

            intent = self.label_encoder.inverse_transform([predicted_class_idx])[0]
            return intent, confidence
        except Exception as e:
            logger.error(f"Error during intent model inference: {e}", exc_info=True)
            return "Error", 0.0

    def predict_entities_sync(self, prompt: str) -> tuple[str, float]:
        """Synchronous entity prediction."""
        attention_mask = None
        tokens = None
        offset_mapping = None

        with self.tokenizer_thread_lock:
            # Use padding="max_length" for consistency
            tokenized_output = self.tokenizer(prompt,
                                        return_offsets_mapping=True,
                                        truncation=True,
                                        padding=True,
                                        max_length=512)

            # Assign the entire lists, removing the incorrect [0] index
            if isinstance(tokenized_output.get("input_ids"), list):
                tokens = tokenized_output["input_ids"]
            if isinstance(tokenized_output.get("attention_mask"), list):
                attention_mask = tokenized_output["attention_mask"]
            if isinstance(tokenized_output.get("offset_mapping"), list):
                offset_mapping = tokenized_output["offset_mapping"]


        if tokens is None or attention_mask is None or offset_mapping is None:
            logging.error(f"Tokenization failed or produced unexpected output for text")
            return []


        with torch.no_grad():
            # Create 2D tensors [1, seq_len]
            input_ids_tensor = torch.tensor([tokens])
            attention_mask_tensor = torch.tensor([attention_mask])

            outputs = self.entity_model(
                input_ids=input_ids_tensor,
                attention_mask=attention_mask_tensor
            )
            predictions = torch.argmax(outputs.logits, dim=2)

        # --- Rest of entity processing logic ---
        # predictions[0] correctly gets the results for the single batch item
        predicted_labels = predictions[0].tolist()
        predicted_entities = []
        current_entity = None

        # Iterate using offset_mapping (which is now the correct list)
        for idx, (start, end) in enumerate(offset_mapping):
            # Add bounds checks for safety, comparing idx against lengths
            if idx >= len(predicted_labels) or idx >= len(tokens):
                logging.warning(f"Index {idx} out of bounds for predicted_labels/tokens.")
                continue

            # Skip special tokens/padding based on offset mapping
            # You might refine this condition based on your tokenizer (e.g., check token ID too)
            if start == end:
                continue

            label_id = predicted_labels[idx]
            label = self.id2tag.get(str(label_id), "O")

            if label.startswith("B-") or label.startswith("I-"):
                entity_type = label[2:]
            else:
                entity_type = "O"

            # Use the 'tokens' list here
            # Inside the first loop of predict_entities_sync
            token_text = self.tokenizer.convert_ids_to_tokens([tokens[idx]])[0]

            # Handle 'O' label
            if entity_type == "O":
                if current_entity:
                    # Add cleanup here if needed (recommended)
                    current_entity["word"] = current_entity["word"].strip()  # Basic strip
                    predicted_entities.append(current_entity)
                    current_entity = None
                continue

            # --- Mimic Old Logic ---
            if token_text.startswith("##") and current_entity:
                # Handle subword
                if current_entity["entity"] == entity_type:  # Ensure it continues the *correct* entity
                    current_entity["word"] += token_text.lstrip("##")
                    current_entity["end"] = end
                else:
                    # Subword starts a new entity type (unusual but handle it)
                    # Finalize previous entity
                    if current_entity:
                        current_entity["word"] = current_entity["word"].strip()
                        predicted_entities.append(current_entity)
                    # Start new entity with the cleaned subword
                    current_entity = {
                        "entity": entity_type,
                        "word": token_text.lstrip("##"),  # Start with cleaned subword
                        "start": start,
                        "end": end
                    }

            elif current_entity:
                # Handle new word when already in an entity
                if current_entity["entity"] == entity_type:
                    # New word continues the same entity type
                    current_entity["word"] += " " + token_text  # Add space
                    current_entity["end"] = end
                else:
                    # New word starts a different entity type
                    # Finalize previous
                    current_entity["word"] = current_entity["word"].strip()
                    predicted_entities.append(current_entity)
                    # Start new
                    current_entity = {
                        "entity": entity_type,
                        "word": token_text,
                        "start": start,
                        "end": end
                    }
            else:
                # Handle new word when not previously in an entity
                current_entity = {
                    "entity": entity_type,
                    "word": token_text,
                    "start": start,
                    "end": end
                }

        # Finalize last entity outside loop
        if current_entity:
            current_entity["word"] = current_entity["word"].strip()
            predicted_entities.append(current_entity)

        # REMOVE the second merging loop entirely
        return predicted_entities