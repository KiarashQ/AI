# src/porsit_chatbot/grpc_server/core/chatbot_logic.py
import logging
from typing import List, Optional, Union, Tuple
import datetime
import pytz

# LangChain Imports
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser, StrOutputParser
from pydantic import BaseModel, Field, field_validator, ValidationError
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.language_models.chat_models import BaseChatModel
# from openai import AsyncOpenAI # Keep this for direct client if needed, but ChatOpenAI wraps it

# Import config and settings
from porsit_chatbot.grpc_server.config import settings

logger = logging.getLogger(__name__)

# --- Pydantic Models (Define structure for LLM output parsing) ---
class EntityModel(BaseModel):
    word: str = Field(description="The exact Farsi text substring identified as the entity.")
    entity: str = Field(description=f"The English entity type. Must be one of: {settings.ENTITY_LABELS_ENGLISH}")
    start: int = Field(description="The starting character index of the entity in the original Farsi text.")
    end: int = Field(description="The ending character index (exclusive) of the entity in the original Farsi text.")

class NLUOutputModel(BaseModel):
    intent: str = Field(description=f"The single most likely English user intent. Must be one of: {settings.INTENT_LABELS_ENGLISH}")
    confidence: float = Field(description="Confidence score (0.0-1.0) for the intent.", ge=0.0, le=1.0)
    entities: List[EntityModel] = Field(description="List of named entities found in the latest user utterance. Return empty list [] if none found.")

    @field_validator('intent')
    @classmethod
    def validate_intent(cls, value):
        if value not in settings.INTENT_LABELS_ENGLISH:
            # Log warning, but allow Pydantic to proceed. Further handling in API layer.
            logger.warning(f"Pydantic Validator: LLM returned intent '{value}' not in defined list.")
            # raise ValueError(f"Intent must be one of {settings.INTENT_LABELS_ENGLISH}")
        return value


# --- LangChain Setup ---
class ChatbotLogic:
    """Handles LangChain interactions, prompt construction, and LLM calls."""
    def __init__(self):
        logger.info("Initializing ChatbotLogic...")

        # --- Parsers ---
        self.nlu_parser = PydanticOutputParser(pydantic_object=NLUOutputModel)
        self.string_parser = StrOutputParser()

        # --- Prompt Templates ---
        self.nlu_prompt_template = ChatPromptTemplate.from_messages([
             ("system", "{nlu_system_instructions}\n\n{format_instructions}"), # Dynamic system instructions
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])

        self.generation_prompt_template = ChatPromptTemplate.from_messages([
            ("system", "{persona_system_role_content}\n--- Additional Context ---\n{background_context_from_go}"),
            # MessagesPlaceholder(variable_name="history"), # History integrated into context?
            ("human", "{latest_user_utterance}")
        ])

        self.summarize_prompt_template = ChatPromptTemplate.from_messages([
            ("system", "{summary_system_instructions}"),
            ("human", "Please summarize the following conversation history:\n\n{conversation_history}")
        ])

        # --- Default LLM for NLU (Can be overridden) ---
        self.default_nlu_llm = ChatOpenAI(
            api_key=settings.OPENAI_API_KEY,
            model="gpt-4.1-mini", # Match original default
            temperature=0.1,
            max_retries=1,
            model_kwargs={"response_format": {"type": "json_object"}}
        )

        # --- Define NLU Chain ---
        self.nlu_chain = (
            self.nlu_prompt_template
            | self.default_nlu_llm
            | self.nlu_parser
        )
        logger.info("ChatbotLogic initialized with LangChain components.")
  
    def _is_client_in_working_hours(self, client_name: str) -> bool:
        """
        Checks if the given client is currently within their defined working hours.
        Returns True if client has no specific hours defined (considered 24/7 for this check).
        """
        client_hours_config = settings.CLIENT_SPECIFIC_WORKING_HOURS.get(client_name)

        if not client_hours_config:
            logger.debug(f"No specific working hours defined for client '{client_name}'. Assuming 24/7 or time-agnostic behavior.")
            return True # Default to "working hours" or time-agnostic

        try:
            tz_str = client_hours_config["timezone"]
            start_hour = client_hours_config["start_hour"]
            end_hour = client_hours_config["end_hour"]
            working_days = client_hours_config.get("days") # .get() allows it to be optional

            client_tz = pytz.timezone(tz_str)
            now_local = datetime.datetime.now(client_tz)

            if working_days is not None: # Check day only if defined
                if now_local.weekday() not in working_days:
                    logger.info(f"Client '{client_name}' currently off-hours: Not a working day (Day: {now_local.weekday()}).")
                    return False
            
            if not (start_hour <= now_local.hour < end_hour):
                logger.info(f"Client '{client_name}' currently off-hours: Hour {now_local.hour} is outside {start_hour}-{end_hour}.")
                return False
            
            logger.info(f"Client '{client_name}' currently within working hours (Day: {now_local.weekday()}, Hour: {now_local.hour}).")
            return True
        except pytz.exceptions.UnknownTimeZoneError:
            logger.error(f"CRITICAL: Unknown timezone '{tz_str}' for client '{client_name}'. Defaulting to 'is working hours'.")
            return True 
        except KeyError as e:
            logger.error(f"Missing working hours configuration for client '{client_name}': {e}. Defaulting to 'is working hours'.")
            return True
        except Exception as e:
            logger.error(f"Unexpected error checking working hours for '{client_name}': {e}. Defaulting to 'is working hours'.", exc_info=True)
            return True

    def _get_dynamic_prompt_key_suffix(self, client_name: str) -> str:
        """
        Determines the appropriate key suffix (_working_hours, _off_hours)
        if the client has time-dependent settings. Returns empty string otherwise.
        """
        if client_name in settings.CLIENT_SPECIFIC_WORKING_HOURS:
            return "_working_hours" if self._is_client_in_working_hours(client_name) else "_off_hours"
        return ""
  
    def _get_system_prompt_content(self, category: str, client_name: str, key_suffix: str) -> str:
        """
        Fetches system prompt content.
        key_suffix will be "_working_hours", "_off_hours", or ""
        """
        prompts_for_category = settings.SYSTEM_ROLES.get(category, {})
        
        # 1. Try client-specific with mode suffix (e.g., "tipax_off_hours")
        specific_key_with_suffix = f"{client_name}{key_suffix}"
        if key_suffix and specific_key_with_suffix in prompts_for_category:
            logger.info(f"Using prompt: category='{category}', key='{specific_key_with_suffix}'")
            return prompts_for_category[specific_key_with_suffix]

        # 2. Try client-specific without suffix (e.g., "porsit" for persona, or "tipax" if tipax_working_hours was not found)
        if client_name in prompts_for_category:
            logger.info(f"Using prompt: category='{category}', key='{client_name}' (client default or no mode needed)")
            return prompts_for_category[client_name]

        # 3. Fallback to generic with mode suffix (e.g., "generic_off_hours")
        generic_key_with_suffix = f"generic{key_suffix}"
        if key_suffix and generic_key_with_suffix in prompts_for_category:
            logger.info(f"Using prompt: category='{category}', key='{generic_key_with_suffix}' (generic mode fallback)")
            return prompts_for_category[generic_key_with_suffix]
        
        # 4. Fallback to absolute generic default (e.g., "generic_default" or "generic")
        generic_fallback_key = "generic_default" if "generic_default" in prompts_for_category else "generic"
        if generic_fallback_key in prompts_for_category:
            logger.info(f"Using prompt: category='{category}', key='{generic_fallback_key}' (absolute generic fallback)")
            return prompts_for_category[generic_fallback_key]

        logger.warning(f"No suitable prompt for category='{category}', client='{client_name}', suffix='{key_suffix}'. Using hardcoded.")
        if category == "persona": return "You are a helpful assistant."
        if category == "nlu_instructions": return "Analyze user input for intent and entities."
        return "Default system instruction."
    
    def _parse_model_choice(self, model_choice_str: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Parses the model_choice string (e.g., "provider/model_name_for_provider")
        into (provider_key, actual_model_name).
        Returns ("openai", original_model_choice_str) if no provider prefix is found.
        """
        if "/" in model_choice_str:
            parts = model_choice_str.split("/", 1)
            if len(parts) == 2:
                provider_key = parts[0].lower()
                actual_model_name = parts[1]
                return provider_key, actual_model_name
        return "openai", model_choice_str

    def _get_llm_instance(self, model_choice: str, temperature: Optional[float]) -> BaseChatModel:
        """
        Creates a LangChain Chat Model instance based on the model_choice string.
        model_choice is expected to be in format "provider_key/actual_model_name"
        e.g., "anthropic/claude-3-opus-20240229", "openai/gpt-4o-mini"
        """
        provider_key, actual_model_name = self._parse_model_choice(model_choice)
        logger.info(f"Attempting to get LLM for provider: '{provider_key}', model: '{actual_model_name}'")

        provider_config = settings.API_PROVIDER_CONFIG.get(provider_key)

        if not provider_config:
            logger.error(f"Configuration for LLM provider '{provider_key}' not found in settings.API_PROVIDER_CONFIG.")
            if settings.OPENAI_API_KEY: # Try fallback to OpenAI
                logger.warning(f"Falling back to default OpenAI settings for model '{actual_model_name}'.")
                provider_config = settings.API_PROVIDER_CONFIG.get("openai")
                if not provider_config:
                    raise ValueError("Default OpenAI provider configuration missing.")
            else:
                raise ValueError(f"Unsupported LLM provider '{provider_key}' and no fallback OpenAI API key available.")

        target_api_key = provider_config.get("api_key")
        langchain_class_name = provider_config.get("langchain_class", "ChatOpenAI")

        if not target_api_key:
            error_msg = f"API key for provider '{provider_key}' (model: {actual_model_name}) is not configured. Cannot create LLM instance."
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(
            f"Creating LangChain instance: Class='{langchain_class_name}', Provider='{provider_key}', Model='{actual_model_name}', Temp={temperature}"
        )

        if langchain_class_name == "ChatAnthropic":
            return ChatAnthropic(
                model=actual_model_name,
                temperature=temperature,
                anthropic_api_key=target_api_key,
                max_retries=1,
            )
        elif langchain_class_name == "ChatOpenAI":
            target_base_url = provider_config.get("base_url")
            logger.info(f"Using ChatOpenAI with BaseURL='{target_base_url}'")
            return ChatOpenAI(
                model=actual_model_name,
                temperature=temperature,
                api_key=target_api_key,
                base_url=target_base_url,
                max_retries=1,
            )
        elif langchain_class_name == "ChatMistralAI":
            return ChatMistralAI(
                model=actual_model_name,
                temperature=temperature,
                mistral_api_key=target_api_key,
                max_retries=1,
                # You can add other Mistral-specific parameters here if needed
            )
        else:
            logger.error(f"Unknown LangChain class '{langchain_class_name}' specified for provider '{provider_key}'. Attempting with ChatOpenAI.")
            target_base_url = provider_config.get("base_url")
            return ChatOpenAI(
                model=actual_model_name,
                temperature=temperature,
                api_key=target_api_key,
                base_url=target_base_url,
                max_retries=1,
            )

    def _parse_history_string(self, history_str: str) -> List[BaseMessage]:
        """Parses the backend history string into LangChain messages."""
        messages: List[BaseMessage] = []
        if not history_str:
            return messages
        try:
            # Assuming "User:" and "LLM Response:" prefixes
            turns = history_str.strip().split('\n')
            for turn in turns:
                turn_stripped = turn.strip()
                if turn_stripped.startswith("کاربر:"):
                    content = turn_stripped[len("کاربر:"):].strip()
                    if content: messages.append(HumanMessage(content=content))
                elif turn_stripped.startswith("پرسیت:"):
                    content = turn_stripped[len("پرسیت:"):].strip()
                    if content: messages.append(AIMessage(content=content))
                elif turn_stripped: # Handle lines without prefix? Maybe log warning.
                     logger.debug(f"Ignoring unexpected line in history: {turn_stripped[:100]}...")

        except Exception as e:
            logger.warning(f"Could not parse history string: {e}. History: '{history_str[:100]}...'")
            return [] # Return empty list on failure
        return messages

    async def generate_llm_response(self, prompt: str, context_text: str, client_name: str, model_choice: str) -> str:
        """Generates a conversational response using a dynamically configured LLM chain."""
        key_suffix = self._get_dynamic_prompt_key_suffix(client_name)
        logger.info(f"Generate: client='{client_name}', model='{model_choice}', suffix='{key_suffix}'")

        persona_system_role_content_variable = self._get_system_prompt_content(
            "persona", client_name, key_suffix)
        target_temperature: Optional[float] = 0.7
        provider, model_name_for_provider = self._parse_model_choice(model_choice)

        if client_name == "tipax":
            cost_instructions_map = settings.SYSTEM_ROLES.get("cost_estimation_instructions", {})
            tipax_cost_role_content = cost_instructions_map.get("tipax")

            if tipax_cost_role_content:
                cost_model_criteria_openai = provider == "openai" and model_name_for_provider.lower() in ["o4-mini", "o3"]
                cost_model_criteria_gemini = provider == "gemini" and "gemini-2.5-flash-preview-04-17" in model_name_for_provider.lower()

                if cost_model_criteria_openai or cost_model_criteria_gemini:
                    persona_system_role_content_variable = tipax_cost_role_content # Override persona with cost role
                    target_temperature = 0.3
                    logger.info(f"Applying Tipax COST_ROLE with model: {model_choice}")
            else:
                logger.warning("Tipax cost role content not found in settings. Using standard persona.")

        if provider == "openai" and "o4-mini" in model_name_for_provider.lower():
            target_temperature = None

        try:
            llm_instance = self._get_llm_instance(model_choice, target_temperature)
        except ValueError as e:
            logger.error(f"Failed to get LLM instance for model_choice '{model_choice}': {e}")
            raise

        # Chain construction remains the same due to LangChain's composability (LCEL)
        generation_chain = self.generation_prompt_template | llm_instance | self.string_parser

        try:
            # Log the actual model name being used if available from the instance
            invoked_model_name = getattr(llm_instance, 'model_name', getattr(llm_instance, 'model', model_choice))
            logger.debug(f"Invoking generation chain with model: {invoked_model_name}")
            response_text = await generation_chain.ainvoke({
                "persona_system_role_content": persona_system_role_content_variable,
                "background_context_from_go": context_text or "",
                "latest_user_utterance": prompt
            })
            logger.debug(f"LLM generation successful using {model_choice}.")
            return response_text.strip()
        except Exception as e:
            # Use the original model_choice in error message for clarity to the caller
            logger.error(f"Error during LLMResponse chain invocation ({model_choice}): {e}", exc_info=True)
            raise

    async def summarize_conversation_lc(self, conversation_history: str, client_name: str) -> str:
        """Summarizes conversation history using LangChain."""
        if not conversation_history:
             logger.warning("SummarizeConversation called with empty history.")
             return ""

        summary_instructions_map = settings.SYSTEM_ROLES.get("summary_instructions", {})
        summary_system_instructions = summary_instructions_map.get(client_name, "tipax")

        try:
            # Use a specific, cost-effective model for summarization
            summarizer_llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.2,
                max_retries=1,
                api_key=settings.OPENAI_API_KEY
            )

            summarization_chain = self.summarize_prompt_template | summarizer_llm | self.string_parser

            logger.debug("Invoking summarization chain...")
            summary = await summarization_chain.ainvoke({
                "summary_system_instructions": summary_system_instructions,
                "conversation_history": conversation_history
            })
            logger.debug(f"Summarization successful for client '{client_name}'.")
            return summary.strip()

        except Exception as e:
            logger.error(f"3001: Error during SummarizeConversation chain call: {e}", exc_info=True)
            raise

    async def classify_all_lc(self, text: str, conversation_history_str: str, client_name: str, model_choice: Optional[str] = None) -> NLUOutputModel:
        key_suffix = self._get_dynamic_prompt_key_suffix(client_name)
        logger.info(f"NLU: client='{client_name}', model='{model_choice or 'default_nlu_llm'}', suffix='{key_suffix}'. Text: '{text[:50]}...'")

        nlu_system_instructions = self._get_system_prompt_content("nlu_instructions", client_name, key_suffix)

        try:
            history_messages = self._parse_history_string(conversation_history_str)

            logger.debug("Invoking LangChain NLU chain...")
            # Provide format instructions to the NLU chain
            parsed_data: NLUOutputModel = await self.nlu_chain.ainvoke({
                "nlu_system_instructions": nlu_system_instructions,
                "input": text,
                "history": history_messages,
                "format_instructions": self.nlu_parser.get_format_instructions(),
            })
            logger.info(f"Successfully parsed LangChain NLU response.")
            logger.debug(f"Parsed NLU data (Pydantic): {parsed_data}")
            return parsed_data

        except ValidationError as e:
            logger.error(f"Pydantic validation error during NLU parsing: {e}", exc_info=True)
            # LLM output didn't match the NLUOutputModel schema
            raise ValueError(f"LLM output validation failed: {e}") from e # Raise specific error type
        except Exception as e:
            # Includes OutputParserException, API errors, etc.
            logger.error(f"Error invoking or parsing NLU chain: {e}", exc_info=True)
            # Check if it's a parser error specifically
            if "OutputParserException" in str(type(e).__name__):
                 raise ValueError("NLU failed due to LLM output format error.") from e
            else:
                 raise # Re-raise other exceptions