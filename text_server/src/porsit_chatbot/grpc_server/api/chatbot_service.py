# src/porsit_chatbot/grpc_server/api/chatbot_service.py
import grpc
import asyncio
import logging

# Import gRPC generated files
from porsit_chatbot.grpc_server.protos import models_pb2
from porsit_chatbot.grpc_server.protos import models_pb2_grpc

# Import core logic components
from porsit_chatbot.grpc_server.core.model_loader import ModelLoader
from porsit_chatbot.grpc_server.core.chatbot_logic import ChatbotLogic, NLUOutputModel
from porsit_chatbot.grpc_server.config import settings # For labels

logger = logging.getLogger(__name__)

# Numerical error guide (Consider centralizing or using enums)
# 1000 API/ 2000 loading/ 3000 parsing/ 4000 unexpected
# 0001 LLMResponse/ 0002 SummarizeConversation/ 0003 ClassifyAll/ 0004 ClassifyAll_LC


class ChatbotServiceServicer(models_pb2_grpc.ModelServiceServicer):
    """
    gRPC Servicer implementation for the Chatbot ModelService.
    Delegates core logic to ModelLoader and ChatbotLogic.
    """
    def __init__(self, model_loader: ModelLoader, chatbot_logic: ChatbotLogic):
        self.model_loader = model_loader
        self.chatbot_logic = chatbot_logic
        logger.info("ChatbotServiceServicer initialized with dependencies.")

    async def LLMResponse(self, request: models_pb2.GenerateRequest, context):
        """Handles the LLMResponse RPC call."""
        model_choice = request.model_choice
        prompt = request.prompt
        context_text = request.context
        client_name = request.client_name or "porsit" # Default client

        logging.info(f"gRPC LLMResponse: client='{client_name}', model='{model_choice}'. Context provided: {bool(context_text)}")
        logging.debug(f"gRPC LLMResponse context received: {context_text[:200]}...")

        try:
            response_text = await self.chatbot_logic.generate_llm_response(
                prompt=prompt,
                context_text=context_text,
                client_name=client_name,
                model_choice=model_choice
            )
            return models_pb2.GenerateResponse(response=response_text)

        except ValueError as e: # Catch specific errors like invalid model/key
             logger.error(f"LLMResponse configuration error: {e}")
             # Use more specific error codes if available
             await context.abort(grpc.StatusCode.FAILED_PRECONDITION, f"1021: Configuration error: {e}")
        except Exception as e:
            # Catch-all for other errors (API errors, timeouts, etc.)
            logger.error(f"Error during LLMResponse logic: {e}", exc_info=True)
            error_code = "1001" # Default API error
            err_str = str(e).lower()
            if "rate limit" in err_str: error_code = "1011"
            elif "authentication" in err_str: error_code = "1021" # Should be caught by ValueError above ideally
            elif "timeout" in err_str: error_code = "1031"

            await context.abort(grpc.StatusCode.INTERNAL, f"{error_code}: Failed to generate response: {e}")


    async def SummarizeConversation(self, request: models_pb2.SummarizeRequest, context):
        """Handles the SummarizeConversation RPC call."""
        conversation_history = request.conversation_history
        client_name = request.client_name or "porsit"
        logging.info("gRPC SummarizeConversation request received.")

        try:
            summary = await self.chatbot_logic.summarize_conversation_lc(
                conversation_history=conversation_history,
                client_name=client_name,
            )
            return models_pb2.GenerateResponse(response=summary)
        except Exception as e:
            logger.error(f"Error during SummarizeConversation logic: {e}", exc_info=True)
            error_code = "1002" # Default Summarization API error
            await context.abort(grpc.StatusCode.INTERNAL, f"{error_code}: Summarization failed: {e}")


    async def ClassifyAll(self, request: models_pb2.ClassifyAllRequest, context):
        """Handles the ClassifyAll RPC call (using local ML models)."""
        text = request.text
        logging.info(f"gRPC ClassifyAll request for: {text[:50]}...")

        if not self.model_loader or not self.model_loader.tokenizer: # Basic check
            logger.error("ClassifyAll called but ModelLoader not ready.")
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "2003: Models not loaded or available.")
            return models_pb2.CombinedResponse() # Return empty

        try:
            # Run sync predictions in thread pool via asyncio.to_thread
            intent_future = asyncio.to_thread(self.model_loader.predict_intent_sync, text)
            entity_future = asyncio.to_thread(self.model_loader.predict_entities_sync, text)

            # Await results
            (intent, confidence), entities = await asyncio.gather(intent_future, entity_future)

            logging.info(f"ClassifyAll completed: Intent={intent}, Confidence={confidence:.2f}, Entities Count={len(entities)}")

            # --- Build Protobuf Response ---
            intent_response = models_pb2.IntentResponse(
                intent=intent,
                confidence=confidence
            )
            entity_responses = [
                models_pb2.Entity(
                    word=entity.get("word", ""),
                    entity=entity.get("entity", "O"),
                    start=entity.get("start", 0),
                    end=entity.get("end", 0)
                )
                for entity in entities # entities is the list of dicts from predict_entities_sync
            ]
            entity_response = models_pb2.EntityResponse(entities=entity_responses)

            return models_pb2.CombinedResponse(
                intent_response=intent_response,
                entity_response=entity_response
            )
        except Exception as e:
            logger.error(f"Error during ClassifyAll logic or prediction: {e}", exc_info=True)
            # 4000 unexpected, 2000 loading/model related
            error_code = "4003" if "predict" not in str(e).lower() else "2003"
            await context.abort(grpc.StatusCode.INTERNAL, f"{error_code}: Combined classification failed: {e}")


    async def ClassifyAll_LC(self, request: models_pb2.ClassifyRequestLC, context):
        """Handles the ClassifyAll_LC RPC call (using LangChain)."""
        text = request.text
        conversation_history_str = request.conversation_history
        client_name = request.client_name or "porsit"
        logging.info(f"gRPC ClassifyAll_LC request. History: {bool(conversation_history_str)}. Text: {text[:100]}...")

        try:
            # Call the core logic function
            parsed_data: NLUOutputModel = await self.chatbot_logic.classify_all_lc(
                text=text,
                conversation_history_str=conversation_history_str,
                client_name=client_name,
            )

            # --- Validate and Build Protobuf Response ---
            intent = parsed_data.intent
            confidence = parsed_data.confidence
            entities_data = parsed_data.entities # List of EntityModel

            # The Pydantic model validator already warns, but you could enforce here.
            if intent not in settings.INTENT_LABELS_ENGLISH:
                 logger.warning(f"ClassifyAll_LC: LLM returned intent '{intent}' not in known list. Using 'Unclear'.")
                 intent = "Unclear"
                 confidence = 0.5 # Assign default confidence for unclear

            entity_responses = []
            for entity_model in entities_data:
                entity_label = entity_model.entity
                # Validate entity label against known list (excluding 'O')
                if entity_label != "O" and entity_label not in settings.ENTITY_LABELS_ENGLISH:
                     logger.warning(f"ClassifyAll_LC: LLM returned entity type '{entity_label}' not in known list. Skipping entity: '{entity_model.word}'")
                     continue

                entity_responses.append(
                    models_pb2.Entity(
                        word=entity_model.word,
                        entity=entity_label,
                        start=entity_model.start,
                        end=entity_model.end
                    )
                )

            logging.info(f"ClassifyAll_LC completed: Intent={intent}, Confidence={confidence:.2f}, Entities Count={len(entity_responses)}")

            intent_response = models_pb2.IntentResponse(intent=intent, confidence=confidence)
            entity_response = models_pb2.EntityResponse(entities=entity_responses)

            return models_pb2.CombinedResponse(
                intent_response=intent_response,
                entity_response=entity_response
            )

        except ValueError as e: # Catch specific errors from chatbot_logic (validation, parsing)
             logger.error(f"ClassifyAll_LC Value Error (likely parsing/validation): {e}", exc_info=True)
             error_code = "3004" # Parsing/Format Error Code
             await context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"{error_code}: NLU classification failed: {e}")
        except Exception as e:
             logger.error(f"Unexpected error during ClassifyAll_LC logic: {e}", exc_info=True)
             error_code = "4004" # Unexpected NLU Error
             await context.abort(grpc.StatusCode.INTERNAL, f"{error_code}: NLU classification failed: {e}")