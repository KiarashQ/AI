# src/porsit_chatbot/grpc_server/main.py
import asyncio
import grpc.aio
from concurrent.futures import ThreadPoolExecutor
import logging
from dotenv import load_dotenv

load_dotenv()

# Import settings and setup logging first
from porsit_chatbot.grpc_server.config import settings
settings.setup_logging()

# Import core components and servicer
from porsit_chatbot.grpc_server.core.model_loader import ModelLoader
from porsit_chatbot.grpc_server.core.chatbot_logic import ChatbotLogic
from porsit_chatbot.grpc_server.api.chatbot_service import ChatbotServiceServicer
from porsit_chatbot.grpc_server.protos import models_pb2_grpc

logger = logging.getLogger(__name__)

# --- Global Resource Initialization ---
# These are initialized once when the module loads (and thus when the server starts)
# Handle potential errors during initialization gracefully.
try:
    model_loader = ModelLoader()
except Exception as e:
    logger.critical(f"CRITICAL: Failed to initialize ModelLoader: {e}. Server cannot start.", exc_info=True)
    # Depending on deployment, might exit or raise to prevent server start
    raise SystemExit("Failed to load ML models.") from e

try:
    chatbot_logic = ChatbotLogic()
except Exception as e:
    logger.critical(f"CRITICAL: Failed to initialize ChatbotLogic: {e}. Server cannot start.", exc_info=True)
    raise SystemExit("Failed to initialize LangChain components.") from e


async def serve():
    """Creates, configures, and runs the async gRPC server."""
    server = grpc.aio.server(ThreadPoolExecutor(max_workers=30)) # Use thread pool for sync tasks

    # Instantiate the servicer, injecting the initialized dependencies
    servicer = ChatbotServiceServicer(model_loader, chatbot_logic)

    # Register the servicer with the server
    models_pb2_grpc.add_ModelServiceServicer_to_server(servicer, server)

    listen_addr = settings.GRPC_LISTEN_ADDRESS
    server.add_insecure_port(listen_addr)

    logger.info(f"Starting async gRPC server on {listen_addr}...")
    await server.start()
    logger.info("Server started successfully.")

    # Graceful shutdown handling
    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Received KeyboardInterrupt, stopping server...")
        # Optionally add a grace period
        await server.stop(grace=5.0)
        logger.info("Server stopped gracefully.")
    except asyncio.CancelledError:
         logger.info("Server task cancelled, stopping server...")
         await server.stop(grace=1.0)
         logger.info("Server stopped.")
    except Exception as e:
         logger.error(f"Unexpected error during server runtime: {e}", exc_info=True)
         await server.stop(grace=1.0)
         logger.info("Server stopped due to error.")


if __name__ == "__main__":
    # Validate API keys before starting server (optional, but good practice)
    settings.validate_keys()

    try:
        asyncio.run(serve())
    except SystemExit as e:
         logger.critical(f"Server failed to start due to initialization error: {e}")
         # Exit code indicates failure
         exit(1)
    except Exception as e:
        logger.critical(f"Failed to run the async server: {e}", exc_info=True)
        exit(1) # Exit with error status