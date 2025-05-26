# src/porsit_chatbot/grpc_server/speech_main.py
import asyncio
import grpc
import logging
from concurrent import futures

from porsit_chatbot.grpc_server.protos import speech_service_pb2_grpc
from porsit_chatbot.grpc_server.api.speech_service_api import SpeechServiceServicer
from porsit_chatbot.grpc_server.core.stt_tts_logic import SpeechLogic
from porsit_chatbot.grpc_server.config import settings # For logging setup

async def serve_speech_service():
    settings.setup_logging()
    logger = logging.getLogger(__name__)

    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
    
    # Initialize dependencies
    speech_logic = SpeechLogic()
    servicer = SpeechServiceServicer(speech_logic)
    speech_service_pb2_grpc.add_SpeechServiceServicer_to_server(servicer, server)
    
    # Define a port for this service
    listen_address = settings.SPEECH_GRPC_LISTEN_ADDRESS
    
    server.add_insecure_port(listen_address)
    logger.info(f"Starting Speech STT/TTS gRPC server on {listen_address}...")
    await server.start()
    logger.info("Speech STT/TTS Server started successfully.")
    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Speech STT/TTS Server stopping...")
    finally:
        await server.stop(0)
        logger.info("Speech STT/TTS Server stopped.")

if __name__ == "__main__":
    asyncio.run(serve_speech_service())