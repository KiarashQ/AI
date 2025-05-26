# src/porsit_chatbot/grpc_server/api/speech_service_api.py
import grpc
import logging

from porsit_chatbot.grpc_server.protos import speech_service_pb2
from porsit_chatbot.grpc_server.protos import speech_service_pb2_grpc
from porsit_chatbot.grpc_server.core.stt_tts_logic import SpeechLogic

logger = logging.getLogger(__name__)

class SpeechServiceServicer(speech_service_pb2_grpc.SpeechServiceServicer):
    def __init__(self, speech_logic: SpeechLogic):
        self.speech_logic = speech_logic
        logger.info("SpeechServiceServicer initialized.")

    async def TranscribeAudio(self, request: speech_service_pb2.TranscribeRequest, context):
        logger.info(f"gRPC TranscribeAudio request: format='{request.audio_format}', lang='{request.language_code}', model_choice='{request.stt_model_choice if request.HasField('stt_model_choice') else 'Not set'}'")
        if not request.audio_data:
            logger.warning("TranscribeAudio called with no audio data.")
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Audio data is required.")
            return speech_service_pb2.TranscriptionResponse()
        
        # Pass the stt_model_choice directly; SpeechLogic will use its default if this is None/empty
        requested_stt_config = request.stt_model_choice if request.HasField("stt_model_choice") and request.stt_model_choice else None

        transcript, error_msg = await self.speech_logic.transcribe_audio(
            request.audio_data,
            request.audio_format,
            request.language_code,
            requested_stt_config=requested_stt_config # Pass the choice from the request
        )

        if error_msg:
            logger.error(f"Transcription failed: {error_msg}")
            return speech_service_pb2.TranscriptionResponse(error_message=error_msg)
        
        logger.info("Transcription successful via gRPC.")
        # Confidence is still not populated here, but the structure is ready if you add it.
        return speech_service_pb2.TranscriptionResponse(transcript=transcript or "")

    async def SynthesizeSpeech(self, request: speech_service_pb2.SynthesizeRequest, context):
        logger.info(f"gRPC SynthesizeSpeech request: lang='{request.language_code}', text='{request.text[:50]}...', model_choice='{request.tts_model_choice if request.HasField('tts_model_choice') else 'Not set'}'")
        if not request.text:
            logger.warning("SynthesizeSpeech called with no text.")
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Text is required for synthesis.")
            return speech_service_pb2.SynthesisResponse()
        
        output_format = request.audio_format or "mp3" # Default if not specified by client
        # Pass the tts_model_choice directly
        requested_tts_config = request.tts_model_choice if request.HasField("tts_model_choice") and request.tts_model_choice else None

        audio_data, actual_mime_type, error_msg = await self.speech_logic.synthesize_speech(
            request.text,
            request.language_code,
            request.voice_name if request.HasField("voice_name") else None, # Pass optional voice_name
            output_format,
            requested_tts_config=requested_tts_config # Pass the choice from the request
        )

        if error_msg:
            logger.error(f"Synthesis failed: {error_msg}")
            return speech_service_pb2.SynthesisResponse(error_message=error_msg)
            
        logger.info("Synthesis successful via gRPC.")
        return speech_service_pb2.SynthesisResponse(
            audio_data=audio_data or b"",
            audio_format=actual_mime_type or ""
        )