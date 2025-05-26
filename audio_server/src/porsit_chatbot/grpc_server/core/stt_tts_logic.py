# src/porsit_chatbot/grpc_server/core/stt_tts_logic.py
import logging
import io
import os
import tempfile
import asyncio
from typing import Optional, Tuple

from openai import AsyncOpenAI as OpenAIClient

try:
    from speechmatics.models import ConnectionSettings as SpeechmaticsConnectionSettings
    from speechmatics.batch_client import BatchClient as SpeechmaticsBatchClient
    from httpx import HTTPStatusError as SpeechmaticsHTTPStatusError
    SPEECHMATICS_AVAILABLE = True
except ImportError:
    SPEECHMATICS_AVAILABLE = False
    SpeechmaticsConnectionSettings = None
    SpeechmaticsBatchClient = None
    SpeechmaticsHTTPStatusError = None
    logging.warning("Speechmatics Python library not installed. Speechmatics STT will not be available.")

from porsit_chatbot.grpc_server.config import settings

logger = logging.getLogger(__name__)

class SpeechLogic:
    def __init__(self):
        stt_default_conf = settings.STT_PROVIDER_MODEL_CONFIG.lower()
        tts_default_conf = settings.TTS_PROVIDER_MODEL_CONFIG.lower()
        logger.info(f"Initializing SpeechLogic with default STT config: {stt_default_conf}, default TTS config: {tts_default_conf}")

        self.openai_client = None
        self.speechmatics_settings: Optional[SpeechmaticsConnectionSettings] = None # Type hint

        if settings.OPENAI_API_KEY:
            self.openai_client = OpenAIClient(api_key=settings.OPENAI_API_KEY)
            logger.info("OpenAI client initialized (API key found).")
        else:
            if stt_default_conf.startswith("openai/") or tts_default_conf.startswith("openai/"):
                logger.warning("OpenAI is a potential provider (based on default config or common usage) but OPENAI_API_KEY is not set.")

        if SPEECHMATICS_AVAILABLE and settings.SPEECHMATICS_API_KEY:
            self.speechmatics_settings = SpeechmaticsConnectionSettings(
                url=settings.SPEECHMATICS_BATCH_API_URL,
                auth_token=settings.SPEECHMATICS_API_KEY,
            )
            logger.info("Speechmatics settings configured (library and API key found).")
        elif stt_default_conf.startswith("speechmatics/"):
            if not SPEECHMATICS_AVAILABLE:
                 logger.warning("Speechmatics is suggested by default STT config but library is not installed.")
            elif not settings.SPEECHMATICS_API_KEY:
                logger.warning("Speechmatics is suggested by default STT config but SPEECHMATICS_API_KEY is not set.")

    def _parse_provider_model_config(self, config_str: str, service_type: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Parses a 'provider/model_identifier' string.
        Returns (provider_name, model_identifier, error_message).
        """
        if not config_str:
            err = f"{service_type} configuration string is empty."
            logger.error(err)
            return None, None, err
        
        parts = config_str.split('/', 1)
        provider_name = parts[0].lower()
        model_identifier = None

        if len(parts) > 1 and parts[1]:
            model_identifier = parts[1] # Could be "whisper-1", "batch", "tts-1-hd"
        elif provider_name in ["openai"]: # Providers that strictly need a model_id
            err = f"Model identifier missing for {service_type} provider '{provider_name}' in config '{config_str}'. Expected 'provider/model_identifier'."
            logger.error(err)
            return None, None, err
        # For some providers, model_identifier might be optional or implied (e.g., "speechmatics/batch" where batch is the key part)

        return provider_name, model_identifier, None

    async def transcribe_audio(
        self,
        audio_data: bytes,
        audio_format: str,
        language_code: str,
        requested_stt_config: Optional[str] = None  # Renamed from stt_model_choice for clarity
    ) -> Tuple[Optional[str], Optional[str]]:
        effective_stt_config = requested_stt_config or settings.STT_PROVIDER_MODEL_CONFIG
        logger.info(f"STT: Effective config string: '{effective_stt_config}' (Requested: '{requested_stt_config}')")

        provider_name, model_id, error_msg = self._parse_provider_model_config(effective_stt_config, "STT")
        if error_msg:
            return None, error_msg

        if provider_name == "openai":
            if not self.openai_client:
                return None, "OpenAI client not initialized (OpenAI API key likely missing)."
            try:
                model_to_use = model_id # e.g., "whisper-1", "gpt-4o-transcribe"
                if not model_to_use: # Should have been caught by _parse_provider_model_config for openai
                    return None, f"OpenAI STT model not specified in config '{effective_stt_config}'."

                dummy_filename = f"audio_input.{audio_format.lower()}"
                audio_file_tuple = (dummy_filename, audio_data, f"audio/{audio_format.lower()}")
                whisper_language = language_code.split('-')[0].lower() if '-' in language_code else language_code.lower()
                response_format_stt = "text" # Keep as text for simplicity, or json for more details

                logger.info(f"Transcribing with OpenAI STT (model: {model_to_use}, lang: {whisper_language})")
                transcription_obj = await self.openai_client.audio.transcriptions.create(
                    model=model_to_use,
                    file=audio_file_tuple,
                    language=whisper_language,
                    response_format=response_format_stt
                )
                transcript = transcription_obj.strip() if isinstance(transcription_obj, str) else getattr(transcription_obj, 'text', "").strip()
                logger.info(f"OpenAI STT transcription successful: {transcript[:100]}...")
                return transcript, None
            except Exception as e:
                logger.error(f"OpenAI STT error: {e}", exc_info=True)
                return None, str(e)

        elif provider_name == "speechmatics":
            # For "speechmatics/batch", model_id will be "batch"
            if model_id != "batch": # Or handle other Speechmatics models/modes if they exist
                return None, f"Unsupported Speechmatics STT model/mode: '{model_id}'. Expected 'batch'."
            if not SPEECHMATICS_AVAILABLE or not self.speechmatics_settings:
                return None, "Speechmatics STT not available or not configured (check library install and API key)."
            try:
                sm_language = language_code.split('-')[0].lower() if '-' in language_code else language_code.lower()
                conf = {"type": "transcription", "transcription_config": {"language": sm_language}}
                logger.info(f"Transcribing with Speechmatics Batch (lang: {sm_language})")

                with tempfile.NamedTemporaryFile(mode="wb", suffix=f".{audio_format.lower()}", delete=False) as tmp_audio_file:
                    tmp_audio_file.write(audio_data)
                    tmp_audio_file_path = tmp_audio_file.name
                
                transcript_text = None
                error_msg_sm = None
                try:
                    def run_speechmatics_job():
                        # Ensure speechmatics_settings is not None (checked above)
                        with SpeechmaticsBatchClient(self.speechmatics_settings) as client: # type: ignore
                            job_id = client.submit_job(audio=tmp_audio_file_path, transcription_config=conf)
                            logger.info(f'Speechmatics job {job_id} submitted, waiting for transcript...')
                            return client.wait_for_completion(job_id, transcription_format='txt')
                    transcript_text = await asyncio.to_thread(run_speechmatics_job)
                    if transcript_text:
                        logger.info(f"Speechmatics transcription successful: {transcript_text[:100]}...")
                except SpeechmaticsHTTPStatusError as e: # type: ignore
                    logger.error(f"Speechmatics API HTTP error: {e.response.status_code} - {e.response.text}", exc_info=True)
                    error_msg_sm = f"Speechmatics API error: {e.response.status_code}"
                except Exception as e:
                    logger.error(f"Speechmatics STT processing error: {e}", exc_info=True)
                    error_msg_sm = str(e)
                finally:
                    if os.path.exists(tmp_audio_file_path):
                        os.remove(tmp_audio_file_path)
                return transcript_text, error_msg_sm
            except Exception as e:
                logger.error(f"General error during Speechmatics STT setup: {e}", exc_info=True)
                return None, str(e)
        else:
            err = f"Unsupported STT provider: '{provider_name}' from config '{effective_stt_config}'"
            logger.error(err)
            return None, err

    async def synthesize_speech(
        self,
        text: str,
        language_code: str,
        voice_name: Optional[str],
        output_audio_format: str,
        requested_tts_config: Optional[str] = None # Renamed from tts_model_choice
    ) -> Tuple[Optional[bytes], Optional[str], Optional[str]]:
        effective_tts_config = requested_tts_config or settings.TTS_PROVIDER_MODEL_CONFIG
        logger.info(f"TTS: Effective config string: '{effective_tts_config}' (Requested: '{requested_tts_config}')")

        provider_name, model_id, error_msg = self._parse_provider_model_config(effective_tts_config, "TTS")
        if error_msg:
            return None, None, error_msg

        if provider_name == "openai":
            if not self.openai_client:
                return None, None, "OpenAI client not initialized (OpenAI API key likely missing)."
            try:
                model_to_use = model_id # e.g., "tts-1", "tts-1-hd"
                if not model_to_use: # Should have been caught by _parse_provider_model_config
                    return None, None, f"OpenAI TTS model not specified in config '{effective_tts_config}'."

                selected_voice = voice_name or settings.OPENAI_TTS_DEFAULT_VOICE
                
                # Map user format to OpenAI's expected `response_format`
                openai_response_format = output_audio_format.lower()
                if openai_response_format == "ogg_opus": # OpenAI expects "opus" for this
                    openai_response_format = "opus"
                
                supported_openai_formats = {"mp3", "opus", "aac", "flac", "wav", "pcm"}
                if openai_response_format not in supported_openai_formats:
                    err_msg = f"Unsupported audio format '{openai_response_format}' for OpenAI TTS. Supported: {supported_openai_formats}"
                    logger.error(err_msg)
                    return None, None, err_msg

                # Determine the correct MIME type for the response
                actual_mime_type = f"audio/{openai_response_format}"
                if openai_response_format == "mp3":
                    actual_mime_type = "audio/mpeg" # More standard for mp3
                # Add other specific MIME types if necessary (e.g., audio/aac, audio/flac, audio/wav, audio/L16 for pcm)

                logger.info(f"Synthesizing with OpenAI TTS (model: {model_to_use}, voice: {selected_voice}, format: {openai_response_format})")
                response = await self.openai_client.audio.speech.create(
                    model=model_to_use,
                    voice=selected_voice,
                    input=text,
                    response_format=openai_response_format
                )
                audio_bytes = await response.aread()
                logger.info(f"OpenAI TTS synthesis successful, bytes: {len(audio_bytes)}, mime_type: {actual_mime_type}")
                return audio_bytes, actual_mime_type, None
            except Exception as e:
                logger.error(f"OpenAI TTS error: {e}", exc_info=True)
                return None, None, str(e)
        else:
            err = f"Unsupported TTS provider: '{provider_name}' from config '{effective_tts_config}'"
            logger.error(err)
            return None, None, err