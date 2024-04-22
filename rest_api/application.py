import logging
from typing import Optional, Annotated
import asyncio
import time
import urllib.parse
from fastapi import Depends

# ASR
from asr.utils import map_speaker_to_words
from asr.utils import process_speaker_aware_transcription, get_speaker_ts, map_speaker_to_sentences
from asr.utils import get_talk_time, get_switches_per_conversation, get_longest_monologue, get_words_per_min

# restapi
from rest_api.utils import get_app, load_config_file
from rest_api.schemas import TranscriptionResponse, BaseDiarizationResponse, AudioFilePathSchema
from rest_api.config import MODEL_CONFIG_FILE, FILE_UPLOAD_PATH

# ray serve
from ray.serve.handle import DeploymentHandle
from rest_api.deployments import (
    TranscriptionDeployment,
    TranscriptionAlignmentDeployment,
    DiarizationDeployment,
)
from ray import serve

logging.basicConfig(format="%(asctime)s %(message)s",
                    datefmt="%m/%d/%Y %I:%M:%S %p")
logger = logging.getLogger(__name__)

model_config = load_config_file(MODEL_CONFIG_FILE)
app = get_app()

logger.info(
    "Open http://127.0.0.1:8000/docs to see Swagger API Documentation."
)


AudioPathDep = Annotated[AudioFilePathSchema, Depends()]

@serve.deployment
@serve.ingress(app)
class ASRIngress:
    def __init__(
        self,
        transcription_handle: DeploymentHandle,
        alignment_handle: DeploymentHandle,
        diarization_handle: DeploymentHandle,
    ) -> None:

        self.transcription_handle = transcription_handle
        self.alignment_handle = alignment_handle
        self.diarization_handle = diarization_handle

    @app.get("/status")
    def check_status(self):
        """
        Use this endpoint to check if the server is ready to take any requests.
        Also, to check the health of all deployments.

        """
        return serve.status()

    @app.get("/transcription")
    async def infer_transcription(
        self, audio_path : AudioPathDep, align_transcription: Optional[bool] = False
    ) -> TranscriptionResponse:
        file_name = audio_path.dict()['audio_path']
        audio = urllib.parse.urljoin(FILE_UPLOAD_PATH,file_name)
        return await self.run_transcription(audio, align_transcription)

    @app.get("/diarization")
    async def infer_diarization(self, audio_path : AudioPathDep) -> BaseDiarizationResponse:
        file_name = audio_path.dict()['audio_path']
        audio = urllib.parse.urljoin(FILE_UPLOAD_PATH,file_name)
        return await self.run_diarization(audio)

    async def run_transcription(
        self, audio : str, align_transcription=False, post_process_alignment=True
    ):
        if align_transcription is False:
            return await self.transcription_handle.remote(audio)
  
        transcription_results = self.transcription_handle.infer.remote(
            audio)
        if post_process_alignment is True:
            alignment_results = self.alignment_handle.remote(
                audio, transcription_results
            )
        else:
            alignment_results = self.alignment_handle.infer.remote(
                audio, transcription_results
            )
        return await alignment_results

    async def run_diarization(self, audio : str, return_rttm : bool =True):
        return await self.diarization_handle.remote(audio, return_rttm)

    @app.get("/speaker_aware_transcription")
    async def infer_speaker_aware_transcription(self, audio_path:AudioPathDep) -> TranscriptionResponse:
        file_name = audio_path.dict()['audio_path']
        audio = urllib.parse.urljoin(FILE_UPLOAD_PATH,file_name)
        s = time.perf_counter()
        alignment_results, diarization_results = await asyncio.gather(
            self.run_transcription(audio, True, False),
            self.run_diarization(audio, False),
        )

        sts = get_speaker_ts(diarization_results)
        wsm = map_speaker_to_words(
            alignment_results["word_segments"], sts
        )
        speaker_aware_transcription = map_speaker_to_sentences(wsm, sts)
        speaker_aware_transcription = process_speaker_aware_transcription(
            speaker_aware_transcription
        )
        elapsed = time.perf_counter() - s
        print(f"{__file__} executed in {elapsed:0.2f} seconds.")
        return speaker_aware_transcription

    @app.post("/speech_analysis")
    async def analyse_speech(self, transcription_results: TranscriptionResponse, audio_duration: float):
        transcription_results = transcription_results.dict()['segments']
        speaker_talk_time = get_talk_time(transcription_results, audio_duration)
        speaker_switches = get_switches_per_conversation(transcription_results)
        lm_dict = get_longest_monologue(transcription_results)
        spk_words_sum = get_words_per_min(transcription_results, speaker_talk_time)

        result = {}

        result['speaker_talk_time_distribution'] = speaker_talk_time
        result['switches_per_conversation'] = speaker_switches
        result['longest_monologue'] = lm_dict
        result['words_per_min'] = spk_words_sum

        return result

asr_ingress = ASRIngress.bind(TranscriptionDeployment.bind(model_config), # pylint: disable=E1101
                            TranscriptionAlignmentDeployment.bind(model_config), # pylint: disable=E1101
                            DiarizationDeployment.bind(model_config)) # pylint: disable=E1101

