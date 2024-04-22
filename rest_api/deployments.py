import logging

# ASR
from asr.transcription import tr_factory
from asr.dia import dia_factory
from asr.utils import process_transcription_results, Benchmark

# Ray serve
from ray import serve


@serve.deployment
class TranscriptionDeployment:
    """
    A Transcription model deployment. It loads a requested transcription model.
    """

    def __init__(self, config):
        self.config = config
        self.model = tr_factory(
            self.config["transcription"]["backend"],
            self.config["transcription"]
        )
        self.bench = Benchmark()
        self.track = self.bench.track

    def infer(self, audio):
        transcription_results = None
        try:
            with self.track("Transcription"):
                transcription_results = self.model(audio)
        except Exception as exp: # pylint: disable=W0718
            logging.error("An error ocurred during transcription %s", exp)
        return transcription_results

    def __call__(self, audio):
        transcription_results = self.infer(audio)
        transcription_results = process_transcription_results(
            transcription_results)
        return transcription_results


@serve.deployment
class TranscriptionAlignmentDeployment:
    """
    A Transcription model deployment. It loads a requested transcription model.
    """

    def __init__(self, config):
        self.config = config
        self.model = tr_factory(
            self.config["alignment"]["backend"], self.config["alignment"]
        )
        self.bench = Benchmark()
        self.track = self.bench.track

    def infer(self, audio, transcription_results):
        alignment_results = None
        try:
            with self.track("Alignment"):
                alignment_results = self.model(
                    audio, transcription=transcription_results
                )
        except Exception as exp: # pylint: disable=W0718
            logging.error("An error ocurred during transcription alignment %s",
                           exp)

        return alignment_results

    def __call__(self, audio, transcription_results):
        alignment_results = self.infer(
            audio, transcription_results=transcription_results
        )
        alignment_results = process_transcription_results(alignment_results)

        return alignment_results


@serve.deployment
class DiarizationDeployment:
    """
    A Transcription model deployment. It loads a requested transcription model.
    """

    def __init__(self, config):
        self.config = config
        self.model = dia_factory(
            self.config["diarization"]["backend"], self.config["diarization"]
        )
        self.bench = Benchmark()
        self.track = self.bench.track

    def infer(self, audio):
        diarization_results = None
        try:
            with self.track("Diarization"):
                diarization_results = self.model(audio)
        except Exception as exp: # pylint: disable=W0718
            logging.error("An error ocurred during diarization %s", exp)
        return diarization_results

    def __call__(self, audio, return_rttm=True):
        diarization_results = self.infer(audio)
        rttm = diarization_results.to_rttm()
        uri = diarization_results.uri
        if return_rttm:
            output = {"rttm": rttm, "uri": uri}
        else:
            output = diarization_results
        return output
