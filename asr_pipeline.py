"""
Given the configuration, ASRPipeline loads models. 
Then it runs model inference on given input audio file and 
returns the results after post processing.

"""
import sys

# ASR
from asr.transcription import tr_factory
from asr.dia import dia_factory
from asr import utils
from asr.utils import Benchmark


class ASRPipeline:

    """
    Given the configuration, ASRPipeline loads models. 
    Then it runs model inference on given input audio file and 
    returns the results after post processing.
    """

    def __init__(self, config):

        self.config = config

        self.tr_model = None
        self.al_model = None
        self.dia_model = None

        self.bench = Benchmark()
        self.track = self.bench.track
        self.load_transcription_model()
        if self.config["is_alignment"] is True:
            self.load_alignment_model()

        if self.config["is_diarization"] is True:
            self.load_diarization_model()

    def load_transcription_model(self):

        """
        loads transcription model.
        """

        with self.track(
                f"Loading {self.config['transcription']['backend']} | \
                {self.config['transcription']['topology']} | \
                {self.config['transcription']['compute_type']}  Model"
            ):
            self.tr_model = tr_factory(
                self.config["transcription"]["backend"],
                self.config["transcription"]
            )

    def load_alignment_model(self):

        """
        Loads alignment model
        """

        with self.track(f"Loading {self.config['alignment']['backend']} Model"):
            self.al_model = tr_factory(
                self.config["alignment"]["backend"], self.config["alignment"]
            )

    def load_diarization_model(self):

        """
        Loads diarization  model
        """

        with self.track(f"Loading {self.config['diarization']['backend']} Model"):
            self.dia_model = dia_factory(
                self.config["diarization"]["backend"], self.config["diarization"]
            )

    def process_results(self, transcription_results, alignment_results, dia_results):

        """
        Post processing of the results return from the model inference.
        """

        output_transcription = []
        output_alignment = None
        output_diarization = None
        for segment in transcription_results["segments"]:
            transcription_dict = {}
            transcription_dict["speaker"] = "Unknown"
            transcription_dict["start_time"] = segment["start"]
            transcription_dict["end_time"] = segment["end"]
            transcription_dict["text"] = segment["text"]
            output_transcription.append(transcription_dict)
        if alignment_results is not None:
            output_alignment = []
            for segment in alignment_results["segments"]:
                alignment_dict = {}
                alignment_dict["speaker"] = "Unknown"
                alignment_dict["start_time"] = segment["start"]
                alignment_dict["end_time"] = segment["end"]
                alignment_dict["text"] = segment["text"]
                output_alignment.append(alignment_dict)
        if dia_results is not None:
            output_diarization = []
            for segment in dia_results:
                dia_dict = {}
                dia_dict["speaker"] = segment["speaker"]
                dia_dict["start_time"] = segment["st"] / 1000
                dia_dict["end_time"] = segment["et"] / 1000
                dia_dict["text"] = segment["text"]
                output_diarization.append(dia_dict)
        return output_transcription, output_alignment, output_diarization

    def __call__(self, audio_file):

        """
        callable to run inference in input audio_file.
        """

        alignment_results = None
        dia_results = None
        with self.track("Transcription"):
            try:
                transcription_results = self.tr_model(audio_file)
            except Exception as exp: # pylint: disable=W0718
                print("!!Error encountered during Transcription:", exp)
                sys.exit()
        if self.al_model is not None:
            with self.track("Transcription Alignment"):
                try:
                    alignment_results = self.al_model(
                        audio_file, transcription=transcription_results
                    )
                except Exception as exp: # pylint: disable=W0718
                    print("!!Error encountered during Alignment:", exp)
                    sys.exit()
        if self.dia_model is not None and alignment_results is not None:
            with self.track("Speaker Diarization"):
                try:
                    diarization = self.dia_model(audio_file)
                except Exception as exp: # pylint: disable=W0718
                    print("!!Error encountered during Diarization:", exp)
                    sys.exit()
            with self.track("Processing Diarization"):
                try:
                    wsm = utils.map_speaker_to_words(
                        alignment_results["word_segments"],
                        utils.get_speaker_ts(diarization),
                    )
                    dia_results = utils.map_speaker_to_sentences(
                        wsm, utils.get_speaker_ts(diarization)
                    )
                except Exception as exp: # pylint: disable=W0718
                    print("!!Error encountered during Diarization processing:", exp)
                    sys.exit()
        with self.track("Processing results"):
            try:
                processed_results = self.process_results(
                    transcription_results, alignment_results, dia_results
                )
            except Exception as exp: # pylint: disable=W0718
                print("!!Error encountered during result processing:", exp)
                sys.exit()
        return processed_results
