"""
This module contains transcription related classes. 
"""

import logging

import numpy as np
import whisperx
import torch

# Transformers
from transformers import AutoModelForSpeechSeq2Seq, WhisperProcessor, pipeline

# faster whisper
from faster_whisper import WhisperModel

from asr.utils import ModelInferenceBuilder
from asr.whisper_online import OnlineASRProcessor


class WhisperXTranscription(ModelInferenceBuilder):
    """
    Transcription through whisperx transcription pipeline.
    """

    def load_model(self):
        """
        loads model
        """

        if self.config["compute_type"] == "fp32":
            compute_type = "float32"
        elif self.config["compute_type"] == "int8":
            compute_type = "int8"
        else:
            compute_type = "float32"
            logging.warning(
                "Invalid compute type for whisperx  \
                transcription. Using %s ",
                compute_type,
            )

        self.model = whisperx.load_model(
            self.config["topology"], self.config["device"],
            compute_type=compute_type
        )

    def optimize_model(self):
        """
        Optimize the model
        """

    def load_audio(self, audio_file):
        """
        Loads audio.
        """

        audio = whisperx.load_audio(audio_file)
        return audio

    def transcribe_audio(self, audio_file):
        """
        Given the audio_file, transcribes the audio.
        """

        audio = self.load_audio(audio_file)
        result = self.model.transcribe(audio, batch_size=self.config["batch_size"])
        return result

    def inference(self, audio_file, **kwargs):
        """
        Inference wrapper.
        """

        return self.transcribe_audio(audio_file, **kwargs)

    def post_process(self, results):
        """
        A post_process method for results
        """
        raise NotImplementedError

    def pre_process(self, inputs):
        """
        A pre_process method for inputs
        """
        raise NotImplementedError


class FasterWhisperTranscription(ModelInferenceBuilder):
    """
    Transcription through FasterWhisper transcription pipeline.
    """

    sep = ""

    def load_model(self):  # pylint: disable=C0116
        if self.config["compute_type"] == "fp32":
            compute_type = "float32"
        elif self.config["compute_type"] == "int8":
            compute_type = "int8"
        else:
            compute_type = "float32"
            logging.warning(
                "Invalid compute type for whisperx  \
                transcription. Using %s ",
                compute_type,
            )

        self.model = WhisperModel(
            self.config["topology"], self.config["device"],
            compute_type=compute_type
        )
        self.original_language = self.config["language"]

    def transcribe_audio(self, audio, init_prompt=""):  # pylint: disable=C0116
        segments, _ = self.model.transcribe(
            audio,
            language=self.original_language,
            initial_prompt=init_prompt,
            beam_size=5,
            word_timestamps=True,
            condition_on_previous_text=True,
        )
        return list(segments)

    def optimize_model(self):
        """
        Optimize the model
        """

    def ts_words(self, segments):  # pylint: disable=C0116
        o = []
        for segment in segments:
            for word in segment.words:
                w = word.word
                t = (word.start, word.end, w)
                o.append(t)
        return o

    def segments_end_ts(self, res):  # pylint: disable=C0116
        return [s.end for s in res]

    def inference(self, audio_file, **kwargs):  # pylint: disable=C0116
        return self.transcribe_audio(audio_file, **kwargs)

    def post_process(self, results):
        """
        A post_process method for results
        """
        raise NotImplementedError

    def pre_process(self, inputs):
        """
        A pre_process method for inputs
        """
        raise NotImplementedError


class HFWhisperTranscription(ModelInferenceBuilder):
    """
    Transcription through HuggingFace transcription pipeline.
    """

    def load_model(self):  # pylint: disable=C0116

        self.processor = WhisperProcessor.from_pretrained(self.config["topology"])
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.config["topology"],
            torch_dtype=torch.float32,  # loading model in float32
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )

    def tune_fp32(self):
        """
        Optimize the model
        """
        self.pipe = pipeline( # pylint: disable=W0201
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,  # pylint: disable=E1101
            feature_extractor=self.processor.feature_extractor,  # pylint: disable=E1101
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=self.config["batch_size"],
            torch_dtype=torch.float32,
            device=self.config["device"],
            return_timestamps=True,
        )  # pylint: disable=E1101

    def tune_int8_dynamic(self):  # pylint: disable=C0116
        logging.info("==> Tuning INT8 (Dynamic Quant) Model.")
        # import here. This is temporary
        import intel_extension_for_pytorch as ipex  # pylint: disable=C0415
        from intel_extension_for_pytorch.quantization import (  # pylint: disable=C0415
            prepare,
            convert,
        )  # pylint: disable=C0415

        qconfig = ipex.quantization.default_dynamic_qconfig
        prepared_model = prepare(
            self.model, qconfig, example_inputs=torch.randn((1, 80, 30000))
        )
        self.model = convert(prepared_model)
        self.pipe = pipeline( # pylint: disable=W0201
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,  # pylint: disable=E1101
            feature_extractor=self.processor.feature_extractor,  # pylint: disable=E1101
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=self.config["batch_size"],
            torch_dtype=torch.float32,
            device=self.config["device"],
            return_timestamps=True,
        )  # pylint: disable=E1101

    def tune_bf16(self):  # pylint: disable=C0116
        logging.info("==> Tuning BF16 Model.")
        # import here. This is temporary
        import intel_extension_for_pytorch as ipex  # pylint: disable=C0415

        self.model = ipex.optimize(self.model, dtype=torch.bfloat16)
        self.pipe = pipeline( # pylint: disable=W0201
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,  # pylint: disable=E1101
            feature_extractor=self.processor.feature_extractor,  # pylint: disable=E1101
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=self.config["batch_size"],
            torch_dtype=torch.float32,
            device=self.config["device"],
            return_timestamps=True,
        )  # pylint: disable=E1101

    def optimize_model(self):
        """
        Optimize the model
        """
        if self.config["compute_type"] == "fp32":
            self.tune_fp32()
            self.__transcribe_audio = self.transcribe_audio
        elif self.config["compute_type"] == "bf16":
            self.tune_bf16()
            self.__transcribe_audio = self.transcribe_audio_amp
        elif self.config["compute_type"] == "int8":
            self.tune_int8_dynamic()
            self.__transcribe_audio = self.transcribe_audio
        else:
            logging.exception(
                "Invalid value %s for Compute_type. Valid values are fp32, \
                bf16, int8 ",
                self.config["compute_type"],
            )

    def process_chunks(self, results):
        """
        Given results, processes chunks and return dictionary.
        """
        output = {}
        output["text"] = results["text"]
        output["segments"] = []
        for chunk in results["chunks"]:
            segment = {}
            ts = chunk["timestamp"]
            segment["start"] = ts[0]
            segment["end"] = ts[1] if ts[1] else ts[0]
            segment["text"] = chunk["text"]
            output["segments"].append(segment)
        return output

    def transcribe_audio_amp(self, audio_file):
        """
        AMP inference for the audio_file
        """

        with torch.no_grad(), torch.cpu.amp.autocast():
            results = self.pipe(audio_file)
        return self.process_chunks(results)

    def transcribe_audio(self, audio_file):
        """
        Runs the transcription on audio_file
        """

        results = self.pipe(audio_file)
        return self.process_chunks(results)

    def inference(self, audio_file, **kwargs):
        return self.__transcribe_audio(audio_file, **kwargs)

    def post_process(self, results):
        """
        A post_process method for results
        """
        raise NotImplementedError

    def pre_process(self, inputs):
        """
        A pre_process method for inputs
        """
        raise NotImplementedError


class WhisperXAlignment(ModelInferenceBuilder):
    """
    Aligns the transcription results.
    """

    def load_model(self):
        self.model, self.metadata = whisperx.load_align_model(
            language_code=self.config["language"], device=self.config["device"]
        )

    def optimize_model(self):
        """
        Optimize the model
        """
        if self.config["compute_type"] == "bf16":
            import intel_extension_for_pytorch as ipex  # pylint: disable=C0415

            compute_type = torch.bfloat16
            print(
                f"Running IPEX optimization with compute_type {self.config['compute_type']}"
            )
            ipex_feature_extractor = ipex.optimize(
                self.model.feature_extractor, dtype=compute_type
            )
            self.model.feature_extractor = ipex_feature_extractor
            print("Done IPEX optimization")

    def amp_align(self, audio_file, transcription):
        """
        Inference for bfloat16 (amp) dtype.
        """

        with torch.no_grad(), torch.cpu.amp.autocast():
            transcript_aligned = whisperx.align(
                transcription["segments"],
                self.model,
                self.metadata,
                audio_file,
                self.config["device"],
            )
        return transcript_aligned

    def align(self, audio_file, transcription):
        """
        Inference for fp32 and int8 dtype
        """

        transcript_aligned = whisperx.align(
            transcription["segments"],
            self.model,
            self.metadata,
            audio_file,
            self.config["device"],
        )
        return transcript_aligned

    def align_transcription(self, audio_file, transcription):
        """
        This function aligns the transcription.
        """

        if self.config["compute_type"] == "bf16":
            return self.amp_align(audio_file, transcription)
        return self.align(audio_file, transcription)

    def inference(self, audio_file, **kwargs):
        """
        Inference wrapper
        """

        return self.align_transcription(audio_file, **kwargs)

    def post_process(self, results):
        """
        A post_process method for results
        """
        raise NotImplementedError

    def pre_process(self, inputs):
        """
        A pre_process method for inputs
        """
        raise NotImplementedError


def tr_factory(model_key, config):
    """
    Factory method. Initializes the transcription
    model based on the config.
    """

    tr_registry = {
        "whisperx": WhisperXTranscription,
        "HFWhisper": HFWhisperTranscription,
        "whisperx_alignment": WhisperXAlignment,
        "FasterWhisper": FasterWhisperTranscription,
    }
    return tr_registry[model_key](config)


class WhisperLive: # pylint: disable=R0902
    """
    Wrapper class around whisper_online
    """

    def __init__(self, config):  # pylint: disable=C0116
        self.model = None
        self.online = None
        self.config = config
        self.init()

    def init(self):  # pylint: disable=C0116

        # load transcription model
        self.load_transcription_model()

        # init audio buffer & prompt
        self.audio_buffer = np.array([], dtype=np.float32)
        self.prompt_buffer = ""
        self.sampling_rate = 16000
        self.min_sample_length = 1 * self.sampling_rate
        self.init_prompt = ""

    def load_transcription_model(self):  # pylint: disable=C0116
        self.model = tr_factory(
            self.config["transcription"]["backend"], self.config["transcription"]
        )  # loads and wraps Whisper model
        self.online = OnlineASRProcessor(self.model)  # create processing object

    def stream_transcription(self, aud_buffer):
        """
        Given a generator aud_buffer, runs the live transcription.
        """
        complete_text = ""
        out = []
        out_len = 0
        for chunk, is_final in aud_buffer:

            if chunk is not None:
                out.append(chunk)
                out_len += len(chunk)

            if (is_final or out_len >= self.min_sample_length) and out_len > 0:
                a = np.concatenate(out)
                self.online.insert_audio_chunk(a)

            if out_len > self.min_sample_length:
                o = self.online.process_iter()
                logging.info("-----" * 10)
                complete_text = complete_text + o[2]
                logging.info(
                    " - %s" , complete_text
                )  
                if len(complete_text) > 0:
                    print(f"-> {complete_text}")
                logging.info("-----" * 10)
                out = []
                out_len = 0

            if is_final:
                o = self.online.finish()
                
                logging.info("-----" * 10)
                complete_text = complete_text + o[2]
                logging.info(
                    " - %s" , complete_text
                )  
               
                print(f"-> {complete_text}")
                logging.info("-----" * 10)
                self.online.init()
                out = []
                out_len = 0
