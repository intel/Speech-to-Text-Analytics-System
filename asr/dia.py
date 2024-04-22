"""
Diarization inference class.
"""


# Torch
import os
import logging
import torch
from dotenv import load_dotenv

# pyannote
from pyannote.audio import Pipeline
from asr.utils import ModelInferenceBuilder

load_dotenv()


class PyannoteDiarization(ModelInferenceBuilder):
    """
    PyannoteDiarization is a diarization inference class. It loads
    model from the given configuration. 
    """

    def load_model(self):

        """
        loads model, either from the specified path or downloads
        directly from HF hub
        """

        if self.config["use_auth_token"] is True:
            hf_token = os.environ.get("HF_ACCESS_TOKEN", None)
            if hf_token is None:
                logging.error(
                    "Failed to load access token, \
                    please define access token HF_ACCESS_TOKEN .env file. "
                )
            self.model = Pipeline.from_pretrained(
                self.config["pipeline_config"], use_auth_token=hf_token
            )
        else:
            self.model = Pipeline.from_pretrained(
                self.config["pipeline_config"],
                use_auth_token=self.config["use_auth_token"],
            )

        if self.model is not None:
            self.model.to(torch.device(self.config["device"]))
        else:
            logging.error("Failed to Load Diarization Model.")

    def optimize_model(self):
        """
        Runs BF16 model optimizations using IPEX.
        """

        if self.config["compute_type"] == "bf16":
            import intel_extension_for_pytorch as ipex # pylint: disable=C0415

            print(f"running IPEX optimization {self.config['compute_type']}")
            ipex_seg_model = ipex.optimize(
                self.model._segmentation.model, dtype=torch.bfloat16 # pylint: disable=W0212
            )
            # self.model._embedding.classifier_.eval()
            # ipex_emb_model = ipex.optimize(self.model._embedding.classifier_,dtype=torch.bfloat16)
            # self.model._embedding.classifier_ = ipex_emb_model
            self.model._segmentation.model = ipex_seg_model # pylint: disable=W0212
            print("done ! running IPEX optimization")

    def amp_apply_pipeline(
        self,
        audio_file,
        num_speakers: int = None,
        min_speakers: int = None,
        max_speakers: int = None,
        return_embeddings: bool = False,
    ): # pylint: disable=R0913

        """
        applies BF16 the pipeline to the given input audio.
        """
        with torch.no_grad(), torch.cpu.amp.autocast():
            diarization = self.model(
                audio_file,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                return_embeddings=return_embeddings,
            )

        return diarization

    def apply_pipeline(
        self,
        audio_file,
        num_speakers: int = None,
        min_speakers: int = None,
        max_speakers: int = None,
        return_embeddings: bool = False,
    ): # pylint: disable=R0913

        """
        applies BF16 the pipeline to the given input audio.
        """
        diarization = self.model(
            audio_file,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            return_embeddings=return_embeddings,
        )
        return diarization

    def inference(self, audio_file, **kwargs):
        if self.config["compute_type"] == "bf16":
            return self.amp_apply_pipeline(audio_file, **kwargs)
        return self.apply_pipeline(audio_file, **kwargs)

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

def dia_factory(model_key, config):

    """
    A factory method to load a model based on the give key.
    """

    tr_factory = {"pyannote_diarization": PyannoteDiarization}
    return tr_factory[model_key](config)
