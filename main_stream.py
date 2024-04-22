"""
Give a input audio file, this script simulates the live transcription
"""

from pathlib import Path
import os
import sys
import argparse
import logging
import datetime
import yaml

# ASR
from asr.transcription import WhisperLive
from asr.audio_stream import stream_audio_buffer_from_file

#Schema
from asr.schemas import ConfigSchema
from pydantic import ValidationError

def file_exists(raw_path):
    if not os.path.exists(raw_path):
        raise argparse.ArgumentTypeError('"{}" does not exist'.format(raw_path))
    return os.path.abspath(raw_path)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--ground_truth",
    default=None,
    type=str,
    help="Ground truth transcription.\
    If provided WER and Accuracy will be calculated",
)
parser.add_argument(
    "--output_dir",
    default="logs/",
    type=str,
    help="output directory where transcription results\
    will be saved. Default is logs",
)
parser.add_argument(
    "--config_file",
    default="config/asr.yaml",
    type=file_exists,
    help="ASR pipeline configuration file. Default is config/asr.yaml",
)
parser.add_argument("--input_audio", type=file_exists, required=True,
                    help="Input audio file")
args = parser.parse_args()


def prepare_config(cli_args):
    """
    Reads configuration file and return dictionary. Also creates
    a logging directory.
    """

    config_file_basename = os.path.basename(cli_args.config_file)
    config_file_path = os.path.join('./config', config_file_basename)
    with open(config_file_path, encoding="utf8") as fp:
        cfg = yaml.safe_load(fp)
    fp.close()

    try:
        cs = ConfigSchema.parse_obj(cfg)
        print("Input configuration is valid")
    except ValidationError as exp:
        print("\n!!Input configuration error:", exp)
        sys.exit()

    log_base_dir = os.path.basename(cli_args.output_dir)
    log_dir = os.path.join(
        os.path.join('output/',log_base_dir), datetime.datetime.now().strftime("%m%d%H%M%S")
    )
    directory_path = Path(log_dir)
    directory_path.mkdir(parents=True, exist_ok=True)
    cfg["log_dir"] = log_dir
    logging.basicConfig(
        filename=os.path.join(log_dir, "logs.txt"),
        level=logging.INFO,
        filemode="w",
        format="%(funcName)-10s: %(message)s",
    )
    logging.info("-" * 50)
    logging.info(cfg)
    print(f"log directory is {directory_path}")
    return cfg


def main_stream_transcription(config):
    """
    This function simulates the live transcription
    """

    whisperlive = WhisperLive(config)
    stream = stream_audio_buffer_from_file(args.input_audio)
    whisperlive.stream_transcription(stream)


if __name__ == "__main__":

    config_ = prepare_config(args)
    main_stream_transcription(config_)
