"""
A main script to execute full pipeline for the given configuration file.
"""

# General
from pathlib import Path
import os
import sys
import argparse
import logging
import datetime
import yaml

# ASR
from asr import utils
from asr_pipeline import ASRPipeline

#Schema
from asr.schemas import ConfigSchema
from pydantic import ValidationError

def file_exists(raw_path):
    if not os.path.exists(raw_path):
        raise argparse.ArgumentTypeError('"{}" does not exist'.format(raw_path))
    return os.path.abspath(raw_path)

def list_of_strings(arg):
    return arg.split(',')

parser = argparse.ArgumentParser()
parser.add_argument(
    "--ground_truth",
    default=None,
    type=str,
    help="Ground truth transcription. If provided WER and Accuracy will be calculated",
)
parser.add_argument(
    "--output_dir",
    default="logs/",
    type=str,
    help="output directory where transcription results will be saved. Default is logs",
)
parser.add_argument(
    "--config_file",
    default="config/asr.yaml",
    type=file_exists,
    help="ASR pipeline configuration file. Default is config/asr.yaml",
)
parser.add_argument("--target_keywords",
                    type=list_of_strings,
                    default=[],
                    help="Target keywords to detect. Input as comma separated values.")
parser.add_argument("--input_audio", type=file_exists, required=True, help="Input audio file")
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
        _ = ConfigSchema.parse_obj(cfg)
        print("Input configuration is valid")
    except ValidationError as exp:
        print("\n!!Input configuration error:", exp)
        sys.exit()

    ext = os.path.splitext(cli_args.input_audio)[1][1:]
    choices = ['wav', 'mp3', 'flac']
    if ext not in choices:
        print ("File format not supported. Supported formats: {}".format(choices))
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


def main_func(config):  # pylint: disable=R0914
    """
    main function to execute pipeline.
    """

    asrpipe = ASRPipeline(config)
    tr_results, al_results, dia_results = asrpipe(args.input_audio)
    duration, rate, frames = utils.get_audio_info(args.input_audio)
    logging.info(
        "Audio info duration : %s framerate: %s frames: %s", duration, rate, frames
    )
    utils.write_to_file(tr_results, "segment_transcription.txt", config["log_dir"])

    if al_results is not None:
        utils.write_to_file(al_results, "aligned_transcription.txt", config["log_dir"])
    if dia_results is not None:
        utils.write_to_file(
            dia_results, "diarization_transcription.txt", config["log_dir"]
        )

        targets_detected = utils.get_keyword_spotting(dia_results, args.target_keywords)
        spk_time_sum = utils.get_talk_time(dia_results, duration)
        speaker_switches = utils.get_switches_per_conversation(dia_results)
        lm_dict = utils.get_longest_monologue(dia_results)
        spk_words_sum = utils.get_words_per_min(dia_results, spk_time_sum)

        with open(
            os.path.join(config["log_dir"], "analytics.txt"), "w", encoding="utf-8"
        ) as fp:
            fp.write(f"Audio file : {args.input_audio} \n\n")
            fp.write("Longest monologue \n")
            fp.write("-------------------------\n")
            fp.write(
                f"""{lm_dict['lm_spk']}
                Duration: {str(round(lm_dict['lm_time'],2)) + 's'}
                Start_time - End_time: {lm_dict['mono_st']} - {lm_dict['mono_et']}
                Text: {lm_dict['mono_text']} \n"""
            )

            fp.write("\nSpeaker Distribution \n")
            fp.write("-------------------------\n")
            for key, value in spk_time_sum.items():
                fp.write(f"{key} : {value}s\n")

            fp.write("\nWords per min \n")
            fp.write("-------------------------\n")
            for key, value in spk_words_sum.items():
                fp.write(f"{key} : {value}\n")

            fp.write(f"\n\nTarget words detected : {targets_detected} \n")

            fp.write(f"\nSwitches in the conversation : {speaker_switches}\n")
        fp.close()


if __name__ == "__main__":

    config_ = prepare_config(args)
    main_func(config_)
