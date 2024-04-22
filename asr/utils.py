"""
Utility functions.
"""

import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter_ns
import logging
import datetime
import librosa
import yaml

SAMPLING_RATE = 16000.0

def map_speaker_to_words(word_timestamps, speaker_timestamps):

    """
    This function maps speakers to words based on speaker and word timestamps.
    """

    word_speaker_mapping = []
    spk_idx = 0
    cur_spk = speaker_timestamps[spk_idx][-1]

    for word_ts in word_timestamps:
        if "start" not in word_ts.keys():
            # sometimes word_dict does not contain start and stop keys,
            # so continue... may be need to check in transcription.
            continue
        ws, we, word = (int(word_ts["start"] * 1000),
            int(word_ts["end"] * 1000),
                 word_ts["word"],
            )
        end = speaker_timestamps[spk_idx][1]
        while ws >= end:
            spk_idx = spk_idx + 1
            spk_idx = min(spk_idx, len(speaker_timestamps)- 1)
            cur_spk = speaker_timestamps[spk_idx][-1]
            end = speaker_timestamps[spk_idx][1]
            if spk_idx == len(speaker_timestamps) - 1:
                end = we
        word_speaker_mapping.append (
                        {
                         "word": word, 
                         "st": ws, 
                         "et": we, 
                         "speaker": cur_spk
                        }
                    )
    return word_speaker_mapping

def map_speaker_to_sentences(word_speaker_mappings, speaker_timestamps):
    """
    This function maps speaker to sentences based on speaker and word
    speaker mapping
    """
    sentences = []

    s, e, speaker = speaker_timestamps[0]
    sentence = {'speaker': str(speaker), 'st': s, 'et': e, 'text': ""}
    prev_speaker = speaker

    for wsm in word_speaker_mappings:
        
        if wsm["speaker"] != prev_speaker:
            sentences.append(sentence)
            # Re-init, the new sentence
            sentence = {'speaker': str(wsm['speaker']), 
                        'st': wsm['st'], 'et': wsm['et'], 'text': ''}

        prev_speaker = wsm["speaker"]
        sentence["et"] = wsm["et"]
        sentence["text"] += wsm["word"] + " "
        
    sentences.append(sentence)
    return sentences


def get_speaker_ts(diarization):
    """
    Gets speaker time stamps
    """

    out_segs = list(diarization.itertracks())
    speaker_ts = []
    for seg in out_segs:
        start, end = seg[0]
        spk = diarization[seg[0], seg[1]]
        s = int(start * 1000)
        e = int(end * 1000)
        speaker_ts.append([s, e, spk])
    return speaker_ts


def save_speaker_aware_transcript(sentences_speaker_mapping, output_file):
    """
    saves speaker aware transcription
    """

    with open(output_file, "w", encoding='utf-8') as fp:
        for sentence_dict in sentences_speaker_mapping:
            sp = sentence_dict["speaker"]
            text = sentence_dict["text"]
            fp.write(f"\n{sp}: {text}")
        fp.close()


def get_speaker_aware_transcript(sentences_speaker_mapping):
    """
    returns speaker aware transcription for the give setence_speaker_mapping
    """

    spk_transcription = []
    for sentence_dict in sentences_speaker_mapping:
        sp = sentence_dict["speaker"]
        text = sentence_dict["text"]
        spk_transcription.append((sp, text))


def load_audio_rosa(fname, sampling_rate=SAMPLING_RATE):
    """
    loads audio to buffer for the give input file.
    """

    aud, _ = librosa.load(fname, sr=sampling_rate)
    return aud


def get_audio_info(audio_file):
    """
    Return audio meta information.
    """

    aud = load_audio_rosa(audio_file, SAMPLING_RATE)
    duration = len(aud) / SAMPLING_RATE
    rate = SAMPLING_RATE
    frames = len(aud)
    return duration, rate, frames


def get_talk_time(diarization_results, total_duration):
    """
    Given the spaker aware transcription in diarization_reults,
    returnst the speaker talk time distribution.
    """

    total_talk_time = 0.0
    spk_time = {
        "Speaker": [],
        "Talk Time": [],
    }
    spk_time_sum = {}

    for segment in diarization_results:
        spk_utter_time = segment["end_time"] - segment["start_time"]
        total_talk_time += spk_utter_time
        spk_time["Speaker"].append(segment["speaker"])
        spk_time["Talk Time"].append(round(spk_utter_time, 2))
    silence_time = total_duration - total_talk_time
    spk_time["Speaker"].append("Pause")
    spk_time["Talk Time"].append(round(silence_time, 2))

    for spk, tt in zip(spk_time["Speaker"], spk_time["Talk Time"]):
        if spk in spk_time_sum:
            spk_time_sum[spk] += tt
        else:
            spk_time_sum[spk] = tt

    return spk_time_sum


def get_switches_per_conversation(diarization_results):
    """
    Calucates the swithces in conversation.
    """

    index = 0
    prev_spk = None
    switch = 0
    for segment in diarization_results:
        if index > 0:
            if prev_spk != segment["speaker"]:
                switch += 1
        prev_spk = segment["speaker"]
        index += 1
    return switch


def write_to_file(results, filename, output_dir):
    """
    Writes results to the file.
    """

    if not os.path.exists(os.path.join(output_dir, "raw_transcription.txt")):
        text = ""
        for segment in results:
            text = text + segment["text"]
        with open(os.path.join(output_dir, "raw_transcription.txt"),
                  "w", encoding='utf-8') as fp:
            fp.write(text)
        fp.close()
    with open(os.path.join(output_dir, filename), "w", encoding='utf-8') as fp:
        for segment in results:
            sp = segment["speaker"]
            text = segment["text"]
            st = str(datetime.timedelta(seconds=segment["start_time"]))[:7]
            et = str(datetime.timedelta(seconds=segment["end_time"]))[:7]
            fp.write(f"{sp} : {st} - {et} : {text}\n")
        fp.close()


def get_keyword_spotting(diarization_results, target_word):
    """
    Searches the given keywords in the diarization_results.
    """

    target_word = [x.lower() for x in target_word]
    targets_detected = []
    text = ""

    for segment in diarization_results:
        text = text + segment["text"]
    text_list = list(text.split(" "))
    text_list = [
        word.lower()
        .replace(".", "")
        .replace(",", "")
        .replace(":", "")
        .replace("!", "")
        .replace("?", "")
        .replace("'", "")
        .replace(" ", "")
        for word in text_list
    ]

    numeral_dict = {
        "1": "one",
        "2": "two",
        "3": "three",
        "4": "four",
        "5": "five",
        "6": "six",
        "7": "seven",
        "8": "eight",
        "9": "nine",
        "0": "zero",
    }

    for i, j in enumerate(text_list):
        if j in numeral_dict:
            text_list[i] = numeral_dict[j]

    if any(alert in target_word for alert in text_list):
        for target_ in target_word:
            for text_ in text_list:
                if text_ == target_ and target_ not in targets_detected:
                    targets_detected.append(target_)

    return targets_detected


def get_longest_monologue(diarization_results):
    """
    Calculates the longest monologue in the speaker
    aware transcription.
    """

    lm_dict = {
        "lm_time": 0,
    }

    for segment in diarization_results:
        sp = segment["speaker"]
        text = segment["text"]
        et = segment["end_time"]
        st = segment["start_time"]
        dur = et - st
        if dur > lm_dict["lm_time"]:
            lm_dict["lm_time"] = dur
            lm_dict["lm_spk"] = sp
            lm_dict["mono_text"] = text
            lm_dict["mono_st"] = st
            lm_dict["mono_et"] = et

    return lm_dict


def get_words_per_min(diarization_results, spk_time):
    """
    Calulates the talking speed in words/min for each speaker.
    """

    speaker_text = {}

    for segment in diarization_results:
        sp = segment["speaker"]
        text = segment["text"]
        if sp not in speaker_text.keys(): # pylint: disable=C0201
            speaker_text[sp] = ""
        speaker_text[sp] = speaker_text[sp] + text + " "
    spk_words_sum = {}

    for spk in spk_time:
        if spk in speaker_text.keys(): # pylint: disable=C0201
            text = speaker_text[spk]
            total_words = len(text.strip().split(" "))
            wpm = round((60 * total_words) / spk_time[spk], 2)
            spk_words_sum[spk] = wpm
        if spk == "Pause":
            spk_words_sum[spk] = "NA"

    return spk_words_sum


class ModelInferenceBuilder:
    """
    An abstract class for building the infernece pipeline.
    """

    def __init__(self, config):
        self.config = config
        self.model = None
        self.load_model()
        self.optimize_model()

    def load_model(self):

        """
        A load model method should be implemented by the derived class
        """

        raise NotImplementedError

    def optimize_model(self):
        """
        optimize_model method should be implemented by the derived class
        """
        raise NotImplementedError

    def inference(self, audio_file, **kwargs):
        """
        An inference method should be implemented by the derived class
        """
        raise NotImplementedError

    def post_process(self, results):
        """
        A post_processl method should be implemented by the derived class
        """
        raise NotImplementedError

    def pre_process(self, inputs):
        """
        A pre_processl method should be implemented by the derived class
        """
        raise NotImplementedError

    def __call__(self, inputs, **kwargs):
        """
        A callable , calls infernece method on the given input.
        """
        return self.inference(inputs, **kwargs)


SEC_TO_NS_SCALE = 1000000000


@dataclass
class Benchmark:
    """
    Benchmarking class and functions.
    """

    summary_msg: str = field(default_factory=str)

    @contextmanager
    def track(self, step):

        """
        Context manager to track the latency
        """

        start = perf_counter_ns()
        yield
        ns = perf_counter_ns() - start
        msg = f"==>'{step}' took {ns / SEC_TO_NS_SCALE:.3f}s ({ns:,}ns)"
        print(msg)
        logging.info(msg)
        self.summary_msg += msg + "\n"

    def summary(self):
        """
        prints summary of benchmark.
        """

        print(f"\n{'#' * 30}\nBenchmark Summary:\n{'#' * 30}\n\n{self.summary_msg}")


def process_transcription_results(transcription_results):
    """
    process the transcription results and return in the dictionary format.
    """

    output_transcription = []
    for segment in transcription_results["segments"]:
        transcription_dict = {}
        transcription_dict["speaker"] = "Unknown"
        transcription_dict["start_time"] = segment["start"]
        transcription_dict["end_time"] = segment["end"]
        transcription_dict["text"] = segment["text"]
        output_transcription.append(transcription_dict)
    output = {"segments": output_transcription}
    return output


def process_speaker_aware_transcription(diarization_results):
    """
    process the process_speaker_aware_transcription results and return in the dictionary format.
    """

    output_transcription = []
    for segment in diarization_results:
        dia_dict = {}
        dia_dict["speaker"] = segment["speaker"]
        dia_dict["start_time"] = segment["st"] / 1000
        dia_dict["end_time"] = segment["et"] / 1000
        dia_dict["text"] = segment["text"]
        output_transcription.append(dia_dict)
    output = {"segments": output_transcription}
    return output


def load_config_file(config_file):
    """
    loads the configuration yaml file and returns the dictionary.
    """

    logging.info("Loading Model Configuration File.")
    with open(config_file, encoding="utf-8") as fp:
        cfg = yaml.safe_load(fp)
    fp.close()
    logging.info("-" * 50)
    logging.info(cfg)
    return cfg
