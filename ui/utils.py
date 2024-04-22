import logging
import requests
import librosa
from time import sleep
from ui.config import FILE_UPLOAD, STATUS, API_ENDPOINT, SAMPLING_RATE

REQ_TIMEOUT = 3600

def server_is_ready():
    """
    Used to show the "Haystack is loading..." message
    """
    response = None
    url = f"{API_ENDPOINT}/{STATUS}"
    try:
        response = requests.get(url,timeout=REQ_TIMEOUT)
        if response.status_code < 400:
            return True, response.json()
    except Exception as e: # pylint: disable=W0718
        print(e)
        logging.error("Error establishing connection to the server")
        sleep(1)  # To avoid spamming a non-existing endpoint at startup
    return False, response


def upload_audio_file(audio_file):
    url = f"{API_ENDPOINT}/{FILE_UPLOAD}"
    response = None
    try:
        files = [("audio_file", audio_file)]
        response = requests.post(url, files=files,timeout=REQ_TIMEOUT)
        print(response)
        if response.status_code < 400:
            return response.json()
    except Exception as e: # pylint: disable=W0718
        print(e)
        logging.error("Error uploading file")
    return response


def process_audio(audio_file, is_alignment=False, is_speaker_aware=False):
    """
    Used to show the "Haystack is loading..." message
    """
    params = {}
    params["audio_path"] = audio_file
    if is_speaker_aware:
        url = f"{API_ENDPOINT}/speaker_aware_transcription"
    else:
        url = f"{API_ENDPOINT}/transcription"
        params["align_transcription"] = False
        if is_alignment:
            params["align_transcription"] = True
    print(params)
    response = None
    try:
        response = requests.get(url, params=params,timeout=REQ_TIMEOUT)
        if response.status_code < 400:
            return response.json()
    except Exception as e: # pylint: disable=W0718
        print(e)
        logging.error("Error in processing audio %s", url)
    return response


def get_speech_analysis(transcription_results,audio_duration):
    """
    Downstream analytics of the conversation. transcription_results should contain speaker information.
    """
    params = {}
    params['audio_duration'] = audio_duration
    url = f"{API_ENDPOINT}/speech_analysis"
    response = None
    try:
        response = requests.post(url, params=params,json=transcription_results,timeout=REQ_TIMEOUT)
        if response.status_code < 400:
            return response.json()
    except Exception as e: # pylint: disable=W0718
        print(e)
        logging.error("Error in processing audio %s", url)
    return response


def load_audio_rosa(fname,sampling_rate=SAMPLING_RATE):
    aud, _ = librosa.load(fname, sr=sampling_rate)
    return aud

def get_audio_info(audio_file):
    aud = load_audio_rosa(audio_file, SAMPLING_RATE)
    audio_info = {}
    audio_info['duration'] = len(aud)/SAMPLING_RATE
    audio_info['rate'] = SAMPLING_RATE 
    audio_info['frames'] = len(aud)
    return audio_info