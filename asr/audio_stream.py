"""
Functions to stream audio chucks at every second. 
"""
import sys
import time
import librosa



def load_audio(fname, sampling_rate=16000):

    """
    loads the audio into buffer.
    """
    try:
        aud, _ = librosa.load(fname, sr=sampling_rate)
    except Exception: # pylint: disable=W0718
        print("!! Input audio format error !!")
        sys.exit()
    return aud


def load_audio_chunk(fname, beg, end, sampling_rate=16000):
    """
    loads the audio chuck
    """
    audio = load_audio(fname, sampling_rate)
    beg_s = int(beg * sampling_rate)
    end_s = int(end * sampling_rate)
    return audio[beg_s:end_s]


def stream_audio_buffer_from_file(
    audio_file_path, sampling_rate=16000, start_at=0, min_chunk=1.0
):
    """
    A generator function to stream audio every min_chunk sec. for
    real-time simulation from audio file
    """
    audio = load_audio(audio_file_path, sampling_rate)
    duration = len(audio) / sampling_rate
    beg = start_at
    start = time.time() - beg
    end = 0
    is_final = False

    while True:

        now = time.time() - start

        if now < end + min_chunk:
            time.sleep(min_chunk + end - now)

        end = time.time() - start
        if end >= duration:
            is_final = True

        yield load_audio_chunk(audio_file_path, beg, end, sampling_rate),\
            is_final

        beg = end
        now = time.time() - start
        if end >= duration:
            break
