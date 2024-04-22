import os
#from pathlib import Path

# FILE_UPLOAD_PATH = os.getenv(
#     "FILE_UPLOAD_PATH",
#     str((Path(__file__).parent.parent / "data" / "file-upload/").absolute()),
# )
FILE_UPLOAD_PATH = 'data/file-upload/'
MODEL_CONFIG_FILE = "./config/asr_speaker_aware.yaml"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
ROOT_PATH = os.getenv("ROOT_PATH", "")
VALID_AUDIO_EXT = ['.wav', '.mp3', '.flac']
