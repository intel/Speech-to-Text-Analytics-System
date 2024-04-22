""" This is a test module to test pipeline with various configuration settings"""

# General
import os
from pathlib import Path
import pytest

# ASR
from asr.utils import load_config_file, get_audio_info
from asr_pipeline import ASRPipeline

TEST_DIR = os.path.join(str(Path(__file__).parent), "configs")
DATA_DIR = os.path.join(str(Path(__file__).parent), "test_samples")
SAMPLE_FILE = "test.wav"


TRANSCRIPTION_TEST_DATA = [
    [
        os.path.join(DATA_DIR, SAMPLE_FILE),
        os.path.join(TEST_DIR, "asr_transcription_test.yaml"),
        "fp32",
        "distil-whisper/distil-large-v2",
    ],
    [
        os.path.join(DATA_DIR, SAMPLE_FILE),
        os.path.join(TEST_DIR, "asr_transcription_test.yaml"),
        "bf16",
        "distil-whisper/distil-large-v2",
    ],
    [
        os.path.join(DATA_DIR, SAMPLE_FILE),
        os.path.join(TEST_DIR, "asr_transcription_test.yaml"),
        "int8",
        "distil-whisper/distil-large-v2",
    ],
    [
        os.path.join(DATA_DIR, SAMPLE_FILE),
        os.path.join(TEST_DIR, "asr_transcription_test.yaml"),
        "fp32",
        "openai/whisper-large-v2",
    ],
    [
        os.path.join(DATA_DIR, SAMPLE_FILE),
        os.path.join(TEST_DIR, "asr_transcription_test.yaml"),
        "bf16",
        "openai/whisper-large-v2",
    ],
    [
        os.path.join(DATA_DIR, SAMPLE_FILE),
        os.path.join(TEST_DIR, "asr_transcription_test.yaml"),
        "int8",
        "openai/whisper-large-v2",
    ],
]


@pytest.mark.pytorch
@pytest.mark.parametrize(
    "audio_file,config_file",
    [
        [
            os.path.join(DATA_DIR, SAMPLE_FILE),
            os.path.join(TEST_DIR, "asr_pipeline.yaml"),
        ],
    ],
)
def test_pipeline_fp32(audio_file, config_file):
    """
    Test the pipeline
    """

    config = load_config_file(config_file)
    asrpipe = ASRPipeline(config)
    tr_results, al_results, dia_results = asrpipe(audio_file)
    duration, _, frames = get_audio_info(audio_file)

    assert 42 < duration < 42.02
    assert frames == 672160
    assert tr_results is not None
    assert al_results is not None
    assert dia_results is not None
    assert len(tr_results) == 9
    assert len(al_results) == 10
    assert len(dia_results) == 4
    assert "rabbit" in tr_results[0]["text"]
    assert "dialogues" in tr_results[-1]["text"]


@pytest.mark.pytorch
@pytest.mark.parametrize(
    "audio_file,config_file,compute_type,topology",
    TRANSCRIPTION_TEST_DATA,
)
def test_transcription_distill_whisper(audio_file, config_file, compute_type, topology):
    """
    Test the transcription and alignment
    """

    config = load_config_file(config_file)
    config["transcription"]["compute_type"] = compute_type
    config["transcription"]["topology"] = topology
    asrpipe = ASRPipeline(config)
    tr_results, al_results, _ = asrpipe(audio_file)
    duration, _, frames = get_audio_info(audio_file)

    assert 42 < duration < 42.02
    assert frames == 672160
    assert tr_results is not None
    assert al_results is not None
    assert "rabbit" in tr_results[0]["text"]
    assert "dialogues" in tr_results[-1]["text"]


@pytest.mark.pytorch
@pytest.mark.parametrize(
    "audio_file,config_file,compute_type,topology",
    [
        [
            os.path.join(DATA_DIR, SAMPLE_FILE),
            os.path.join(TEST_DIR, "asr_transcription_test.yaml"),
            "fp32",
            "distil-whisper/distil-large-v2",
        ],
        [
            os.path.join(DATA_DIR, SAMPLE_FILE),
            os.path.join(TEST_DIR, "asr_transcription_test.yaml"),
            "bf16",
            "distil-whisper/distil-large-v2",
        ],
    ],
)
def test_alignment_with_dtypes(audio_file, config_file, compute_type, topology):
    """
    Test the transcription and alignment
    """

    config = load_config_file(config_file)
    config["alignment"]["compute_type"] = compute_type
    config["transcription"]["topology"] = topology
    asrpipe = ASRPipeline(config)
    tr_results, al_results, _ = asrpipe(audio_file)
    duration, _, frames = get_audio_info(audio_file)

    assert 42 < duration < 42.02
    assert frames == 672160
    assert tr_results is not None
    assert al_results is not None
    assert len(tr_results) == 9
    assert len(al_results) == 10
    assert "rabbit" in tr_results[0]["text"]
    assert "dialogues" in tr_results[-1]["text"]


@pytest.mark.pytorch
@pytest.mark.parametrize(
    "audio_file,config_file,compute_type,topology",
    [
        [
            os.path.join(DATA_DIR, SAMPLE_FILE),
            os.path.join(TEST_DIR, "asr_pipeline.yaml"),
            "fp32",
            "distil-whisper/distil-large-v2",
        ],
        [
            os.path.join(DATA_DIR, SAMPLE_FILE),
            os.path.join(TEST_DIR, "asr_pipeline.yaml"),
            "bf16",
            "distil-whisper/distil-large-v2",
        ],
    ],
)
def test_diarization_with_dtypes(audio_file, config_file, compute_type, topology):
    """
    Test the transcription and alignment
    """

    config = load_config_file(config_file)
    config["diarization"]["compute_type"] = compute_type
    config["transcription"]["topology"] = topology
    asrpipe = ASRPipeline(config)
    tr_results, al_results, dia_results = asrpipe(audio_file)
    duration, _, frames = get_audio_info(audio_file)

    assert 42 < duration < 42.02
    assert frames == 672160
    assert tr_results is not None
    assert al_results is not None
    assert len(tr_results) == 9
    assert len(al_results) == 10
    assert len(dia_results) == 4
    assert "rabbit" in tr_results[0]["text"]
    assert "dialogues" in tr_results[-1]["text"]
