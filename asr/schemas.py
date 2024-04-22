from __future__ import annotations
from typing import List
from pydantic import BaseModel, field_validator
from  pathlib import Path

VALID_MODELS_TR = [
    'distil-whisper/distil-large-v2',
    'distil-whisper/distil-large-v3',
    'openai/whisper-large-v2',
    'tiny'
]

VALID_BACKEND_TR = [
    'HFWhisper',
    'whisperx',
    'FasterWhisper'
]

VALID_COMPUTE_TYPE = [
    'fp32',
    'bf16',
    'int8'
]

VALID_COMPUTE_DIA_AL = [
    'fp32',
    'bf16',
    
]

VALID_BACKEND_DIA = [
    'pyannote_diarization'
]

VALID_MODEL_DIA = [
    'pyannote/speaker-diarization-3.1'
]

VALID_BACKEND_AL = [
    'whisperx_alignment'
]


class TranscriptionSchema(BaseModel):
    topology: str 
    compute_type: str 
    backend: str 
    batch_size: int 
    device: str

    @field_validator('topology')
    @classmethod
    def check_valid_topology(cls, v: str) -> str:
        if v not in VALID_MODELS_TR:
            raise ValueError(f'Invalid model id. valid values are {VALID_MODELS_TR}')
        return v

    @field_validator('backend')
    @classmethod
    def check_valid_backend(cls, v: str) -> str:
        if v not in VALID_BACKEND_TR:
            raise ValueError(f'Invalid backend id. valid values are {VALID_BACKEND_TR}')
        return v
    
    @field_validator('compute_type')
    @classmethod
    def check_valid_compute_type(cls, v: str) -> str:
        if v not in VALID_COMPUTE_TYPE:
            raise ValueError(f'Invalid compute type. valid values are {VALID_COMPUTE_TYPE}')
        return v

class AlignmentSchema(BaseModel): 
    compute_type: str 
    backend: str 
    language: str
    device: str

    @field_validator('backend')
    @classmethod
    def check_valid_backend(cls, v: str) -> str:
        if v not in VALID_BACKEND_AL:
            raise ValueError(f'Invalid backend id. valid values are {VALID_BACKEND_AL}')
        return v
    
    @field_validator('compute_type')
    @classmethod
    def check_valid_compute_type(cls, v: str) -> str:
        if v not in VALID_COMPUTE_TYPE:
            raise ValueError(f'Invalid compute type. valid values are {VALID_COMPUTE_DIA_AL}')
        return v

class DiarizationSchema(BaseModel):
    pipeline_config: str 
    compute_type: str 
    backend: str 
    device: str
    use_auth_token: bool

    @field_validator('pipeline_config')
    @classmethod
    def check_valid_topology(cls, v: str) -> str:
        if v not in VALID_MODEL_DIA:
            raise ValueError(f'Invalid model id. valid values are {VALID_MODEL_DIA}')
        return v

    @field_validator('backend')
    @classmethod
    def check_valid_backend(cls, v: str) -> str:
        if v not in VALID_BACKEND_DIA:
            raise ValueError(f'Invalid backend id. valid values are {VALID_BACKEND_DIA}')
        return v
    
    @field_validator('compute_type')
    @classmethod
    def check_valid_compute_type(cls, v: str) -> str:
        if v not in VALID_COMPUTE_TYPE:
            raise ValueError(f'Invalid compute type. valid values are {VALID_COMPUTE_DIA_AL}')
        return v

class ConfigSchema(BaseModel):
    name: str
    is_diarization: bool
    is_alignment: bool
    transcription: TranscriptionSchema
    alignment: AlignmentSchema
    diarization: DiarizationSchema
   

