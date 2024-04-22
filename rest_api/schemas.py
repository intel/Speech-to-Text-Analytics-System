from __future__ import annotations
from typing import List
from pydantic import BaseModel, field_validator
from  pathlib import Path
from rest_api.config import VALID_AUDIO_EXT

class Segment(BaseModel):
    speaker: str
    start_time: float
    end_time: float
    text: str


class TranscriptionResponse(BaseModel):
    segments: List[Segment]


class BaseDiarizationResponse(BaseModel):
    rttm: str
    uri: str


class DiarizationResponse(BaseDiarizationResponse):
    segments: List[Segment]


class AudioFilePathSchema(BaseModel):
    audio_path: str

    @field_validator('audio_path')
    @classmethod
    def audio_path_name_suffix_valid(cls, v: str) -> str:
        if Path(v).suffix not in VALID_AUDIO_EXT:
            raise ValueError(f'Invalid Audio type. Supported types are {VALID_AUDIO_EXT}')
        return v
