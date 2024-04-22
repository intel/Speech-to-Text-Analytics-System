from typing import Optional
import json
import shutil
import uuid
from pathlib import Path
import os

# FAST API
from fastapi import (
    FastAPI,
    APIRouter,
    UploadFile,
    File,
    Form,
    HTTPException,
    status,
)
from pydantic import BaseModel

# Rest API
from rest_api.utils import get_app
from rest_api.config import FILE_UPLOAD_PATH

router = APIRouter()
app: FastAPI = get_app()


class FileUploadResponse(BaseModel):
    file_url: str


@router.post("/file-upload", status_code=status.HTTP_201_CREATED)
def upload_file(
    audio_file: UploadFile = File(...),
    meta: Optional[str] = Form("null"),  # type: ignore
) -> FileUploadResponse:
    """
    You can use this endpoint to upload an audio file for transcription.

    Note: files are removed immediately after being indexed. If you want
    to keep them, pass the `keep_files=true` parameter in the request payload.

    """

    meta_form = json.loads(meta) or {}  # type: ignore

    if not isinstance(meta_form, dict):
        raise HTTPException(
            status_code=500,
            detail=f"The meta field must be a dict or None, not {type(meta_form)}",  # noqa: E501
        )

    file_path = None
    try:
        directory_path = Path(FILE_UPLOAD_PATH)
        # breakpoint()
        directory_path.mkdir(parents=True, exist_ok=True)
        file_path = \
            directory_path / f"{uuid.uuid4().hex}_{audio_file.filename}"
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)

        meta_form["name"] = audio_file.filename
    finally:
        audio_file.file.close()
        buffer.close()

    base_name = os.path.split(file_path)[-1]
    return FileUploadResponse(file_url=str(base_name))
