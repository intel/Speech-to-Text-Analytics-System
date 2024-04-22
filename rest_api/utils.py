import logging
import yaml

# Res API
from rest_api.controller.errors.http_error import http_error_handler

# Fast API
from fastapi import FastAPI, HTTPException, APIRouter
from fastapi.routing import APIRoute
from starlette.middleware.cors import CORSMiddleware

app = None


def get_app() -> FastAPI:
    """
    Initializes the App object and creates the global pipelines as possible.
    """

    global app  # pylint: disable=global-statement
    if app:
        return app

    from rest_api.config import ROOT_PATH

    app = FastAPI(title="ASR Pipeline", debug=True, version="1.0",
                  root_path=ROOT_PATH)

    # Creates the router for the API calls
    from rest_api.controller import file_upload

    router = APIRouter()
    router.include_router(file_upload.router, tags=["file-upload"])

    # This middleware enables allow all cross-domain requests to
    # the API from a browser. For production deployments, it could
    # be made more restrictive.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_exception_handler(HTTPException, http_error_handler)
    app.include_router(router)

    # Simplify operation IDs so that generated API clients have
    # simpler function names (see https://fastapi.tiangolo.com/advanced/
    # path-operation-advanced-configuration
    # /#using-the-path-operation-function-name-as-the-operationid).
    # The operation IDs will be the same as the route names
    # (i.e. the python method names of the endpoints). Should be
    # called only after all routes have been added.
    for route in app.routes:
        if isinstance(route, APIRoute):
            route.operation_id = route.name

    return app


def load_config_file(config_file):
    logging.info("Loading Model Configuration File.")
    with open(config_file, encoding='utf-8') as fp:
        cfg = yaml.safe_load(fp)
    fp.close()
    logging.info("-" * 50)
    logging.info(cfg)
    return cfg
