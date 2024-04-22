# Copyright (c) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

FROM python:3.10

USER root

RUN apt-get update -y &&\
    apt-get install ffmpeg -y --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user

USER user

ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user install.sh ./
RUN chmod 755 install.sh &&\
    ./install.sh -im docker

COPY --chown=user ./asr ./asr
COPY --chown=user ./rest_api ./rest_api
COPY --chown=user ./serve_config.yaml ./
COPY --chown=user ./config ./config

CMD ["serve", "run", "serve_config.yaml"]
