# syntax = docker/dockerfile:1.3

FROM dock.biocad.ru/pythonbuilder:3.9-poetry1.1.13-cuda10.1

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/3bf863cc.pub

RUN apt-get update && apt-get install -y build-essential tmux

RUN poetry config virtualenvs.create false

COPY pyproject.toml poetry.lock poetry-env /EnQA/

RUN --mount=type=secret,id=poetry,target=/poetry-env . /poetry-env && \
	cd /EnQA && poetry install --no-interaction --no-ansi --no-dev

RUN pip install setuptools==59.5.0
