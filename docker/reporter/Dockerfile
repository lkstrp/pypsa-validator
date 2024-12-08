FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    git

WORKDIR /workdir

COPY ./reporter ./reporter
COPY pyproject.toml ./  

RUN pip install .
CMD ["python", "-m", "reporter"]
