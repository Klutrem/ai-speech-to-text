FROM python:3.10-slim

ARG PIP_REQ_FILE=requirements.txt

RUN apt update && apt install git ffmpeg -y

WORKDIR /app

COPY ${PIP_REQ_FILE} ${PIP_REQ_FILE}
COPY src/ src/

COPY trained_model/ trained_model/

RUN pip3 install -r ${PIP_REQ_FILE}

RUN whisper --model small --language ru dummy.wav; exit 0

EXPOSE 8000

CMD [ "fastapi", "run", "src/main.py" ]