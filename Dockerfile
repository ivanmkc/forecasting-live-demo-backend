# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM tiangolo/uvicorn-gunicorn:python3.7

WORKDIR app

COPY ./requirements.txt .
COPY ./main.py .

# Install dependencies.
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
