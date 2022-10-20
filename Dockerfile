# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM tiangolo/uvicorn-gunicorn:python3.9

WORKDIR app

COPY requirements.txt .
COPY main.py .
COPY models models
COPY sample_data sample_data
COPY services services
COPY training_methods training_methods
COPY utils.py .

# Install dependencies.
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
