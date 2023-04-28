
FROM gcr.io/deeplearning-platform-release/tf2-gpu.2-8

WORKDIR /

# Installs hypertune library
RUN pip install cloudml-hypertune

# Copies the trainer code to the Docker image.
COPY task.py .

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python", "-m", "task"]
