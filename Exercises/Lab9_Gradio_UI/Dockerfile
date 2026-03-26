# Dockerfile for Lab 9: Gradio UI
FROM python:3.12-slim-bookworm
WORKDIR /app
COPY requirements.txt requirements.txt
# CPU-only PyTorch avoids multi-GB CUDA/NVIDIA wheels (default pip torch on Linux aarch64 pulls cuda-toolkit).
RUN pip install --upgrade pip && \
    pip install --no-cache-dir "gradio>=3.0" "transformers>=4.49.0" && \
    pip install --no-cache-dir "torch>=2.6.0" --index-url https://download.pytorch.org/whl/cpu
COPY app.py app.py
EXPOSE 9000
CMD ["python", "app.py"]
