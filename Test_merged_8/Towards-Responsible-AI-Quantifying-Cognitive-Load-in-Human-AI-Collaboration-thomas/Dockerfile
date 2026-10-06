FROM python:3.10-slim

# Install system dependencies required by OpenCV and MediaPipe (libGLESv2, libGL, etc.)
RUN apt-get update && apt-get install -y \
    libgles2 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files
COPY . .

# Run the backend (Render injects the PORT environment variable automatically)
CMD ["sh", "-c", "uvicorn main:app --app-dir backend --host 0.0.0.0 --port ${PORT:-8002}"]
