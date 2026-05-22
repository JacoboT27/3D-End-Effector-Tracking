FROM python:3.11-slim

# system dependencies for MuJoCo rendering and build tools
RUN apt-get update && apt-get install -y \
    libgl1 \
    libosmesa6 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy project
COPY . .

# headless rendering default
ENV MUJOCO_GL=osmesa
ENV DISPLAY=:0

CMD ["python", "agent/train.py", "--config", "configs/default.yaml"]