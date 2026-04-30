FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Tokyo \
    NVIDIA_DISABLE_REQUIRE=1

RUN apt-get update && apt-get install -y \
        wget curl git tmux \
        ffmpeg libsm6 libxext6 libgl1-mesa-glx \
        build-essential ca-certificates \
        pkg-config \
        libavformat-dev libavcodec-dev libavdevice-dev \
        libavutil-dev libswscale-dev libswresample-dev libavfilter-dev \
        python3 python3-pip \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# ---- PyTorch (match host driver CUDA compatibility) ----
# typing-extensions<4.13 pins to Python 3.8 compatible version
RUN pip install --no-cache-dir \
        "typing-extensions==4.12.2" \
        torch==1.13.1+cu116 \
        torchvision==0.14.1+cu116 \
        --extra-index-url https://download.pytorch.org/whl/cu116

# ---- av: needs Cython<3 (av 9.x is pre-Cython3 era) ----
RUN pip install --no-cache-dir "Cython<3.0" \
    && pip install --no-cache-dir --no-build-isolation av==9.2.0

# ---- All other dependencies ----
RUN pip install --no-cache-dir \
        decord==0.6.0 \
        einops==0.8.0 \
        imageio==2.34.2 \
        numpy==1.24.4 \
        pillow==10.4.0 \
        requests==2.32.3 \
        tqdm==4.66.4 \
        peft==0.13.2 \
        pycocoevalcap==1.2 \
        sentence-transformers==3.0.1 \
        timm==1.0.12 \
        transformers==4.43.2 \
        wandb==0.19.1 \
        opencv-python==4.10.0.84 \
        SoccerNet==0.1.62 \
        huggingface_hub \
        sentencepiece \
        rouge-score

COPY . /workspace/

ENV PYTHONPATH=/workspace

CMD ["/bin/bash"]
