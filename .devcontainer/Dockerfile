# CUDA + Ubuntu 기반
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# 필수 패키지 설치
RUN apt-get update && \
    apt-get install -y build-essential cmake git wget vim && \
    rm -rf /var/lib/apt/lists/*

# 작업 디렉토리
WORKDIR /workspace
