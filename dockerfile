# Dockerfile

# 使用 Ubuntu 22.04 LTS 作为基础镜像
FROM ubuntu:22.04

# 设置环境变量，避免在安装包时弹出交互式提示
ENV DEBIAN_FRONTEND=noninteractive

# --- 1. 安装核心构建工具 ---
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    && \
    rm -rf /var/lib/apt/lists/*

# --- 2. 安装 Armadillo 依赖及其特定功能后端 ---
RUN apt-get update && \
    apt-get install -y \
    libopenblas-dev \
    liblapack-dev \
    libarpack2-dev \
    libsuperlu-dev \
    && \
    rm -rf /var/lib/apt/lists/*

# --- 3. 从源码编译安装 Armadillo 数学库 ---
ENV ARMADILLO_VERSION=14.6.0
RUN wget http://sourceforge.net/projects/arma/files/armadillo-${ARMADILLO_VERSION}.tar.xz -O armadillo-${ARMADILLO_VERSION}.tar.xz && \
    tar -xJf armadillo-${ARMADILLO_VERSION}.tar.xz && \
    cd armadillo-${ARMADILLO_VERSION} && \
    cmake -DCMAKE_BUILD_TYPE=Release . && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    rm -rf /armadillo-${ARMADILLO_VERSION}.tar.xz && \
    rm -rf /armadillo-${ARMADILLO_VERSION}

# --- 4. 从源码编译安装 Google Test (GTest) ---
ENV GTEST_VERSION=1.12.1
ENV GTEST_GIT_REPO="https://github.com/google/googletest.git"
RUN git clone --depth 1 --branch release-${GTEST_VERSION} ${GTEST_GIT_REPO} /googletest-release-${GTEST_VERSION} && \
    cd /googletest-release-${GTEST_VERSION} && \
    cmake -DCMAKE_BUILD_TYPE=Release . && \
    make -j$(nproc) && \
    make install && \
    ldconfig \
    && \
    # Clean up cloned source to reduce image size
    rm -rf /googletest-release-${GTEST_VERSION} && \
    rm -rf /var/lib/apt/lists/*

# --- 5. 从源码编译安装 Google Logging (glog) ---
ENV GLOG_VERSION=0.7.0
ENV GLOG_GIT_REPO="https://github.com/google/glog.git"
RUN git clone --depth 1 --branch v${GLOG_VERSION} ${GLOG_GIT_REPO} /glog-${GLOG_VERSION} && \
    cd /glog-${GLOG_VERSION} && \
    cmake -DCMAKE_BUILD_TYPE=Release -DWITH_GFLAGS=OFF -DWITH_GTEST=OFF . && \
    make -j$(nproc) && \
    make install && \
    ldconfig \
    && \
    rm -rf /glog-${GLOG_VERSION} && \
    rm -rf /var/lib/apt/lists/*

# --- 6. 从源码编译安装 SentencePiece 分词库 ---
ENV SENTENCEPIECE_VERSION=0.2.0
ENV SENTENCEPIECE_GIT_REPO="https://github.com/google/sentencepiece.git"
RUN git clone --depth 1 --branch v${SENTENCEPIECE_VERSION} ${SENTENCEPIECE_GIT_REPO} /sentencepiece-${SENTENCEPIECE_VERSION} && \
    cd /sentencepiece-${SENTENCEPIECE_VERSION} && \
    cmake -DCMAKE_BUILD_TYPE=Release . && \
    make -j$(nproc) && \
    make install && \
    ldconfig \
    && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /sentencepiece-${SENTENCEPIECE_VERSION}


WORKDIR /workspaces/KuiperLLama
CMD ["bash"]