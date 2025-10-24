# Dockerfile
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

WORKDIR /workspace

# システムの最小ツールとPythonパッケージのインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

# Pythonライブラリ（requirements.txtをコピーしてまとめてインストール）
COPY requirements.txt /workspace/
RUN pip install --no-cache-dir -r requirements.txt

# コンテナ起動時の作業ディレクトリ
VOLUME ["/workspace"]
CMD ["/bin/bash"]
