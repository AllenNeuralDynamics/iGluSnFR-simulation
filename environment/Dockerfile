# hash:sha256:0dab7d71f46ef7e1737c8587ef3049c04b124b358127328af8aff4454888b5bf
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:23.1.0-4-python3.10.12-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    argparse \
    h5py==3.11.0 \
    numpy==1.26.4 \
    opencv-python-headless==4.9.0.80 \
    scanimage-tiff-reader==1.4.1.4 \
    scikit-learn==1.5.2 \
    scipy==1.13.0 \
    tifffile==2024.5.10 \
    tqdm==4.66.4
