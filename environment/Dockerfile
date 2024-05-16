# hash:sha256:70a1c9f2ce274642f1ab8910883229f2462c67f72456e589e73179bfba5a6cde
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
    opencv-python \
    scanimage-tiff-reader==1.4.1.4 \
    scipy==1.13.0 \
    tifffile==2024.5.10 \
    tqdm==4.66.4
