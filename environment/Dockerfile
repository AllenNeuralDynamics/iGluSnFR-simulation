# hash:sha256:6dd1c1b9676a3ae2514ad23c7a3eb3ee1e5429c93092cedef910a1bb9706de2f
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:23.1.0-4-python3.10.12-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    argparse \
    h5py==3.11.0 \
    numpy==1.26.4 \
    opencv-python-headless==4.9.0.80 \
    scanimage-tiff-reader==1.4.1.4 \
    scikit-learn==1.5.2 \
    scipy==1.13.0 \
    sparse==0.17.0 \
    tifffile==2024.5.10 \
    tqdm==4.66.4
