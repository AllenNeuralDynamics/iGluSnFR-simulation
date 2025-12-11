# hash:sha256:5ffab529fb6c5f6caca518248df56ce9cbe72246fc6b3bb7118cf56ddb7b05e4
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:23.1.0-4-python3.10.12-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    argparse \
    h5py==3.11.0 \
    matplotlib==3.10.8 \
    numpy==1.26.4 \
    opencv-python-headless==4.9.0.80 \
    scanimage-tiff-reader==1.4.1.4 \
    scikit-learn==1.5.2 \
    scipy==1.13.0 \
    sparse==0.17.0 \
    tifffile==2024.5.10 \
    tqdm==4.66.4
