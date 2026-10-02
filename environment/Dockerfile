# hash:sha256:834bc432a286ede4ef8e19669def310eac217a734250b82d88abb007bfdaf755
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:23.1.0-4-python3.10.12-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

RUN --mount=type=secret,id=git-access-token \
    GIT_ACCESS_TOKEN="$(cat /run/secrets/git-access-token)" \
    pip install -U --no-cache-dir \
    aind-data-schema==2.9.1 \
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
    tqdm==4.66.4 \
    -e git+https://github.com/AllenNeuralDynamics/analysis-pipeline-utils.git@7b6ec46c2abdbc14318f316fc949dd8714d07a9d#egg=analysis-pipeline-utils
