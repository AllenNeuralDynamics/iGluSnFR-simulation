# hash:sha256:20d15f3448fd4788c67ff9f7f3a50f41a18fd7977a87a8b17574eeab08e0c85f
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
    scanimage-tiff-reader==1.4.1.4 \
    scipy==1.13.0 \
    tifffile==2024.5.10 \
    tqdm==4.66.4

ADD "https://github.com/coder/code-server/releases/download/v4.9.0/code-server-4.9.0-linux-amd64.tar.gz" /.code-server/code-server.tar.gz
	
RUN cd /.code-server \
	&& tar -xvf code-server.tar.gz \
	&& rm code-server.tar.gz \
	&& ln -s /.code-server/code-server-4.9.0-linux-amd64/bin/code-server  /usr/bin/code-server
