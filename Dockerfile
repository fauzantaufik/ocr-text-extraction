FROM python:3.7

# add the repository to install "checkinstall"
RUN echo 'deb http://deb.debian.org/debian buster-backports main' >> /etc/apt/sources.list

RUN apt-get update --allow-insecure-repositories && apt-get install --allow-unauthenticated -y \
    tesseract-ocr \
    libgl1-mesa-glx \
    libpq-dev \
    libopenjp2-7-dev \
    libgdk-pixbuf2.0-dev \
    cmake \
    checkinstall \
    libpoppler-cpp-dev

# build poppler
COPY ./poppler-0.90.1.tar.xz  .
RUN tar -xf ./poppler-0.90.1.tar.xz && \
    rm ./poppler-0.90.1.tar.xz
WORKDIR ./poppler-0.90.1/build 
RUN cmake .. 
RUN checkinstall make install

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt
RUN python -c 'import nltk; nltk.download("punkt")'
COPY . /app

RUN ldconfig