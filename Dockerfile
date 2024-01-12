FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-runtime

# update apt-get, install gcc and dev tools for python and upgrade pip
RUN apt-get update
RUN apt-get install -y gcc python3-dev build-essential git
RUN pip install --upgrade pip

# copy requirements.txt and install dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# set the working directory for the code
WORKDIR /mtebscripts