#FROM nvidia/cuda:10.2-base-ubuntu18.04
FROM tensorflow/tensorflow:2.3.0-gpu

#RUN apt-get update
#RUN apt-get install -y python3.6 python3-pip


WORKDIR /app


COPY . .

#RUN python3.6 -m pip install -U pip

RUN python3.6 -m pip install -r requirements.txt


