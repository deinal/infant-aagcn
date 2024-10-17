FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime

WORKDIR /work

RUN mkdir -p /.config/matplotlib

RUN pip install -r requirements.txt

ENV PATH=/work/infant-aagcn/bin:$PATH
ENV PYTHONPATH=/work/infant-aagcn:$PYTHONPATH
