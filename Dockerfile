FROM tensorflow/tensorflow:1.10.1-gpu

RUN apt-get update && apt-get install -y git
RUN pip install -U cython
RUN pip install git+https://github.com/lucasb-eyer/pydensecrf.git