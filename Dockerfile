FROM debian:jessie

USER root

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get -y update && \
    apt-get -y upgrade && \
    apt-get -y install gfortran libopenmpi-dev openmpi-bin openmpi-common \
                       liblapack-dev libatlas-base-dev libatlas-dev mercurial

USER main

RUN wget http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b && \
    rm ~/miniconda.sh

ENV PATH $HOME/miniconda2/bin:$PATH

RUN conda update --yes conda

RUN pip install numpy scipy
RUN pwd
RUN ls
RUN pip install -r requirements.txt
RUN python setup.py install

CMD [ "/bin/bash" ]