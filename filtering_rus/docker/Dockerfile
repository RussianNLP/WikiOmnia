FROM python:3.7

COPY requirements.txt model.json /code/ 

RUN pip install -r /code/requirements.txt

RUN python -m deeppavlov install /code/model.json

RUN python -m deeppavlov download /code/model.json

COPY *.py /code/

WORKDIR /code

