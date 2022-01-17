#!/bin/bash

SCRIPTDIR=$(dirname $0)
cd $SCRIPTDIR
ls

docker rm -f dp_run
docker build -t filtering_rus ./docker

echo run -d --name dp_run -v $(pwd)/data:/data filtering_rus python /code/main.py t5_batch${1}_all
docker run -d --name dp_run -v $(pwd)/data:/data filtering_rus python /code/main.py t5_batch${1}_all
