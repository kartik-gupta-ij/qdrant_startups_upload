#!/usr/bin/env bash


rsync -avP --exclude='venv' \
           . $1:./project/python_files/


