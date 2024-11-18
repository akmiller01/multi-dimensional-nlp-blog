#!/bin/bash

source venv/bin/activate
rm -rf ./data/*
python3 code/download_datastore_api.py && python3 code/merge_and_upload_disability.py
