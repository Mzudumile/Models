#!/usr/bin/env bash
set -e
cd OFA
# clone OFA if it doesn’t exist
if [ ! -d "OFA" ]; then
    echo "Cloning OFA repository..."
    git clone --single-branch --branch feature/add_transformers https://github.com/OFA-Sys/OFA.git
else
    echo "OFA already exists, skipping clone"
fi

# start your FastAPI app
uvicorn test:app --host 0.0.0.0 --port 8000
