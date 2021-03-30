#!/bin/bash
sudo docker run -v "$(pwd)"/data:/app/dfl/data  johan/all-ml ./run-32.sh

