#!/bin/bash
sudo docker run -v data:/app/dfl/data -v /usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib johan/all-ml ./run-32.sh
