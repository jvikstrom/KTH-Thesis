# Decentralized federated learning

## Running
`sudo docker run -d --name agg-hypercube -v "$(pwd)"/data:/app/data -v "$(pwd)"/churn-files:/app/churn -e DATA_DIR=data -e FAIL_PATH=churn/mid-april128-10000-0.1.json -e CUDA_VISIBLE_DEVICES="3" -e PERC_DATA=0.25 johan/all-ml python3 dfl/main.py emnist agg-hypercube 128 5 10 10000 0.01`
