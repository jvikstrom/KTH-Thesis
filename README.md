# Decentralized federated learning

## Running
`sudo docker run -d --name agg-hypercube -v "$(pwd)"/data:/app/data -v "$(pwd)"/churn-files:/app/churn -e DATA_DIR=data -e FAIL_PATH=churn/mid-april128-10000-0.1.json -e CUDA_VISIBLE_DEVICES="3" -e PERC_DATA=0.25 johan/all-ml python3 dfl/main.py emnist agg-hypercube 128 5 10 10000 0.01`


`sudo docker run -d --name exchange-cycle-nn5 -v "$(pwd)"/data-nn5:/app/data -v "$(pwd)"/churn-files:/app/churn -e DATA_DIR=data -e CUDA_VISIBLE_DEVICES="3" -e PERC_DATA=0.25 -e NN5_PATH=data/nn5.csv johan/nn5-ml python3 dfl/main.py nn5 exchange-cycle 64 5 10 1000 0.01
`