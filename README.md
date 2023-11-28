# Implementation of the paper "Tracking Emerges by Colorizing Videos"

## Setup

### Environment
Either start a docker container with all the environment set up or install the dependencies manually.

To start the docker container, run the following command:

```bash
docker compose up -d
```

To install the dependencies manually, please create a virtual Python environment in advance, then install the dependencies with the following command:

```bash
pip install -r requirements.txt
```

### Files

Open the repo directory via docker container or switch to the virtual environment, then download and prepare neccessary files:
    
```bash
chmod +x setup.sh
./setup.sh
```

**Note**: The script only download a subset of Kinetics 700 dataset, please view the script if you want to make change. Please make sure you have enough space to download it.