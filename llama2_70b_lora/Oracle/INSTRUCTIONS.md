# Oracle implementation

## Initial setup

```bash
BASE_DIR=/persistent_storage-daniel/daniel/mlperf_benchmarks/flex_training/llama2_70b_lora/  # Change to your base directory
MAIN_DIR=$BASE_DIR/Oracle/
RESOURCES_DIR=$BASE_DIR/resources/
mkdir -p $RESOURCES_DIR/dataset $RESOURCES_DIR/model
```

## Setup Docker

### Build Docker image:

```bash
docker build --pull -t mlperf-llama-oracle-image $MAIN_DIR
```

### Run Docker container:

```bash
docker run \
  -it \
  --rm \
  --gpus all \
  --name mlperf-llama-oracle-container \
  --volume $RESOURCES_DIR/dataset:/dataset \
  --volume $RESOURCES_DIR/model:/model \
  --volume $BASE_DIR/scripts:/scripts \
  --volume $MAIN_DIR/results:/workspace/ft-llm/results \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  mlperf-llama-oracle-image
```

## Download resources

### Download dataset:

```bash
python3 /scripts/download_dataset.py --data_dir /dataset/
```

### Download model:

```bash
python3 /scripts/download_model.py --model_dir /model/
```

## Convert resources

### Convert dataset

```bash
python3 ./scripts/convert_dataset.py --data_dir /dataset/
```

### Convert model

```bash
python3 ./scripts/convert_model.py \
  --input_name_or_path /model/ \
  --output_path /model/llama2-70b.nemo \
  --hparams_file ./scripts/megatron_llama_config.yaml \
  --tokenizer_path ./scripts/tokenizer/ \
  --precision 16

tar -xvf /model/llama2-70b.nemo -C /model/
```

## Training

### Exit Docker container

```bash
exit
```

### Launch training:

```bash
source $MAIN_DIR/configs/config_flex.sh  # TP_COMM_OVERLAP is disabled to allow Docker run

CONT=mlperf-llama-oracle-image \
DATADIR="$RESOURCES_DIR/dataset/" \
MODEL="$RESOURCES_DIR/model/" \
  $MAIN_DIR/run_with_docker.sh
```