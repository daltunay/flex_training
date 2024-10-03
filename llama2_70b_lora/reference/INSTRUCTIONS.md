# Reference implementation

## Initial setup

```bash
BASE_DIR=/persistent_storage-daniel/daniel/mlperf_benchmarks/flex_training/llama2_70b_lora/  # Change to your base directory
MAIN_DIR=$BASE_DIR/reference/
RESOURCES_DIR=$BASE_DIR/resources/
mkdir -p $RESOURCES_DIR/dataset $RESOURCES_DIR/model
```

## Setup Docker

### Build Docker image:

```bash
docker build --pull -t mlperf-llama-reference-image $MAIN_DIR
```

### Run Docker container:

```bash
docker run \
  -it \
  --rm \
  --gpus all \
  --name mlperf-llama-reference-container \
  --volume $RESOURCES_DIR/dataset:/dataset \
  --volume $RESOURCES_DIR/model:/model \
  --volume $BASE_DIR/scripts:/scripts \
  --volume $MAIN_DIR/results:/workspace/ft-llm/results \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  mlperf-llama-reference-image
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

## Training

### Launch training:

```bash
accelerate launch --config_file ./configs/default_config.yaml ./scripts/train.py \
  --dataset_path "/dataset/data/" \
  --model_path "/model/" \
  --max_seq_len 8192 \
  --bf16 True \
  --logging_steps 24 \
  --eval_steps 48 \
  --output_dir "./results/" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --lr_scheduler_type "cosine" \
  --learning_rate 4e-4 \
  --weight_decay 0.0001 \
  --warmup_ratio 0 \
  --max_grad_norm 0.3 \
  --use_gradient_checkpointing True \
  --target_eval_loss 0.925 \
  --use_peft_lora True \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.1 \
  --max_steps 1024 \
  --use_flash_attn \
  --seed 1234 \
  --lora_target_modules "qkv_proj,o_proj"
```