export BS=12
export MEMCAP=0
export MODEL="6.7b"
export GPUNUM=4
export GMR=0.6
export PWD="checkpoints"
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# make directory for logs
mkdir -p ./logs

# env PYTORCH_NO_CUDA_MEMORY_CACHING=1
# --resume_from_checkpoint ${PWD}/weights_60000 \
torchrun --standalone --nproc_per_node ${GPUNUM} \
    --master_port 29500 \
    run_clm.py \
    --weight_decay 0.1 \
    --num_warmup_steps 16000 \
    --learning_rate 1.2e-5 \
    --resume_from_checkpoint ${PWD}/weights_20000 \
    --model_name_or_path facebook/opt-${MODEL} \
    --output_dir ${PWD} \
    --mem_cap ${MEMCAP} \
    --per_device_train_batch_size ${BS} 2>&1 | tee ./logs/colo_${MODEL}_bs_${BS}_cap_${MEMCAP}_gpu_${GPUNUM}_gpu_margin_mem_ratio_${GMR}.log
