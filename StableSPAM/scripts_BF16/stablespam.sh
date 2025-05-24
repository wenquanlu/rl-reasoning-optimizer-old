

# module purge
# module load 2023
source activate spam
# Your job starts in the directory where you call sbatch
cd ..

torchrun --standalone --nproc_per_node 4 main_pretrain.py \
    --model_config configs/llama_130m.json \
    --eval_every 1000 \
    --save_every 100000 \
    --dtype bfloat16 \
    --batch_size 128 \
    --total_batch_size 512 \
    --lr 0.0008 \
    --warmup_steps 2000 \
    --num_training_steps 20000 \
    --optimizer stablespam \
    --weight_decay 0 \
    --project stablespam \
    --name stablespam_350_fp4_500_0.9_0.7_4e-4 \
    --save_dir /scratch-shared/saved \
    --restore_optimizer \
    --gamma1 0.85 \
    --gamma2 0.99999 \
    --gamma3 0.999 \
    --update_proj_gap 1000 