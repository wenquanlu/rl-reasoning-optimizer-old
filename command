ACCELERATE_LOG_LEVEL=info     accelerate launch --config_file recipes/accelerate_configs/zero2.yaml --num_processes 1     src/open_r1/grpo.py --config recipes/Qwen2.5-0.5B-Instruct/grpo/config_demo.yaml



sbatch --nodes=2 slurm/train.slurm --model Qwen2.5-1.5B-Instruct --task grpo --config demo --accelerator ddp --dp 8 --tp 1



sbatch --nodes=2 \
  --nodelist=gpu02,gpu05 \
  slurm/train.slurm \
  --model Qwen2.5-7B-Instruct \
  --task grpo \
  --config demo \
  --accelerator ddp \
  --dp 2 \
  --tp 1 \
  --args "--eos_token='<|im_end|>'"



sbatch --nodes=2    slurm/train.slurm   --model Qwen2.5-7B-Instruct   --task grpo   --config demo   --accelerator ddp   --dp 2   --tp 1 


sbatch --nodes=1    slurm/train.slurm   --model Llama-3.2-3B-Instruct   --task grpo   --config demo_boxed   --accelerator ddp   --dp 2   --tp 1