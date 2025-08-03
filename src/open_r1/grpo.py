# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
os.environ["WANDB_API_KEY"] = "8d9092d39ddbffe346a0646fd17158425d700aff"
import datasets
import transformers
from datasets import load_dataset
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from open_r1.configs import GRPOConfig, GRPOScriptArguments
from open_r1.rewards import get_reward_funcs
from open_r1.utils import get_model, get_tokenizer
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.wandb_logging import init_wandb_training
from trl import GRPOTrainer, ModelConfig, TrlParser, get_peft_config
import torch

logger = logging.getLogger(__name__)

from transformers import TrainerCallback

import wandb
import re

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")

class GradientMonitorCallback(TrainerCallback):
    def __init__(self):
        self.grad_running_mean = None
        self.grad_running_mean_squared = None

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        return
        model = kwargs["model"]
        accelerator = kwargs["accelerator"]

        # this is step before current step, because step increments after this callback in trainer
        step = state.global_step

        # Collect all gradients in a single flattened vector
        grads = [p.grad.detach().view(-1) for p in model.parameters() if p.grad is not None]
        if not grads:
            return  # Skip if no grads this step

        flat_grad = torch.cat(grads)
        grad_norm = torch.norm(flat_grad, p=2).item()
        grad_var = torch.var(flat_grad).item()

        # Initialize or update running stats
        if self.grad_running_mean is None:
            self.grad_running_mean = flat_grad.clone()
            self.grad_running_mean_squared = flat_grad.clone() ** 2
        else:
            self.grad_running_mean = (self.grad_running_mean * step + flat_grad) / (step + 1)
            self.grad_running_mean_squared = (self.grad_running_mean_squared * step + flat_grad ** 2) / (step + 1)

        # Reduce stats across processes (mean reduction)
        grad_norm_tensor = torch.tensor(grad_norm, device=accelerator.device)
        grad_var_tensor = torch.tensor(grad_var, device=accelerator.device)

        grad_norm_tensor = accelerator.reduce(grad_norm_tensor, reduction="mean")
        grad_var_tensor = accelerator.reduce(grad_var_tensor, reduction="mean")
        flat_grad = accelerator.reduce(flat_grad, reduction="mean")
        self.grad_running_mean = accelerator.reduce(self.grad_running_mean, reduction="mean")
        self.grad_running_mean_squared = accelerator.reduce(self.grad_running_mean_squared, reduction="mean")
        if step + 1 >= 10:
            grad_std = (self.grad_running_mean_squared - self.grad_running_mean ** 2).sqrt()
            lambda_sigma = 3.0
            deviation = (flat_grad - self.grad_running_mean).abs()
            outliers = (deviation > lambda_sigma * grad_std)
            proportion_outliers = outliers.float().mean().item()


        if accelerator.is_main_process:
            info = {
                "grad/post_clip_norm": grad_norm_tensor.item(),
                "grad/variance": grad_var_tensor.item(),
            }
            if step + 1 >= 10:
                info["grad/proportion_spike"] = proportion_outliers
            #print(f"[Step Debug] HF global_step: {state.global_step}, wandb.run.step: {wandb.run.step}")
            wandb.log(info, step=wandb.run.step + 1)
            #print(f"[Step {step + 1}] Pre-clip grad norm: {grad_norm_tensor.item():.4f} | Var: {grad_var_tensor.item():.4f}")
def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    # Load the dataset
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_args, training_args)

    ##############
    # Load model #
    ##############
    logger.info("*** Loading model ***")
    model = get_model(model_args, training_args)

    # Get reward functions from the registry
    reward_funcs = get_reward_funcs(script_args)

    def extract_hash_answer(completion):
        match = ANS_RE.search(completion)
        if match:
            match_str = match.group(1).strip()
            match_str = match_str.replace(",", "")
            return match_str
        else:
            return None
    # Format into conversation
    def make_conversation(example, prompt_column: str = script_args.dataset_prompt_column):
        prompt = []

        if training_args.system_prompt is not None:
            prompt.append({"role": "system", "content": training_args.system_prompt})

        if prompt_column not in example:
            raise ValueError(f"Dataset Question Field Error: {prompt_column} is not supported.")
        solution = extract_hash_answer(example["answer"])
        prompt.append({"role": "user", "content": example[prompt_column]})
        return {"prompt": prompt, "solution": solution}

    dataset = dataset.map(make_conversation)

    for split in dataset:
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")

    #############################
    # Initialize the GRPO trainer
    #############################
    call_backs = get_callbacks(training_args, model_args)
    call_backs.append(GradientMonitorCallback)
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=(dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None),
        peft_config=get_peft_config(model_args),
        callbacks=call_backs,
        processing_class=tokenizer,
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["open-r1"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    #############
    # push to hub
    #############
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)