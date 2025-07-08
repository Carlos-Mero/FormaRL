from datasets import load_from_disk, Dataset, concatenate_datasets
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from bitsandbytes.optim import Adam8bit

import logging
from datetime import datetime
import yaml

from eval import Evaluator
from inference import load_problems

def load_dataset(config: dict) -> Dataset:
    logger = logging.getLogger(__name__)
    dss = []
    for path in config['training_datasets']:
        logger.info(f"loading dataset from {path}")
        dss.append(load_from_disk(path))
    return concatenate_datasets(dss)

class offline_reward_func():
    def __init__(self, ds):
        self.count = 0
        self.rewards = []
        for e in ds:
            self.rewards += e['rewards']
    def __call__(self, prompts, completions):
        lcomp = len(completions)
        self.count += lcomp
        return self.rewards[self.count - lcomp : self.count]

def grpo_online(config: dict):
    ds = load_problems(config)
    prompts = [f"{config['trans_prompt']}\n\n{e['nlp']}" for e in ds]

    messages = Dataset.from_list([{"prompt" : [{'role': 'user', 'content': p}]}
        for p in prompts])

    evaluator = Evaluator(config['beta_1'], config['beta_2'], config['beta_3'], config)

    current_time = datetime.now().strftime("%m%d%H%M")
    opt_dir = './logs' + '/grpo' + current_time
    grpo_configs = GRPOConfig(
        output_dir=opt_dir,
        **config['grpo_configs'])
    model = AutoModelForCausalLM.from_pretrained(config['model'])
    tokenizer = AutoTokenizer.from_pretrained(config['model'])
    tokenizer.pad_token = tokenizer.eos_token
    optimizer = Adam8bit(model.parameters(), lr=grpo_configs.learning_rate)
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=evaluator,
        args=grpo_configs,
        train_dataset=messages,
        # peft_config=LoraConfig(**config['lora_configs'])
    )
    trainer.optimizer = optimizer

    with open(opt_dir + "/grpoconfig.yaml", "w") as config_file:
        yaml.safe_dump(config, config_file, default_flow_style=False)

    trainer.train()
