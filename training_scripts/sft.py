from trl import SFTConfig, SFTTrainer
from datasets import Dataset
from tqdm import tqdm
import yaml

from datetime import datetime
import logging
from inference import load_problems

def load_dataset(config: dict) -> Dataset:
    """
    This function loads datasets from a list of paths, format and concatenate them into a list as the training data.
    It should correctly handle different types of training datasets and rearrange them to a trainable format.
    """
    data = []
    ds = load_problems(config)
    ds_list = ds.to_list()
    for e in tqdm(ds_list):
        prompt = f"{config['trans_prompt']}\n\n{e['nlp']}"
        completion = f"```lean\n{e['gt_flp']}\n```"

        messages = { "messages": [
            {'role': 'user', 'content': prompt},
            {'role': 'assistant', 'content': completion}
        ]}
        data.append(messages)


    training_data = Dataset.from_list(data)
    return training_data

def sft_full(config: dict):
    """
    This function implements full parameter supervised fine-tuning with given model and datasets. The datasets should be generated via the aug data process.
    """
    current_time = datetime.now().strftime("%m%d%H%M")
    logger = logging.getLogger(__name__)
    logger.info(f"Tringing model {config['model']}")

    training_data = load_dataset(config)

    opt_dir = config['log_dir'] + '/sft' + current_time
    training_args = SFTConfig(
        output_dir=opt_dir,
        **config['sft_params']
    )

    trainer = SFTTrainer(
        config['model'],
        train_dataset = training_data,
        args = training_args,
    )

    trainer.train()

    with open(opt_dir + "/sftconfig.yaml", "w") as config_file:
        config['sft_params']['model_init_kwargs']['torch_dtype'] = ''
        yaml.safe_dump(config, config_file, default_flow_style=False)
