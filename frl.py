import argparse
import yaml
import logging
import os

from training_scripts.sft import sft_full
from training_scripts.grpo import grpo_online
from inference import translate_rtpipeline

from utils import set_all_seeds

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('cott.log'),
            logging.StreamHandler()
        ]
    )

def main():
    setup_logging()
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    parser = argparse.ArgumentParser(description="FormaRL training scripts for autoformalization")
    parser.add_argument('-c', '--config', nargs='+', default=[], help="The config file for this program. This config should be in yaml format.")
    parser.add_argument('-t', '--test', action='store_true', default=False, help='enable test output for the code')
    parser.add_argument('-m', '--model', type=str, default='', help="This argument specifies the model path")
    parser.add_argument('-d', '--datasets', nargs='+', default=[], help="The training dataset used in this program")
    parser.add_argument('-e', '--epochs', type=int, default=-1, help="The training epochs over the dataset")
    parser.add_argument('--save_path', type=str, default='', help="The path to save the inference results.")
    parser.add_argument('-s', '--seed', type=int, default=1121, help="The global random seed in this program.")
    parser.add_argument('-n', '--n_samples', type=int, default=-1, help="The samples per prompt used in generating samples for GRPO")
    parser.add_argument('--eval_model', type=str, default="Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8", help="The model used in consistency check")
    parser.add_argument('-r', '--remote', action='store_true', default=False, help="Use remote API call in consistency check")
    parser.add_argument('--grpo_beta', type=float, default=-1.0, help="The hyperparameter of the grpo trainer, used to introduce KL divergence term into the loss function")
    parser.add_argument('--beta_1', type=float, default=-1.0, help="The hyperparameter of the evaluator in grpo")
    parser.add_argument('--beta_2', type=float, default=-1.0, help="The hyperparameter of the evaluator in grpo")
    parser.add_argument('--beta_3', type=float, default=-1.0, help="The hyperparameter of the evaluator in grpo")

    args = parser.parse_args()

    if args.config == []:
        print("The config file is required for this program")
        exit(1)
    else:
        for config_file in args.config:
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)
            # overwrite original configs by command line arguments
            config['test'] = args.test
            config['seed'] = args.seed
            config['remote'] = args.remote
            config['eval_model'] = args.eval_model
            set_all_seeds(args.seed)
            if args.model:
                config['model'] = args.model
            if args.save_path:
                config['save_path'] = args.save_path
            if args.datasets:
                config['training_datasets'] = args.datasets
            if args.epochs > 0:
                config['num_train_epochs'] = args.epochs
            if args.n_samples > 0:
                config['n_samples'] = args.n_samples
            if args.grpo_beta >= 0.0:
                config['grpo_configs']['beta'] = args.grpo_beta
            if args.beta_1 >= 0.0:
                config['beta_1'] = args.beta_1
            if args.beta_2 >= 0.0:
                config['beta_2'] = args.beta_2
            if args.beta_3 >= 0.0:
                config['beta_3'] = args.beta_3

            if config['type'] == "sft" or config['type'] == "sft-baseline":
                sft_full(config)
            elif config['type'] == "infer":
                translate_rtpipeline(config)
            elif config['type'] == "grpo-online":
                grpo_online(config)
            else:
                raise NotImplementedError("Unknow task!")

if __name__ == "__main__":
    main()
