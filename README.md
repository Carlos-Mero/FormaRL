# FormaRL

This is the official repository of the paper accepted by COLM25: *FormaRL: Enhancing Autoformalization with no Labeled Data*.

Camera ready version of this paper and code implementation are coming soon.

Here is the training code used in our project.

## Setup

To start training with FormaRL, we need to firstly setup both python and lean environments. We will need at least 2xA100-80G GPU to train a large language model with 7B parameters.

#### Python Environment

Before installing the dependencies you should make sure you have created a new virtual environment for this project with python 3.12. Then you can directly run the following commands to install required packages.

```sh
pip install -r requirements.txt
```

Our training loop is primarily powered by [trl](https://github.com/huggingface/trl) and [accelerate](https://huggingface.co/docs/accelerate/index). You may need to manually run `accelerate config` to enable more optimizations in distributed training.

Moreover, if you need to use a remote model for consistency check with a openai-compatible API, you will need to create a `.env` file at the root folder of this project with these required information:

```sh
OPENAI_API_KEY=<your_api_key>
OPENAI_BASE_URL=<your_api_base_url/v1>
```

And pass `--remote` argument when you run these scripts.

#### Lean Environment

We used Lean and mathlib4 of version v4.21.0 in this training code. After you have cloned this repository, you need to run

```sh
git submodule init && git submodule update
```

to obtain the required version of mathlib4. Then you should add these dependencies to `mathlib4/lakefile.lean`

```lean
require Cli from git "https://github.com/leanprover/lean4-cli" @ "v4.21.0"
require REPL from git "https://github.com/leanprover-community/repl.git" @ "v4.21.0"
```

We requires the repl capability of lean in syntax check. Then you can build mathlib4 with

```sh
lake exe cache get
lake build
```

## Training

To reproduce the major experimental results in this paper, you can simply run this command

```sh
accelerate launch frl.py -c configs/grpo-coldstart.yaml -m Qwen/Qwen-2.5-Coder-7B --eval_model deepseek-v3 --save_path logs/ --remote
```

For more options and hyperparameters in this project, you can refer to definitions in `frl.py` and config files in configs folder for more information.
