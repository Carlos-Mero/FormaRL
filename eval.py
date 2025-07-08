import os
import json
import torch
from multiprocessing import Manager
from vllm import LLM, SamplingParams
from transformers import pipeline
from tools import DataLoader,CompilerChecker,header
from tools.verifier import Lean4ServerScheduler
from tools.utils import load_config
from more_itertools import chunked
from accelerate import Accelerator
from gptqmodel import GPTQModel
from statistics import mean, stdev
import logging
import openai
from dotenv import load_dotenv

from datasets import Dataset
from utils import find_boxed, extract_code_block, remove_lean_comments


# function: check the compiler errors in formal_statement
# params: 
#   translations: a list of dictionaries, each dictionary has two keys: 'nlp' 'flp'
# return: a list of string, the string is empty when the result is correct or explaining the error when compiler error occurs
def check_compiler_correctness(translations):
    #response_queue = Queue(maxsize=300)
    hd = header()
    translations = [
        {'flp': hd + t['flp']}
    for t in translations]
    manager = Manager()
    response_dict = manager.dict() # Use shared dictionary to store results

    cfg = load_config('configs/eval_cfg.py')

    # create data loader
    data_loader = DataLoader(
        dataset=translations,
        node_rank=cfg.node_rank,
        world_size=cfg.world_size,
    )

    verifier = Lean4ServerScheduler(
        max_concurrent_requests=cfg.lean_max_concurrent_requests,
        memory_limit=cfg.lean_memory_limit,
        timeout=cfg.lean_timeout,
        name='verifier',
    )

    compiler_checkers = [
        CompilerChecker(
            idx=i+cfg.node_rank*cfg.n_check_procs,
            data_loader=data_loader,
            verifier=verifier,
            response_dict = response_dict
        ) 
        for i in range(min(cfg.n_check_procs, data_loader.size()))
    ]
    
    for compiler_checker in compiler_checkers:
        compiler_checker.start()
    
    for compiler_checker in compiler_checkers:
        compiler_checker.join()
    
    verifier.close()
    print(f'All {len(compiler_checkers)} Compiler Check Processes stopped')

    response_list = []
    for i in range(len(response_dict)):
        if response_dict[f"{i}"] == '':
            response_list.append(response_dict[f"{i}"])
        else:
            response_list.append(json.dumps(response_dict[f"{i}"]))
            
    return response_list


def check_consistency(translations: dict, model: str, sp_params: dict) -> list[float]:
    available_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    model = LLM(
        model=model,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=len(available_gpus),
    )

    sampling_params = SamplingParams(**sp_params)

    prompts = [f"Here is a natural language math problem and a translation in formal language Lean 4. You need to carefully analyse these problems and figure out wether they are equivalent or not. These problems must have exactly the same conditions and conclusions, they should be marked false if they violate any of these requirements. You should reply false if the given formal statement is empty or in a weird format.\n\n**Natural Language Problem**\n\n{e['nlp']}\n\n```lean\n{e['flp']}```\n\nState your answer as $\\boxed{{true}}$ or $\\boxed{{false}}$ at the end of your response."
    for e in translations]

    outputs = model.generate(prompts, sampling_params)
    outputs = sorted(outputs, key=lambda x: int(x.request_id))
    outputs = [output.outputs[0].text for output in outputs]
    
    consistency_results = []
    consistency_results = [1.0 if find_boxed(output) == 'true' else 0.0 for output in outputs]

    return consistency_results

def add_compiler_results_column(dataset,results):
    if 'compiler_check_result' in dataset.column_names:
        dataset = dataset.map(lambda x,idx:{'compiler_check_result':results[idx]},with_indices=True)
    else:
        dataset = dataset.add_column('compiler_check_result',results)

    return dataset

def add_consistency_results_column(dataset, results):
    if 'consistency_check_result' in dataset.column_names:
        dataset = dataset.map(lambda x, idx: {'consistency_check_result': results[idx]}, with_indices=True)
    else:
        dataset = dataset.add_column('consistency_check_result', results)

def add_results_column(dataset,results):
    dataset = add_compiler_results_column(dataset=dataset,results=results)
    dataset = add_consistency_results_column(dataset=dataset, results=results['consistency'])
    return dataset

def check_correctness(translations, model, sp_params):
    compiler_results = check_compiler_correctness(translations)
    consistency_results = check_consistency(translations, model, sp_params)
    return ['' if s == '' and cs == 1.0 else 'false' for (s, cs) in zip(compiler_results, consistency_results)]

def evaluate (
        translations: Dataset, # Input dataset, containing two columns: "nlp" to naturan language problem, "flp" the corresponding translation
        beta_1: float, # Hyperparameter 1, indicates the weight of compiler check
        beta_2: float, # Hyperparameter 2, indicates the weight of consistancy check
        beta_3: float, # Hyperparameter 3, the score when the model passed both two checks
        model: str, # The LLM name or path used in consistency check
        sp_params: dict, # The sampling params used in consistency check
    ) -> list[float]: # The result is the reward score of each translation pair
    # Each of the input translation pairs should go through both compiler check and consistancy check
    # The evaluated score should be 0 if one translation faild both checks
    # beta_1 if the translation only passed compiler check
    # beta_2 if the translation only passed consistancy check
    # beta_1 + beta_2 if the translation passed both checks

    compiler_results = check_compiler_correctness(translations)
    scores = [1.0 if result == '' else 0.0 for result in compiler_results]
    consistency_scores = check_consistency(translations, model, sp_params)
    
    final_scores = [beta_3 if (s > 0.0 and cs > 0.0) else beta_1 * s + beta_2 * cs for (s, cs) in zip(scores, consistency_scores)]

    return final_scores

class Evaluator():
    """
    The evaluator used in online GRPO trainer. It is used as the reward function.
    """
    def __init__(self, beta_1: float, beta_2: float, beta_3: float, config):
        self.logger = logging.getLogger(__name__)
        self.hd = header()
        self.accelerator = Accelerator()
        self.eval_device = f"cuda:{int(self.accelerator.num_processes) + 1}" # Set the vllm device to accelerator device + 1
        self.cfg = load_config('configs/eval_cfg.py')
        self.verifier = Lean4ServerScheduler(
            max_concurrent_requests=8,
            memory_limit=self.cfg.lean_memory_limit,
            timeout=self.cfg.lean_timeout,
            name='verifier',
        )
        available_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        self.batch_size = len(available_gpus)
        self.use_back_trans = config['use_back_trans']
        self.exclude_header = config['exclude_header']
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.beta_3 = beta_3

        # load GPTQ model as evaluation model
        if config['deepseek']:
            load_dotenv()
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
            self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
            def eval_pipe(prompts, **args):
                results = []
                for p in prompts:
                    while(True):
                        try:
                            response = self.client.chat.completions.create(
                                model="deepseek-chat",
                                max_completion_tokens=2048,
                                messages=[{'role': 'user', 'content': p}]
                            )
                            results.append([{"generated_text": response.choices[0].message.content}])
                            break
                        except:
                            self.logger.error("Error detected when calling deepseek API, retry")
                return results
            self.eval_pipe = eval_pipe
        else:
            model = GPTQModel.load(config['eval_model'])
            self.eval_pipe = pipeline(task="text-generation", model=model, tokenizer=model.tokenizer, device_map=self.accelerator.device)

    def compiler_check(self, translations, flps):
        manager = Manager()
        response_dict = manager.dict()
        data_loader = DataLoader(
            dataset=translations,
            node_rank=self.cfg.node_rank,
            world_size=self.cfg.world_size,
        )
        compiler_checkers = [
            CompilerChecker(
                idx=i+self.cfg.node_rank*self.cfg.n_check_procs,
                data_loader=data_loader,
                verifier=self.verifier,
                response_dict = response_dict
            )
            for i in range(min(self.cfg.n_check_procs, data_loader.size()))
        ]
        for compiler_checker in compiler_checkers:
            compiler_checker.start()
        for compiler_checker in compiler_checkers:
            compiler_checker.join()

        self.logger.info("compiler check")
        # Important! There might exist reward hack if there is no 'example' in flp
        # The empty string or strings with only comments will still pass the compiler check
        compiler_check_results = [1.0 if (response_dict[f"{i}"] == '' and flps[i] != '') else 0.0 for i in range(len(response_dict))]
        return compiler_check_results

    def consistency_check(self, nlps, flps):
        eval_prompts = [f"Here is a natural language math problem and a translation in formal language Lean 4. You need to carefully analyse these problems and figure out wether they are equivalent or not. These problems must have exactly the same conditions and conclusions, they should be marked false if they violate any of these requirements. You should reply false if the given formal statement is empty or in a weird format.\n\n**Natural Language Problem**\n\n{nlp}\n\n```lean\n{flp}```\n\nState your answer as $\\boxed{{true}}$ or $\\boxed{{false}}$ at the end of your response."
            for (nlp, flp) in zip(nlps, flps)]

        consistency_results = []
        self.logger.info("Running consistency check with huggingface pipeline")
        with torch.no_grad():
            for batch in chunked(eval_prompts, self.batch_size):
                outputs = self.eval_pipe(batch, do_sample=True, top_p=0.95, max_new_tokens=512, truncation=True)
                consistency_results += [1.0 if flps[i] != '' and "1+1=2" not in flps[i] and find_boxed(output[0]['generated_text']) == 'true' else 0.0 for (i, output) in enumerate(outputs)]
        return consistency_results

    def __call__(self, prompts, completions):
        # create nl-fl pairs for reward checks
        nlps = [prompt.split("\n\n", 1)[1] if isinstance(prompt, str) else prompt[0]['content'].split("\n\n", 1)[1] for prompt in prompts] # split between the inference prompt and problem to extract the problem
        flps = [extract_code_block(completion) if isinstance(completion, str) else extract_code_block(completion[0]['content']) for completion in completions]
        flps = [remove_lean_comments(completion) if flp == '' else remove_lean_comments(flp) for (completion, flp) in zip(completions, flps)]

        translations = [
            {'nlp': nlp, 'flp': flp} if self.exclude_header else
            {'nlp': nlp, 'flp': self.hd + flp}
        for (nlp, flp) in zip(nlps, flps)]
        compiler_check_results = self.compiler_check(translations, flps)

        consistency_results = self.consistency_check(nlps, flps)
        
        rewards = [self.beta_3 if (compiler_r > 0.0 and consistency_r > 0.0) else self.beta_1 * compiler_r + self.beta_2 * consistency_r for (compiler_r, consistency_r) in zip(compiler_check_results, consistency_results)]

        self.logger.info(f"rewards: {rewards}")
        if len(rewards) > 1:
            self.logger.info(f"reward mean: {mean(rewards)}, reward std: {stdev(rewards)}")
        if self.accelerator.is_main_process:
            logidx = rewards.index(max(rewards))
            self.logger.info(f"sample output cot with the best reward:\n{prompts[logidx]}\n{completions[logidx]}")
            self.logger.info(f"sample output flp with best reward:\n{flps[logidx]}")
        
        return rewards

    def __del__(self):
        self.verifier.close()

    @property
    def __name__(self):
        return "Evaluator"
