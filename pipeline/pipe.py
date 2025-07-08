# This is the self evolution pipeline for translation to Lean4
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment
from datasets import Dataset
import yaml
import torch
import ray
import contextlib
import openai

import os
import logging
import gc

from utils import extract_code_block
from eval import evaluate
from dotenv import load_dotenv

class RTPipeline:
    """
    This is the pipeline used to run the reason-translation process with correctness check and feedback
    """
    def __init__(self,
                 config: dict, # The global config file for this pipeline
                 model: str, # The path to the model ckpt or huggingface model nane
                 model_init_params: dict, # The parameters to initialize the LLM engine in vllm
                 sampling_params: dict, # The sampling_params for vllm
                 ):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.model = model
        available_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        if self.config['deepseek']:
            load_dotenv()
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
            self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.llm = LLM(
                model=self.model,
                tensor_parallel_size=len(available_gpus),
                distributed_executor_backend="mp",
                disable_custom_all_reduce=True,
                **model_init_params
            )
            self.sampling_params = SamplingParams(
                **sampling_params,
                n=self.config['n_samples']
            )
        self.logger.info("Successfully initialized RTPipeline")

    def check(self, nlfl_pairs):
        # There are also vllm calls in check_correctness, so we need to cleanup vllm stuffs before calling check.
        self.cleanup()
        results = evaluate(nlfl_pairs,
                           beta_1=1.0,
                           beta_2=0.0,
                           beta_3=4.0,
                           model=self.config['eval_model'],
                           sp_params=self.config['sampling_params'])
        return results

    def __call__(self,
                 ds: Dataset, # This huggingface dataset containing problems to be translated (This dataset should have one column "nlp" containing complete natural language problems to be translated)
                 ):
        # direct translate
        problem_count = 0
        success_counts = []
        solved_problems = []
        if self.config['deepseek']:
            conversations = [
                    [{'role': 'user', 'content': f"{self.config['trans_prompt']}\n\n{e['nlp']}"}]
                for e in ds]
            problem_count = len(conversations)
            self.logger.info("Doing translation on these datasets with remote API")
            outputs = []
            for c in conversations:
                while(True):
                    try:
                        response = self.client.chat.completions.create(
                            model=self.model,
                            max_completion_tokens=2048,
                            messages=c
                        )
                        outputs.append(response.choices[0].message.content)
                        break
                    except:
                        self.logger.error("Error detected when calling deepseek API, retry")

        else:
            conversations = [
                    [{'role': 'user', 'content': f"{self.config['trans_prompt']}\n\n{e['nlp']}"}]
                for e in ds]
            problem_count = len(conversations)

            self.logger.info("Doing translation on these datasets")
            outputs = self.llm.chat(
                messages=conversations,
                sampling_params=self.sampling_params,
                use_tqdm=True
            )
            outputs = sorted(outputs, key=lambda x: int(x.request_id))
            outputs = [e.text for output in outputs for e in output.outputs]
        self.logger.info("extracting lean code blocks from the responses")
        flps = [extract_code_block(t) for t in outputs]

        # final check
        n = self.config['n_samples']
        nlfl_pairs = Dataset.from_list([
            {"nlp": e['nlp'], "flp": flps[i * n + k]}
            for (i, e) in enumerate(ds.to_list()) for k in range(n)])

        check_results = self.check(nlfl_pairs)
        if n > 1:
            # check_results = [max(check_results[i * n : (i+1) * n]) for i in range(len(check_results) // n)]
            # best_ids = [next((i for i, v in enumerate(check_results[i * n : (i+1) * n]) if v > 0.0), 0)
            #     for i in range(len(check_results) // n)]
            best_ids = [0] * (len(check_results) // n)
            for i in range(len(check_results) // n):
                for j in range(n):
                    if check_results[i * n + j] > 0.0:
                        best_ids[i] = j
                        break
            check_results = [check_results[n * i + id] for i, id in enumerate(best_ids)]
        else:
            best_ids = [0] * len(check_results)
        compiler_pass_count = len([r for r in check_results if r > 0.0])
        remaining_problems = [
            {**e, 'flp': flps[i * n], 'CoT': outputs[i * n], 'error': r}
        for (i, (e, r)) in enumerate(zip(ds.to_list(), check_results)) if r < 4.0]
        solved_problems = [
            {**e, 'flp': flps[i * n + id], 'CoT': outputs[i * n + id]}
        for (i, (e, id, r)) in enumerate(zip(ds.to_list(), best_ids, check_results)) if r == 4.0]
        success_counts.append(problem_count - len(remaining_problems))
        self.logger.info(f"Compiler check pass count: {compiler_pass_count}")
        self.logger.info(f"Compiler check pass rate: {float(compiler_pass_count)/float(problem_count)}")
        self.logger.info(f"Final success_count: {success_counts[-1]}")
        self.logger.info(f"Final pass rate: {float(success_counts[-1])/float(problem_count)}")
        print("="*100)
        self.logger.info(f"Inference Pipeline ended with pass count: {success_counts} / {problem_count}")
        self.config['pass_count'] = success_counts
        self.config['problem_count'] = problem_count
        # returning output datasets
        return (Dataset.from_list(solved_problems), Dataset.from_list(remaining_problems))

    def save_logs(self, solved_problems: Dataset, remaining_problems: Dataset):
        self.logger.info("Saving samples and logs")
        if (len(solved_problems) > 0):
            solved_problems.save_to_disk(self.config['save_path'] + "/solved")
        if (len(remaining_problems) > 0):
            remaining_problems.save_to_disk(self.config['save_path'] + "/remaining")
        with open(self.config['save_path'] + "/config.yaml", "w") as config_file:
            yaml.safe_dump(self.config, config_file, default_flow_style=False)

    def cleanup(self):
        # This function will cleanup the vllm engine if we do not use it anymore.
        destroy_model_parallel()
        destroy_distributed_environment()
        del self.llm.llm_engine.model_executor
        del self.llm
        with contextlib.suppress(AssertionError):
            torch.distributed.destroy_process_group()
        gc.collect()
        torch.cuda.empty_cache()
        ray.shutdown()
        self.logger.info("Successfully destroyed vllm engine.")
