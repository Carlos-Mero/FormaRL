import argparse
from datasets import load_from_disk
from typing import Iterable, Any

import numpy as np
import torch

import random
import json
import re

def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def remove_lean_comments(code):
    # Remove single-line comments (-- ...)
    code = re.sub(r'--.*', '', code)
    # Remove multi-line comments (/- ... -/)
    code = re.sub(r'/-.*?-/', '', code, flags=re.DOTALL)
    # Remove any leading/trailing whitespace from each line
    # code = '\n'.join(line.strip() for line in code.splitlines() if line.strip())
    return code

def load_jsonl(file: str) -> Iterable[Any]:
    with open (file, "r", encoding='utf-8') as f:
        for line in f:
            try:
                yield json.loads(line)
            except:
                print(f"failed to load json: {line}")
                exit()

def remove_theorem_name(flp: str) -> str:
    """This function removes the name of the given theorem, which is better for training. It will return the modified string back."""
    pattern = r'theorem\s+\w+'
    return re.sub(pattern, 'example', flp)

def cleanup_proofs(example):
    index = (example['flp']).find('by')
    if index != -1:
        example['flp'] = example['flp'][:index] + "sorry"
    else:
        example['flp'] = example['flp'] + "sorry"
    return example

def find_boxed(pred_str: str):
    ans = pred_str.split("boxed")[-1]
    if not ans:
        return ""
    if ans[0] == "{":
        stack = 1
        a = ""
        for c in ans[1:]:
            if c == "{":
                stack += 1
                a += c
            elif c == "}":
                stack -= 1
                if stack == 0:
                    break
                a += c
            else:
                a += c
    else:
        a = ans.split("$")[0].strip()
    return a

def extract_code_block(text: str) -> str:
    """
    This function extracts the content of the last lean code block from the given string.
    """
    pattern = r"```(?:lean|lean4)\s*([\s\S]*?)\s*```"
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1]
    else:
        # returns empty string when no match is found
        return ""
    
def extract_latex_block(text:str) -> str:
    """
    This function extracts the content of LaTex format block from the given string.
    """
    pattern = r"```(?:latex)?\s*([\s\S]*?)\s*```" 
    matches = re.findall(pattern,text)
    if matches:
        return matches[-1]
    else:
        return ""

def formatted_output(e: dict):
    if 'data_type' not in e.keys() or e['data_type'] == 'trans':
        print("\033[1m\033[34mThe translation results are\033[0m\n\n")
        print(f"\033[1m\033[33mOriginal problem:\033[0m\n{e['nlp']}\n\n")
        print(f"\033[1m\033[32mTranslation process:\033[0m\n\n{e['rationle'] if 'rationle' in e.keys() else e['CoT']}")
        print(f"\033[1m\033[36mExtracted translated result:\033[0m\n{e['flp']}")
        print(f"\033[1m\033[32mGround truth translation:\033[0m\n{e['gt_flp']}")
        if 'errors' in e.keys():
            print(f"\033[1m\033[31mErrors:\033[0m\n{e['errors']}")
    elif e['data_type'] == 'scorrect':
        print(f"\033[1m\033[33mOriginal problem:\033[0m\n\n{e['nlp']}\n")
        print(f"\033[1m\033[32mTranslation process:\033[0m\n\n{e['CoT']}\n")
        print(f"\033[1m\033[36mTranslation Result:\033[0m\n\n{e['flp']}\n")
        print(f"\033[1m\033[32mGround Truth Translation:\033[0m\n\n{e['gt_flp']}\n")
        print(f"\033[1m\033[32mScorrect Process:\033[0m{e['rationle']}\n")
    else:
        raise NotImplementedError("Unknown data type for formatted output")

def read_markdown(file_path):
    with open(file_path,'r',encoding='utf-8') as f:
        text = f.read()
    return text

def spilt_text(text,chunk_size=5000):
    return [text[i:i+chunk_size] for i in range(0,len(text),chunk_size)]

def get_chunked_text(file_name):
    file_path = f"/data2/private/hyx/umath/{file_name}.md"
    chunked_text = []
    text = read_markdown(file_path)
    chunked_text.extend(spilt_text(text=text,chunk_size=10000))

    return chunked_text

def main():
    parser = argparse.ArgumentParser(description="utils for cott training scripts.")
    parser.add_argument('--view', type=str, default="", help="Set this argument to view the dataset at the path.")
    parser.add_argument('-n', '--nviews', type=int, default=2, help="The number of elements to view in the dataset, only works when --view is set.")
    parser.add_argument('-s', '--start', type=int, default=0, help="The initial problem index to view")
    parser.add_argument('-f', '--format', action='store_true', default=False, help="enable more readable formatted output, only used for translation outputs.")
    parser.add_argument('-o', '--output', type=str, default='', help='The path to save the concatenated dataset, required when passed cat argument')
    parser.add_argument('--beta_1', type=int, default=1, help='The weight of the compiler check in the reward')
    parser.add_argument('--beta_2', type=int, default=1, help='The weight of the consistancy check in the reward')

    args = parser.parse_args()

    if args.view != "":
        print(f"viewing dataset at path {args.view}")
        ds = load_from_disk(args.view)
        ds = ds.to_list()
        ds_split = ds[args.start:args.start+args.nviews]
        if args.format:
            for e in ds_split:
                formatted_output(e)
        else:
            print(ds_split)
        return
    

if __name__ == "__main__":
    main()
