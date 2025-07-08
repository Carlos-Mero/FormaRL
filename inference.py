from datasets import Dataset, load_dataset, concatenate_datasets, load_from_disk

import logging

from pipeline.pipe import RTPipeline
from utils import load_jsonl

def load_problems(config: dict) -> Dataset:
    """
    This functions loads the natural language problems to be translated into lean4.
    It loads datasets according to the config dictionary and concatenate them into a single huggingface dataset.
    The output dataset contains two columns: "nlp", where nlp is the original problem in natural language. And "gt_flp" which is the ground truth translation of the original problem if exists, otherwise it will be an empty string.
    """
    logger = logging.getLogger(__name__)
    dss = []
    for dsname in config['datasets']:
        logger.info(f'Preparing dataset at path {dsname}')
        columns_to_keep = []
        if dsname == "pkuAI4M/minif2f-lean4-normalized":
            ds = load_dataset(dsname)
            ds = concatenate_datasets([ds['validation'], ds['test']])
            columns_to_keep = ["informal_statement", "formal_statement"]
            removed_columns = [col for col in ds.column_names if col not in columns_to_keep]
        elif dsname == "datasets/proofnet.jsonl":
            ds = Dataset.from_list([e for e in load_jsonl(dsname)])
            columns_to_keep = ["informal_prefix", "formal_statement"]
            removed_columns = [col for col in ds.column_names if col not in columns_to_keep]
        elif dsname == "datasets/math500.jsonl":
            ds = Dataset.from_list([e for e in load_jsonl(dsname)])
            columns_to_keep = ["problem", "answer"]
            removed_columns = [col for col in ds.column_names if col not in columns_to_keep]
        elif dsname == "datasets/uproof":
            ds = load_from_disk(dsname)
            def filter_proof(e):
                return e['type'] == "proof"
            ds = ds.filter(filter_proof)
            columns_to_keep = ["nl_statement"]
            removed_columns = [col for col in ds.column_names if col not in columns_to_keep]
        elif dsname == "internlm/Lean-Workbook":
            ds = load_dataset(dsname)
            ds = ds['train']
            columns_to_keep = ['natural_language_statement', 'formal_statement']
            removed_columns = [col for col in ds.column_names if col not in columns_to_keep]
        else:
            raise NotImplementedError("Unknown dataset name")

        print(f"removing columns:\n{removed_columns}")
        ds = ds.remove_columns(removed_columns)

        if dsname == "pkuAI4M/minif2f-lean4-normalized":
            ds = ds.rename_column("informal_statement", "nlp")
            ds = ds.rename_column("formal_statement", 'gt_flp')
            dss.append(ds)
        elif dsname == "datasets/proofnet.jsonl":
            ds = ds.rename_column("informal_prefix", "nlp")
            ds = ds.rename_column("formal_statement", 'gt_flp')
            dss.append(ds)
        elif dsname == "datasets/math500.jsonl":
            nlps = [f"For problem\n\n{e['problem']}\n\nShow that the answer is {e['answer']}." for e in ds]
            ds = ds.add_column('nlp', nlps)
            ds = ds.remove_columns(['problem', 'answer'])
            ds = ds.add_column('gt_flp', [''] * len(nlps))
            dss.append(ds)
        elif dsname == "datasets/uproof":
            if 'subset-size' in config.keys():
                # select a subset when testing
                ds = ds.shuffle(seed = config['seed'])
                ds = ds.select(range(config['subset-size']))
            nds = [{'nlp': f"{e['nl_statement']}."} for e in ds]
            nds = Dataset.from_list(nds)
            nds = nds.add_column('gt_flp', [''] * len(nds))
            dss.append(nds)
        elif dsname == "internlm/Lean-Workbook":
            ds = ds.rename_column('natural_language_statement', 'nlp')
            ds = ds.rename_column('formal_statement', 'gt_flp')
            return ds.to_list()
        else:
            raise NotImplementedError("The preprocess is not defined for this dataset")
    ds = concatenate_datasets(dss)
    return ds

def translate_rtpipeline(config: dict):
    ds = load_problems(config)
    pipe = RTPipeline(
        config=config,
        model=config['model'],
        model_init_params=config['model_init_params'],
        sampling_params=config['sampling_params']
    )
    pipe.save_logs(*pipe(ds))
