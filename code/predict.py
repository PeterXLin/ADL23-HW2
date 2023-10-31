import argparse
import os

import datasets
import nltk
import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
)

import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"]="1"

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a multiple choice task")
    parser.add_argument(
        "--test_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="The directory where model config, tokenizer config, model bin saved.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=384,
        help="",
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=64,
        help="",
    )
    parser.add_argument(
        "--source_prefix",
        type=str,
        default = "summarize: ",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default = None,
        help = "if use facebook mbart model, need to pass lang",
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_seq_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--temperature",
        type=int,
        default = 1,
        help="modify temperature to controll generation diversity",
    )
    parser.add_argument(
        "--decoding_strategy",
        type=str,
        default = "greedy",
        help="decoding strategy",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()        
    df_prediction = summarize(args)
    df_prediction.to_json(args.output_path, index=False, orient='records', lines=True)
    return 

def get_generate_kwargs(strategy):
    kwargs = {
        "greedy": {

        },
        "beam_search": {
            "num_beams": 5,
            "early_stopping": True,
            "no_repeat_ngram_size": 2
        },
        "beam_search_sampling": {
            "num_beams": 5,
            "early_stopping": True,
            "no_repeat_ngram_size": 2,
            "do_sample": True,
        },
        "sampling": {
            "do_sample": True,
            "top_k": 0,
            "temperature": 0.6,
        },
        "top_k_sampling": {
            "do_sample": True,
            "top_k": 50,
        },
        "top_p_sampling": {
            "do_sample": True,
            "top_p": 0.92,
            "top_k": 0
        }
    }
    return kwargs[strategy]

def summarize(args):

    if args.model_path != None:
        model_path = os.path.join(os.getcwd(), args.model_path)
    else:
        model_path = "/nfs/nas-6.1/whlin/ADL/ADL23-HW2/checkpoint/google_mt5_small_3e-4/checkpoint-23202"

    print("summary_model_path: ", model_path)
    # -------------------------- prepare dataset

    # load raw dataset
    raw_datasets = load_dataset("json", data_files={"test": args.test_path})
    # raw_datasets["test"] = raw_datasets["test"].select(range(10))

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    if args.lang is not None:
        model.config.decoder_start_token_id = tokenizer.lang_code_to_id[args.lang]

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    prefix = args.source_prefix if args.source_prefix is not None else ""
    column_names = raw_datasets["test"].column_names

    padding = "max_length" if args.pad_to_max_length else False
    text_column = "maintext"

    def preprocess_function(examples):
        inputs = examples[text_column]
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=args.max_source_length, padding=padding, truncation=True)
        return model_inputs

    def postprocess_text(preds):
        preds = [pred.strip() for pred in preds]
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        return preds


    test_dataset = raw_datasets["test"].map(
        preprocess_function,
        batched=True,
        remove_columns=column_names,
        desc="Running tokenizer on dataset",
    )

    
    label_pad_token_id = -100 
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=None,
    )

    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=args.batch_size)

    # ----------------- prepare model --------------
    # load model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu");
    model.to(device)

    # --------------- predict ---------------
    model.eval()

    gen_kwargs = get_generate_kwargs(args.decoding_strategy)

    gen_kwargs["max_length"] = args.max_target_length
    gen_kwargs['temperature'] = args.temperature

    all_prediction = list()
    for batch in tqdm(test_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            generated_tokens = model.generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                **gen_kwargs,
            )

            generated_tokens = generated_tokens.detach().cpu().numpy()

            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

            decoded_preds= postprocess_text(decoded_preds)
            all_prediction.extend(decoded_preds)

    df = pd.DataFrame(list(zip(raw_datasets['test']['id'], all_prediction)),
               columns =['id', 'title'])
    
    punctuation_mapping = {
        ord('.'): '。',
        ord(','): '，',
        ord('?'): '？',
        ord('!'): '！',
        ord(':'): '：',
        ord(';'): '；',
        ord('\''): '‘',
        ord('\"'): '“',
        ord('('): '（',
        ord(')'): '）',
        ord('['): '【',
        ord(']'): '】',
        ord('{'): '｛',
        ord('}'): '｝',
        ord('<'): '《',
        ord('>'): '》',
        ord('@'): '＠',
        ord('#'): '＃',
        ord('$'): '＄',
        ord('%'): '％',
        ord('&'): '＆',
        ord('-'): '－',
        ord('_'): '＿',
        ord('='): '＝',
        ord('+'): '＋',
        ord('/'): '／',
        ord('\\'): '＼',
        ord('|'): '｜',
        ord('^'): '＾',
        ord('~'): '～'
    }

    df['title'] = df.apply(lambda x: x['title'].translate(punctuation_mapping), axis = 1)

    return df

if __name__ == "__main__":
    main()