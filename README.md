# ADL23-HW2
Dataset & evaluation script for ADL 2023 homework 2

## Dataset
download datast and model by running this script
```
bash ./download.sh
```

## Usage

### train
```shell
bash ./script/train.sh
```

### Inference
```shell
bash ./run.sh ./data/public.jsonl ./data/predict.jsonl
```

### eval
```shell
cd tw_rouge/
python eval.py -r ../data/public.jsonl -s ../data/predict.jsonl
```