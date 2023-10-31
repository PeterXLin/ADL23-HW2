
# Q1: Model

## Model
- Describe the **model architecture** and **how it works on text summarization.**   



## Preprocessing
- Describe your preprocessing (e.g. **tokenization, data cleaning** and etc.)


# Training 

## Hyperparameter
- Describe your hyperparameter you use and **how you decide it**.

### learning rate
[t5 learning rate](https://wandb.ai/lukaemon/finetune-t5-hello-world/reports/T5-learning-rate--VmlldzozNTI3ODM1)
for learning rate 3e-4, train 5 epoch is enough

## Learning Curves
- Plot the learning curves (**ROUGE versus training steps**)

# Generation Strategies

[Text generation strategies](https://huggingface.co/docs/transformers/generation_strategies)
[how-to-generate](https://huggingface.co/blog/how-to-generate)

## strategies
- **Describe the detail of the following generation strategies**:
    - Greedy
        - 每次decode時就直接選機率最高的字
    - Beam Search
        - 生成時不是只選當前機率最高的word，而是會同時紀錄K條sequence(每次選機率最高的K個word)。比起greedy strategies來說，比較有考慮global的狀況
    - Top-k Sampling
        - 在decode時，選擇ramdomly sample the word via the probability distribution，但是只從機率最高的k個word中sample(避免long tail distribution的問題)
    - Top-p Sampling
        - 也是ramdomly sample，但跟top-k sampling 不同的是，top-p sampleing candidate的數量是會變的，決定candidate set的方式是set裡面所有candidate機率和有沒有超過p，如果沒有，就可以繼續選機率最高的word加入candidate set，如果超過p了，就不再往candidate set裡加word。最後從candidate set 裡面sample一個word出來。
    - Temperature
        - 算softmax時額外除temperature，目的是控制decode的diversity (high temperature: more diversity, low temperature: less diversity)

## Hyperparameters
- **Try at least 2 settings of each strategies and compare the result**. 
- What is your **final generation strategy**? (you can combine any of them)
beam search seems better
## different decoding strategy (based mt5 small 3e-4 epoch 9)
| strategy | rouge-1 f1 | rouge-2 f1 | rouge-l f1 |
| ---------- | -------- | ---------- | ---------- | 
|  greedy           |    25.2    |     9.4       |      22.5      |
|  5 beams          |    26.4    |     10.5      |      23.4      |  
|  5 beams sampling |    25.8    |    10.1       |      22.8      |
|  sampling(t = 0.6)|    26      |    10.1       |     22.9       |
|  top k (50)       |    20.8    |    6.8        |     18.4       |
|  top p (92)       |    19.6    |    6.5        |     17.4       |

=> low diversity seems to be better


[mt5 finetune](https://github.com/KrishnanJothi/MT5_Language_identification_NLP#dp)

# Result 

## different hyperparameter mt5-small model use beam search
| lr | warmup | max s len | rm newline | epoch | rouge-1 f1 | rouge-2 f1 | rouge-l f1 |
| -- | ------ | --------- | ---------- | ----- | ---------- | ---------- | ---------- |
| 3e-4 |   0   | 1024 |  X | 9     |   28.09   |     11.1      |     24.6     | 
| 3e-4 |   0   | 512  |  O | 10(15)|   28.3    |     11.1      |     24.7     |
| 3e-4 |   0   | 512  |  O | 10(10)|   28.3    |     11.2      |     24.7     |
 



# Bonus: Applied RL on Summarization
##  Algorithm 
- **Describe your RL algorithms, reward function, and hyperparameters.**

## Compare to Supervised Learning 
- Observe the loss, ROUGE score and output texts, what differences can you find?

