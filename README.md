# LEAP

This repository contains the implementations of the paper: LEAP: Efficient and Automated Test Method for NLP Software.

## Datesets
There are three datasets used in our experiments:

- [IMDB](https://s3.amazonaws.com/fast-ai-nlp/imdb.tgz)
- [AG's News](https://s3.amazonaws.com/fast-ai-nlp/ag_news_csv.tgz)
- [Poem Sentiment](https://github.com/google-research-datasets/poem-sentiment)


## Requirements
The code was tested with:

- bert-score>=0.3.5
- editdistance
- flair
- filelock
- language_tool_python
- lemminflect
- lru-dict
- datasets
- nltk
- numpy>=1.19.2
- pandas>=1.0.1
- scipy>=1.4.1
- torch>=1.7.0,!=1.8
- transformers>=3.3.0
- terminaltables
- tqdm>=4.27,<4.50.0
- word2number
- num2words
- more-itertools
- PySocks!=1.5.7,>=1.5.6

## How to Run:
Follow these steps to run the attack from the library:

1. Fork this repository

2. Run the following command to install it.

   ```bash
   $ cd TextAttack
   $ pip install -e . ".[dev]"
   
3. Run notebook **leap_demo.ipynb** to test the fine-tuned "bert-base-uncased" model on the "AG's News" dataset

Take a look at the `models` directory in [TextAttack](https://github.com/RishabhMaheshwary/TextAttack/tree/hard_label_attack) to run the attack across any dataset and any target model.


## Experimental result of LEAP on the query times
![image](https://github.com/lumos-xiao/LEAP/blob/main/query-time.png)


## Acknowledgement

This code is based on the [TextAttack](https://github.com/QData/TextAttack) framework.
