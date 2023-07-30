<h1 align="center">LEAP</h1>


<p align="center">
<b>
An efficient and automated test method for NLP software.</b>

<p align="center">
Code release and supplementary materials for:</br>
  <b>"LEAP: Efficient and Automated Test Method for NLP Software"</b></br>
</p>

## Datesets
There are three datasets used in our experiments:

- [IMDB](https://s3.amazonaws.com/fast-ai-nlp/imdb.tgz)
- [AG's News](https://s3.amazonaws.com/fast-ai-nlp/ag_news_csv.tgz)
- [Poem Sentiment](https://github.com/google-research-datasets/poem-sentiment)

## Repo structure
- `datasets`: define the dataset object used for carrying out attacks
- `goal_functions`: determine if the test method generates successful test cases
- `search_methods`: explore the space of potential transformations and try to locate a successful perturbation
- `transformations`: transform the input text, e.g. synonym replacement
- `constraints`: determine whether or not a given transformation is valid

The most important files in this project are as follows:
- `victim models.zip`: victim models obtained by training on three datasets
- `search_methods/leap.py`: search test cases based on PSO
- `attack_recipes/leap_2023.py`: code to execute LEAP in the TextAttack framework
- `leap_demo.ipynb`: an example of testing the fine-tuned "bert-base-uncased" model on the "AG's News" dataset


## Dependencies
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
   $ pip install -e . ".[dev]"
   
3. Run notebook **leap_demo.ipynb** to test the fine-tuned "bert-base-uncased" model on the "AG's News" dataset

Take a look at the `models` directory in [TextAttack](https://github.com/RishabhMaheshwary/TextAttack/tree/hard_label_attack) to run the attack across any dataset and any target model.


## Supplementary
### Experimental result of LEAP on the query times
In RQ2 of Section V EXPERIMENT RESULTS AND ANALYSIS, we measure the efficiency of the test method based on the time overhead and the query times, and due to space constraints, we report the experimental results for the query times in the repository.


![image](https://github.com/lumos-xiao/LEAP/blob/main/query-time.png)

Similar to the experiment on time overhead, the table presents the average query times per successful generation of a test case on the long-text datasets IMDB and AG's News. Compared to existing heuristic testing methods, LEAP demonstrates suboptimal performance in terms of query times. Although IGA has lower query times than LEAP, its time overhead is higher across all victim models, and its success rate in generating test cases is also lower than LEAP.



## License
This code and model are available for non-commercial scientific research purposes as defined in the [LICENSE](LICENSE) file. By downloading and using the code and model you agree to the terms in the [LICENSE](LICENSE).


## Acknowledgement

This code is based on the [TextAttack](https://github.com/QData/TextAttack) framework.
