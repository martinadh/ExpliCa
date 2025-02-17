# Model evaluation

## Query

The query folder contains the python script to query the open models and Gpt models. Variables are at the beginning of the script. Among those GPT API KEY and Hugging Face Token must be inserted.
The provided code is only for few-shot. It is sufficient to comment the "Examples" part of the prompt template to query the model in zero-shot setting.


## Answer analysis

The answer analysis consists in two steps:
* Answer cleaning: must be applied to answers return by the models in the greedy search setup
* outline_accuracy and greedy_accuracy, allow for computing accouracy in these two setups along with perplexity.

The code for the analysis in provided in Jupyter Notebooks.
