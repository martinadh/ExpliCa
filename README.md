# ExpliCa
## Evaluating Explicit Causal Reasoning in Large Language Models

ExpliCa (Explicit Causality) is a dataset designed to evaluate LLMs on commonsense causal reasoning through causal discovery tasks, more specifically via Pairwise Causal Discovery (PCD), which focuses on determining the existence of a causal relationship between two events and establishing the causal direction, that is, identifying which event serves as the cause and which as the effect.
In ExpliCa PCD is formulated so that allows to take into account also the entanglement of causal and temporal relations between events. 

This repository contains
* The dataset with human annotation and models' answers
* The results
* The code used to:
  * Query the LLMs
  * Analyze and plot the results 

## The dataset
ExpliCa contains 600 sentence pairs, each presented in both possible orders. Each sentence pair has been joined in a single sentence through connectives that explicitly indicate causal or temporal relations, as well as the direction of those relations, i.e., iconic (i.e., when the cause/antecedent event in sentence A linguistically appears  before  the effect/the subsequent event in sentence B) and and anti-iconic (effect-cause; subsequent-antecedent):
* A causes B > “so”
* B causes A > “because”
* A precede B > “then”
* B precede A > “after”

## Human ratings 
We collected human acceptability ratings concerning the connective in each sentence via crowdsourcing. Subjects rated on a scale from 1 to 10 how acceptable the connective is to express the relation between the events in the sentences.
For example:
* Jude walked under the rain for an hour, so Jude got sick: Rating 10 (highly acceptable)
* Mary bought some flowers, because Jean went to the dentist: Rating 1 (not acceptable)
Each sentence was rated by 15 English native speakers


## Structure
* Pair_ID: ID of the sentence pair
* Sentence_A: sentence A
* Sentence_B: sentence B
* rating_iconic_causal: the average rating given to sentence pairs connected by a causal connective, where the sentence order follows an iconic direction, i.e., A SO B.
* rating_anticonic_causal: the average rating given to sentence pairs connected by a causal connective, where the sentence order follows an anti-iconic direction, i.e., A BECAUSE B.
* rating_iconic_temporal: the average rating given to sentence pairs connected by a temporal connective, where the sentence order follows an iconic direction, i.e., A THEN B.
* rating_anticonic_temporal: the average rating given to sentence pairs connected by a temporal connective, where the sentence order follows an anti-iconic direction, i.e., A AFTER B.
* human_preferred_connective: connective for which humans provided the highest average rating
* human_preferred_connective_desc: indication of order (iconic vs anti-iconic) and relation (causal vs temporal) of the connective for which humans provided the highest average rating


# The Evaluation

Human ratings served as ground truth for the evaluation of seven LLMs: 
* Mistral-7B-Instruct-v0.3 
* falcon-7b-instruct
* Meta-Llama-3.1-8B-Instruct 
* gemma-2-9b-it 
* Qwen2.5-7B-Instruct (with also versions of 0.5, 1.5, 3, 7, 14, 32B)
* gpt4o 
* gpt4o-mini
  
The evaluation covers four key tasks:
* Three prompting tasks (Acceptability rating task, cloze test and multiple-choice) under different conditions (Few-shot and zero-shot setups, Greedy search vs. the Outlines framework) for response generation
* Perplexity evaluation

For performance assessment, we used accuracy as the primary metric.

Finally, we analyzed the models' acceptability rating distributions and compared them to human ratings, assessing their correlation with human judgment.

This repository provides insights into model behavior and evaluation methodologies, offering valuable benchmarks for future research.
