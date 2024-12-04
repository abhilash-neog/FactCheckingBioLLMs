# CS 6804 Final Project: Evaluating Model Reasoning and Hallucinations in Medical LLMs

This research study focuses on the challenges posed by hallucinations in medical large language models (LLMs), particularly in the context of the medical domain. We evaluate the reasoning abilities of popular open-source medical LLMs on a new benchmark and dataset, Med-HALT (Medical Domain Hallucination Test), designed specifically to evaluate hallucinations in medical domain. 
This benchmark provides a diverse multinational dataset derived from medical examinations across various countries and includes multiple innovative testing modalities. 

You can read more about our work here: [Evaluating Model Reasoning and Hallucinations in Medical LLMs Report](Trustworthiness_of_Medical_LLMs_Project_Report.pdf)

## Dataset

Datasets are hosted in Huggingface's [dataset](https://huggingface.co/datasets/MedHALT/Med-HALT)

## Inference Instructions

```python
python bio_llm.py <test_name> <batch> <model_name>
```
- Here test_name can be FCT, Nota or fake depending on which test we want to run
- Here model_name can be asclepius, alpacare, biomistral or pmc_llama depending on which model we want to run the inference for
- batch is the batch_size. We use different batch sizes like 4,8 and 12.

## Evaluation Instructions

Find the evaluation notebooks here: https://github.com/abhilash-neog/FactCheckingBioLLMs/tree/Evaluation_Codes

## Acknowledgements

We sincerely thank the authors of following open-source projects:

- [Med-Halt](https://github.com/medhalt/medhalt)
- [datasets](https://github.com/huggingface/datasets)
- [transformers](https://github.com/huggingface/transformers)

