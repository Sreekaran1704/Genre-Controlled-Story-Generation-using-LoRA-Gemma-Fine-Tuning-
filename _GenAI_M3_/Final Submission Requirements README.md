# Genre-Controlled Short Story Generation Using QLoRA Adaptation

## Project Overview

This project builds an end-to-end genre-controlled short story generation system. Given a genre and a story prompt, the system generates a short story aligned with the requested genre.

The project compares two systems:

1. **Base system**: the original pretrained `google/gemma-3-1b-it` model without adaptation.
2. **Adapted system**: the same `google/gemma-3-1b-it` model after QLoRA fine-tuning.

The target genres are:

- Fantasy
- Romance
- Science fiction

The main goal is to evaluate whether QLoRA-based parameter-efficient fine-tuning improves the model's performance on genre-conditioned story generation under limited compute resources.

---

## Final Results Summary

| System | Adaptation | Evaluation Set | Validation Loss | Perplexity |
|---|---|---|---:|---:|
| Base Gemma-3-1B-it | None | Same held-out evaluation set | 3.8268 | 45.9134 |
| QLoRA-adapted Gemma-3-1B-it | QLoRA | Same held-out evaluation set | 3.3188 | 27.6261 |

The QLoRA-adapted model improved validation loss and perplexity compared with the base model. However, LLM-as-judge evaluation showed mixed qualitative results: the base model scored higher on the small judged sample, which suggests that better token-level metrics do not always guarantee better creative writing quality.

---

## Dataset

The project uses a processed subset of the WritingPrompts dataset, originally collected from a Reddit creative writing community and accessed through Kaggle.

The full WritingPrompts dataset contains approximately 272K prompt-story pairs. For this project, a smaller filtered genre-controlled subset was created.

Final processed dataset size:

| Split | Number of Examples |
|---|---:|
| Train | 567 |
| Validation | 70 |
| Test | 72 |
| Total | 709 |

Each example follows this general structure:

```text
<GENRE> fantasy

<PROMPT>
[story prompt]

<STORY>
[target story]
```

---

## Preprocessing

The preprocessing pipeline cleaned and filtered raw story examples before training.

Main preprocessing steps:

- Unicode normalization using NFKC.
- Replacement of `<newline>` markers with real newline characters.
- Standardization of carriage returns and line breaks.
- Whitespace cleanup while preserving paragraph breaks.
- Removal of deleted, removed, or empty stories.
- Removal of prompt artifacts and meta-writing language.
- Filtering stories shorter than 180 words or longer than 450 words.
- Checking that stories end with a clean final sentence.
- Removing stories with repeated full sentences.

These steps were used to make the dataset more suitable for short story generation.

---

## Model Details

Base model:

```text
google/gemma-3-1b-it
```

Tokenizer:

```text
AutoTokenizer loaded from google/gemma-3-1b-it
```

Task:

```text
Causal language modeling for genre-controlled short story generation
```

Maximum sequence length:

```text
SFT_MAX_LENGTH = 1024
```

Generation length:

```text
max_new_tokens ≈ 300
```

---

## QLoRA Configuration

The adapted model uses QLoRA for parameter-efficient fine-tuning.

QLoRA settings:

| Setting | Value |
|---|---|
| Quantization | 4-bit NF4 |
| Double quantization | Yes |
| Compute dtype | bfloat16 |
| LoRA rank | 16 |
| LoRA alpha | 16 |
| LoRA dropout | 0.10 |
| Target modules | `self_attn.q_proj`, `self_attn.v_proj` |
| Optimizer | `paged_adamw_32bit` |
| Learning rate | `5e-5` |
| Batch size | 1 |
| Gradient accumulation steps | 4 |
| Max sequence length | 1024 |

Only the LoRA adapter parameters were trained. The base model weights remained frozen.

---

## Evaluation

The comparison between the base and QLoRA-adapted systems was designed to be fair.

Both systems used:

- Same task definition.
- Same held-out evaluation set.
- Same prompt format.
- Same tokenizer.
- Same generation settings.
- Same quantitative metrics.
- Same qualitative evaluation procedure.

### Quantitative Metrics

The main quantitative metrics were:

- Validation loss
- Perplexity

### Qualitative Evaluation

Qualitative review considered:

- Genre fidelity
- Prompt relevance
- Coherence
- Ending quality
- Repetition
- Style quality
- Format following
- Overall quality

An LLM-as-judge setup was also used for sample outputs. The evaluator model was:

```text
llama-3.1-8b-instant
```

The judge was accessed through the Groq API and used as a supporting evaluation tool, not as the only source of judgment.

---

## LLM-as-Judge Summary

| Model | Genre Fidelity | Prompt Relevance | Coherence | Ending Quality | No Repetition | Style Quality | Format Following | Overall Score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | 4.167 | 4.000 | 4.500 | 3.500 | 4.500 | 4.500 | 5.000 | 3.767 |
| QLoRA | 3.167 | 3.500 | 3.500 | 2.500 | 3.500 | 3.667 | 4.167 | 3.000 |

These results show that although QLoRA improved validation loss and perplexity, it did not consistently improve perceived story quality in the small LLM-as-judge sample.

## Environment Setup

Recommended environment:

```text
CUDA-enabled GPU recommended
Google Colab GPU or similar environment
```

Run in Google Colab, in the order of the cells in the notebook.

---

## Main Libraries

The main libraries used in this project are:

```text
torch
transformers
datasets
peft
trl
accelerate
bitsandbytes
pandas
numpy
matplotlib
scikit-learn
groq
streamlit
```

---

## How to Run the Project

Run the notebooks.

### 1. Data Preprocessing

```text
code/data_preprocessing.ipynb
```

This notebook loads and filters the WritingPrompts data, creates the genre-controlled subset, and saves the train/validation/test splits.

Expected outputs:

```text
processed dataset
```
But as I had used manual annaotation of genres you may require a lot of time to annotate the data, so I had added the annotated and cleaned data in the data folder


### Remaining total part

```text
code/model_building_and_QLoRA_finetuning.ipynb
```

This notebook loads all the needed things you want for the project. And run in the order of the cells.

## Streamlit Demo

A Streamlit web application was created as the demo interface.

The app allows users to:

- Select a genre.
- Enter a story prompt.
- Generate a story.
- Inspect model outputs.
- View the end-to-end workflow interactively.

To run the app:
Open this notebook

```text
code/making_streamlit_mini_app.ipynb
```
And run the code in the order of cells by following the instructions available in the notebook.

