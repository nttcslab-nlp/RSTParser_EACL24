# RST Parsing with Llama2

Implementation of "Can we obtain significant success in RST discourse parsing by using Large Language Models?" accepted in the main conference of EACL 2024. Our main parser, which fine-tuned Llama 2 (70B) with QLoRA based on the prompts, significantly outperformed current state-of-the-art parsers for three benchmark datasets, RST-DT, Instr-DT, and the GUM corpus.

**Paper**: [[arXiv]](https://arxiv.org/abs/2403.05065), [[EACL 2024]](https://aclanthology.org/2024.eacl-long.171/)

**Abstract**:
Recently, decoder-only pre-trained large language models (LLMs), with several tens of billion parameters, have significantly impacted a wide range of natural language processing (NLP) tasks. While encoder-only or encoder-decoder pre-trained language models have already proved to be effective in discourse parsing, the extent to which LLMs can perform this task remains an open research question. Therefore, this paper explores how beneficial such LLMs are for Rhetorical Structure Theory (RST) discourse parsing. Here, the parsing process for both fundamental top-down and bottom-up strategies is converted into prompts, which LLMs can work with. We employ Llama 2 and fine-tune it with QLoRA, which has fewer parameters that can be tuned. Experimental results on three benchmark datasets, RST-DT, Instr-DT, and the GUM corpus, demonstrate that Llama 2 with 70 billion parameters in the bottom-up strategy obtained state-of-the-art (SOTA) results with significant differences. Furthermore, our parsers demonstrated generalizability when evaluated on RST-DT, showing that, in spite of being trained with the GUM corpus, it obtained similar performances to those of existing parsers trained with RST-DT.

**Pre-trained weights**: [[Hugging Face Hub]](https://huggingface.co/collections/arumaekawa/rst-parser-with-llama-2-660cf1bf5dcbe4ca96541a42)

## Results

F1 scores of fully-labeled spans for RST-DT, Instr-DT, and the GUM corpus.

| Strategy      | Method           | Model             |   RST-DT | Instr-DT |      GUM |
| :------------ | :--------------- | :---------------- | -------: | -------: | -------: |
| Top-down      | Kobayashi et al. | DeBERTa (140M)    |     54.4 |     43.4 |     48.7 |
| Top-down      | Ours             | Llama 2 (70B)     |     56.0 |     45.2 |     54.8 |
| Bottom-up     | Kobayashi et al. | DeBERTa (140M)    |     55.4 |     44.4 |     48.5 |
| **Bottom-up** | **Ours**         | **Llama 2 (70B)** | **58.1** | **47.3** | **55.2** |

## Scripts

### Preprocess

```
$ python src/preprocess.py --corpus rstdt --save_dir preprocessed_data
```

#### Data type

| key            | value                                                  |
| -------------- | ------------------------------------------------------ |
| `span`         | Action prediction for shift-reduce parsing             |
| `nuc`          | Nucleus label prediction                               |
| `rel`          | Relation label prediction                              |
| `rel_with_nuc` | Relation label prediction with predicted nucleus label |
| `top_down`     | Split point prediction for top-down                    |

### Train

```
$ bash scripts/general/train.sh rstdt 7b span
$ bash scripts/general/train.sh rstdt 7b nuc
$ bash scripts/general/train.sh rstdt 7b rel-with-nuc
```

Note: `train.py` is used from the official implementation of QLoRA (https://github.com/artidoro/qlora).

### Checkpoint Selection

```
$ bash scripts/general/test_with_oracle.sh rstdt 7b span
$ bash scripts/general/test_with_oracle.sh rstdt 7b nuc
$ bash scripts/general/test_with_oracle.sh rstdt 7b rel-with-nuc
```

### Test

```
$ bash scripts/general/test.sh rstdt rstdt 7b bottom_up
```

Test our models ([Hugging Face Hub](https://huggingface.co/collections/arumaekawa/rst-parser-with-llama-2-660cf1bf5dcbe4ca96541a42)):

```
bash scripts/general/test_public_model.sh rstdt rstdt 7b bottom_up
```
