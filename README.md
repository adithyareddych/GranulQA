GranulQA
========

GranulQA is a Retrieval-Augmented Question Answering library built on top of RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval). It combines the recursive tree-based indexing of RAPTOR with a lightweight QA pipeline for efficient, context-aware question answering over large documents.

[![Paper page](https://huggingface.co/datasets/huggingface/badges/resolve/main/paper-page-sm.svg)](https://huggingface.co/papers/2401.18059)  [![PapersWithCode SOTA](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/raptor-recursive-abstractive-processing-for/question-answering-on-quality)](https://paperswithcode.com/sota/question-answering-on-quality?p=raptor-recursive-abstractive-processing-for)

## Prerequisites

- **Python** 3.8 or higher  
- **Git** 2.x  
- An OpenAI API key (for default summarization and QA models)  
- *(Optional)* Jupyter Notebook or JupyterLab for running the demo.

## Installation

1. **Clone the GranulQA repository**:
   ```bash
   git clone https://github.com/teja00/GranulQA.git granulqa
   cd granulqa
   ```

2. **Install GranulQA dependencies** (and link to RAPTOR):
   ```bash
   pip install -r requirements.txt
   pip install -e ../raptor-core    # install core RAPTOR package in editable mode
   ```

3. **Configure your OpenAI API key**:
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   ```

## Basic Usage

1. **Initialize the RAPTORLLM pipeline**:
   ```python
   from raptor.qa_pipeline import RAPTORLLM

   raptor = RAPTORLLM()
   ```

2. **Index one or more documents**:
   ```python
   with open('sample.txt', 'r') as f:
       text = f.read()
   raptor.index_corpus([text])
   ```

3. **Answer a question**:
   ```python
   question = "How did Cinderella reach her happy ending?"
   answer = raptor.answer_question(question)
   print("Answer:", answer)
   ```

4. **Save and reload the retrieval tree**:
   ```python
   # Save tree to disk
   raptor.save_tree('demo/cinderella_tree')

   # Load later
   from raptor.qa_pipeline import RetrievalAugmentation
   ra = RetrievalAugmentation(tree_path='demo/cinderella_tree')
   answer2 = ra.answer_question(question)
   ```

## Running the Demo Notebook

We include `demo.ipynb` illustrating basic and advanced usage of GranulQA. To run:

1. Install Jupyter if needed:
   ```bash
   pip install jupyterlab
   ```
2. Launch:
   ```bash
   jupyter lab demo.ipynb
   ```
3. **Note**: the **last cell** of `demo.ipynb` contains our custom GRANULQA implementation and output examples.

## Extending with Custom Models

GranulQA supports plugging in your own summarization, QA, and embedding models by subclassing the respective base classes which is exactly extended from the raptor paper:

- **Custom Summarizer**: extend `raptor.base.BaseSummarizationModel` and implement `summarize(context, max_tokens)`.
- **Custom QA**: extend `raptor.base.BaseQAModel` and implement `answer_question(context, question)`.
- **Custom Embedding**: extend `raptor.base.BaseEmbeddingModel` and implement `create_embedding(text)`.

Refer to the examples in the repository for implementation templates and configuration.

## RAPTOR Citation

Please also cite the core RAPTOR paper this project builds upon:

```bibtex
@inproceedings{sarthi2024raptor,
  title={RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval},
  author={Sarthi, Parth and Abdullah, Salman and Tuli, Aditi and Khanna, Shubh and Goldie, Anna and Manning, Christopher D.},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2024}
}
```

