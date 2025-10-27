# Hierarchical Transformer Encoders for Vietnamese Spelling Correction (Unofficial Implementation)

This repository provides an **unofficial PyTorch implementation** of the
paper *Hierarchical Transformer Encoders for Vietnamese Spelling
Correction* by Tran et al. (2021).

------------------------------------------------------------------------

## 📦 Dependencies

Ensure the following packages are installed before running the code:

-   [PyTorch](https://pytorch.org/)
-   [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/)
-   [NLTK](https://www.nltk.org/)
-   Transformer (compatible version)

You can install the required dependencies using:

``` bash
pip install torch pytorch-lightning nltk transformers
```

------------------------------------------------------------------------

## 🏋️‍♂️ Training Pipeline

### 1. Train the Word-Level Tokenizer

Prepare a Vietnamese text corpus and execute the following command to
train the tokenizer:

``` bash
python -m models.word_char_tokenizer
```

### 2. Prepare the Baseline Model

Initialize the baseline model setup using:

``` bash
python -m models.baseline
```

### 3. Train the Main Model

Modify your training parameters as needed inside `main.py`, then run:

``` bash
python main.py
```

### 4. Distributed Training (Multi-GPU or Cluster)

To train across multiple nodes or GPUs using PyTorch Lightning's
distributed setup:

``` bash
# Reference: https://pytorch-lightning.readthedocs.io/en/latest/clouds/cluster_intermediate_1.html

export MASTER_PORT=...
export MASTER_ADDR=...
export WORLD_SIZE=2
export NODE_RANK=1

# Enable NCCL debugging and specify network interface
NCCL_SOCKET_IFNAME="enp3s0" NCCL_DEBUG=INFO python main.py
```

------------------------------------------------------------------------

## 📚 Reference

If you use or refer to this work, please cite the original paper:

``` bibtex
@misc{https://doi.org/10.48550/arxiv.2105.13578,
  doi = {10.48550/ARXIV.2105.13578},
  url = {https://arxiv.org/abs/2105.13578},
  author = {Tran, Hieu and Dinh, Cuong V. and Phan, Long and Nguyen, Son T.},
  keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences},
  title = {Hierarchical Transformer Encoders for Vietnamese Spelling Correction},
  publisher = {arXiv},
  year = {2021},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```

------------------------------------------------------------------------

## 📝 License

This implementation is released under the [Apache License
2.0](http://www.apache.org/licenses/LICENSE-2.0).
