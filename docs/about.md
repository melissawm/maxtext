# What is Maxtext?

MaxText is a Google initiated open source project for high performance, highly scalable, open-source LLM written in pure Python/[JAX](https://jax.readthedocs.io/en/latest/index.html) and targeting Google Cloud TPUs and GPUs for training and inference.

MaxText achieves very high MFUs (Model Flop Utilization) and scales from single host to very large clusters while staying simple and "optimization-free".

MaxText additionally provides an highly optimized reference implementations for popular Open Source models like:

- Llama 2, 3 and 3.1
- Mistral and Mixtral
- Gemma and Gemma2
- GPT

These reference implementations support pre-training and full fine tuning.  Maxtext also allows you to create various sized models for benchmarking purposes.

The key value proposition of using MaxText for pre-training or full fine tuning is:

- Very high performance of average of 50% MFU
- Open code base - Code base can be found at the following github location.
- Easy to understand: MaxText is purely written in JAX and Python, which makes it accessible to ML developers interested in inspecting the implementation or stepping through it. It is written at the block-by-block level, with code for Embeddings, Attention, Normalization etc. Different Attention mechanisms like MQA and GQA are all present. For quantization, it uses the JAX AQT library. The implementation is suitable for both GPUs and TPUs.

MaxText aims to be a launching off point for ambitious LLM projects both in research and production. We encourage users to start by experimenting with MaxText out of the box and then fork and modify MaxText to meet their needs.

!!! note

    Maxtext today only supports Pre-training and Full Fine Tuning of the models. It does not support PEFT/LoRA, Supervised Fine Tuning or RLHF

## Who is the target user of Maxtext?

- Any individual or a company that is interested in forking maxtext and seeing it as a reference implementation of a high performance Large Language Models and wants to build their own LLMs on TPU and GPU.
- Any individual or a company that is interested in performing a pre-training or Full Fine Tuning of the supported open source models, can use Maxtext as a blackbox to perform full fine tuning. Maxtext attains an extremely high MFU, resulting in large savings in training costs.
