
---

<h1 align="center">
Recursive Language Models
</h1>

## Overview
Recursive Language Models (RLMs) are a task-agnostic inference paradigm for language models (LMs) to handle near unbounded contexts by enabling the LM to *programmatically* examine, decompose, and recursively call itself over its input. RLMs replace the canonical `llm.completion(prompt, model)` call with a `rlm.completion(prompt, model)` call. RLMs offload the context as a variable in a REPL environment that the LM can interact with and launch sub-LM calls inside of.

This repository provides an extensible inference engine for using RLMs around standard API-based and local LLMs. The initial experiments and idea were proposed in a [blogpost](https://alexzhang13.github.io/blog/2025/rlm/) in 2025, with expanded results in an [arXiv preprint](...).

## Installation
```
pip install rlm
```
To install the latest from `main`:
```
pip install git+https://github.com/alexzhang13/rlm.git
```

## Local Setup
```
curl -LsSf https://astral.sh/uv/install.sh | sh
uv init && uv venv --python 3.12  # change version as needed
uv pip install -e .
```

## Reading