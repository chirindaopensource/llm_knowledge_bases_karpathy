# **`README.md`**

# LLM-Operated Knowledge Base Architectures: A Formal Specification

<!-- PROJECT SHIELDS -->
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Source](https://img.shields.io/badge/Source-X%20(Twitter)%20Post-1DA1F2.svg)](https://x.com/karpathy/status/2039805659525644595)
[![Journal](https://img.shields.io/badge/Journal-Social%20Media%20Note-003366)](https://x.com/karpathy/status/2039805659525644595)
[![Year](https://img.shields.io/badge/Year-2026-purple)](https://github.com/chirindaopensource/llm_knowledge_bases_karpathy)
[![Discipline](https://img.shields.io/badge/Discipline-Systems%20Engineering%20%7C%20Knowledge%20Management-00529B)](https://github.com/chirindaopensource/llm_knowledge_bases_karpathy)
[![Data Sources](https://img.shields.io/badge/Data-Heterogeneous%20Local%20Corpora-lightgrey)](https://github.com/chirindaopensource/llm_knowledge_bases_karpathy)
[![Core Method](https://img.shields.io/badge/Method-Incremental%20Compilation%20%7C%20Graph%20Accretion-orange)](https://github.com/chirindaopensource/llm_knowledge_bases_karpathy)
[![Analysis](https://img.shields.io/badge/Analysis-LLM%20Orchestration%20%7C%20BM25%20%7C%20LoRA%20SFT-red)](https://github.com/chirindaopensource/llm_knowledge_bases_karpathy)
[![Validation](https://img.shields.io/badge/Validation-Citation%20Correctness%20%7C%20Grounding%20Invariants-green)](https://github.com/chirindaopensource/llm_knowledge_bases_karpathy)
[![Robustness](https://img.shields.io/badge/Robustness-AST%20Parsing%20%7C%20Docker%20Sandboxing-yellow)](https://github.com/chirindaopensource/llm_knowledge_bases_karpathy)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Type Checking: mypy](https://img.shields.io/badge/type%20checking-mypy-blue)](http://mypy-lang.org/)
[![OpenAI](https://img.shields.io/badge/OpenAI-API-%23412991.svg?style=flat&logo=openai&logoColor=white)](https://openai.com/)
[![NetworkX](https://img.shields.io/badge/NetworkX-Graph%20Theory-blue.svg)](https://networkx.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-Hugging%20Face-yellow.svg)](https://huggingface.co/)
[![Docker](https://img.shields.io/badge/Docker-Sandboxing-%230db7ed.svg?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![YAML](https://img.shields.io/badge/YAML-%23CB171E.svg?style=flat&logo=yaml&logoColor=white)](https://yaml.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-%23F37626.svg?style=flat&logo=Jupyter&logoColor=white)](https://jupyter.org/)
[![Open Source](https://img.shields.io/badge/Open%20Source-%E2%9D%A4-brightgreen)](https://github.com/chirindaopensource/llm_knowledge_bases_karpathy)

**Repository:** `https://github.com/chirindaopensource/llm_knowledge_bases_karpathy`

**Owner:** 2026 Craig Chirinda (Open Source Projects)

This repository contains an **independent**, professional-grade Python implementation of the systems engineering workflow described in the April 2026 note entitled **"LLM Knowledge Bases"** by:

*   **Andrej Karpathy**

The project provides a complete, end-to-end computational framework for operationalizing the concept of a continuously compounding, locally grounded "research and decision memory" system. It delivers a modular, highly optimized pipeline that executes the entire knowledge management workflow: from the rigorous ingestion and normalization of heterogeneous raw artifacts, through the incremental LLM-driven compilation of a directed Markdown graph, to grounded question-answering, artifact rendering, and optional parameter-efficient fine-tuning (LoRA).

## Table of Contents

- [Introduction](#introduction)
- [Theoretical Background](#theoretical-background)
- [Features](#features)
- [Methodology Implemented](#methodology-implemented)
- [Core Components (Notebook Structure)](#core-components-notebook-structure)
- [Key Callable: `run_knowledge_base_system`](#key-callable-run_knowledge_base_system)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Input Data Structure](#input-data-structure)
- [Usage](#usage)
- [Output Structure](#output-structure)
- [Project Structure](#project-structure)
- [Customization](#customization)
- [Contributing](#contributing)
- [Recommended Extensions](#recommended-extensions)
- [License](#license)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## Introduction

This project provides a rigorous Python implementation of the architectural pattern outlined by Karpathy (2026). The core of this repository is the iPython Notebook `llm_knowledge_bases_karpathy_draft.ipynb`, which contains a comprehensive suite of 20 orchestrated tasks to replicate and formalize the workflow.

The pipeline addresses a critical vulnerability in modern LLM usage: the reliance on ephemeral chat interactions that fail to accumulate durable knowledge. By treating the LLM not as a stateless oracle, but as a **knowledge organization engine** operating over a version-controllable filesystem substrate, the system enables cumulative knowledge accretion.

The codebase operationalizes this paradigm by:
-   **Ingesting** raw documents (PDFs, web clippings, repositories) into an immutable, append-only `raw/` directory.
-   **Compiling** these sources incrementally into a structured Markdown wiki, extracting concepts, and maintaining bidirectional links.
-   **Retrieving** information deterministically using auto-maintained indices and a field-aware BM25 search engine, bypassing the need for complex vector RAG at the ~400K word scale.
-   **Synthesizing** grounded answers that strictly enforce citation invariants, rendering them into durable artifacts (reports, slides, figures) that are filed back into the wiki.
-   **Auditing** the knowledge base via LLM linting to detect contradictions and impute missing data with strict provenance.

## Theoretical Background

The implemented methods formalize the workflow into a rigorous systems architecture.

**1. System Architecture:**
The system $\mathcal{S}$ is formalized as an 8-tuple defining the state and the operators acting upon that state:
$$ \mathcal{S} = \langle \mathcal{D}_{\text{raw}}, \mathcal{K}, \Phi_{\text{compile}}, \Phi_{\text{qa}}, \Phi_{\text{render}}, \Phi_{\text{lint}}, \mathcal{T}, \mathcal{L} \rangle $$

**2. Knowledge Graph Representation:**
The compiled wiki $\mathcal{K}$ is not a flat directory, but a directed labeled multigraph embedded in the filesystem:
$$ \mathcal{K} = (V, E, \lambda_V, \lambda_E) $$
Where nodes $V$ are Markdown files, edges $E$ are wikilinks, and $\lambda_V$ maps nodes to content and metadata (e.g., provenance, timestamps).

**3. Deterministic Local Retrieval:**
To support tool-assisted Q&A without vector embeddings, the system implements a field-aware BM25 scoring function:
$$ \text{score}(v, q) = \sum_{i=1}^m \text{idf}(t_i) \cdot \frac{\text{tf}_w(t_i, v) \cdot (k_1 + 1)}{\text{tf}_w(t_i, v) + k_1 \cdot \left(1 - b + b \cdot \frac{|v|}{\text{avgdl}}\right)} $$
Where $\text{tf}_w$ applies configured boosts to titles, headings, body text, and tags.

**4. The Compounding Cycle:**
The system executes a recurring transformation cycle, ensuring that generated artifacts $\mathcal{A}^{(t)}$ are reintegrated into the knowledge base:
$$ \mathcal{D}_{\text{raw}}^{(t)} \xrightarrow{\Phi_{\text{compile}}} \mathcal{K}^{(t)} \xrightarrow{\Phi_{\text{qa}}} \mathcal{A}^{(t)} \xrightarrow{\text{file}} \mathcal{K}^{(t+1)} \xrightarrow{\Phi_{\text{lint}}} \mathcal{K}_{\text{corrected}}^{(t+1)} $$

Below is a diagram which summarizes the proposed approach:

<div align="center">
  <img src="https://github.com/chirindaopensource/llm_knowledge_bases_karpathy/blob/main/llm_knowledge_bases_karpathy_ipo_main_three.png" alt="LLM Knowledge Base Architecture" width="100%">
</div>

## Features

The provided iPython Notebook (`llm_knowledge_bases_karpathy_draft.ipynb`) implements the full research pipeline, including:

-   **Cryptographic Determinism:** Utilizes full-file chunked SHA-256 hashing and canonical JSON serialization to guarantee cross-platform byte-for-byte determinism for run manifests and delta detection.
-   **AST-Aware Markdown Parsing:** Integrates `markdown-it-py` to parse Abstract Syntax Trees, ensuring that wikilinks and section headers are extracted securely, ignoring false positives inside fenced code blocks.
-   **Secure Docker Sandboxing:** Neutralizes the severe security risks of executing LLM-generated Python code (for matplotlib figures) by isolating execution within an ephemeral, network-disabled Docker container.
-   **Computational Linguistics (NLP):** Replaces brittle regex heuristics with `spaCy` for Named Entity Recognition (NER) to detect unsupported factual claims, and `nltk` for formal Sentence Boundary Disambiguation (SBD) during summary extraction.
-   **$O(1)$ Incremental Graph Updates:** Updates the in-memory `networkx` multigraph incrementally during artifact filing, avoiding computationally naive $O(|V| + |E|)$ full filesystem rebuilds.
-   **Configuration-Driven Design:** All study parameters, prompt templates, decoding settings, and directory schemas are managed in an external `config.yaml` file, ensuring strict methodological reproducibility.

## Methodology Implemented

The core analytical steps directly implement the workflow from the source note:

1.  **Initialization & Governance (Tasks 1-3, 7-9):** Validates the `config.yaml` via deep recursive schema traversal. Materializes the filesystem substrate (`raw/`, `wiki/`, `logs/`) and enforces strict symlink-aware write policies. Registers hashed prompt templates.
2.  **Ingestion & Normalization (Tasks 4-6):** Acquires heterogeneous sources, validates them via binary magic-byte inspection, and normalizes them to UTF-8 derivatives. Computes cryptographic hashes to detect incremental deltas ($\Delta \mathcal{D}$).
3.  **Incremental Compilation (Task 10):** Assembles semantic chunks and invokes the LLM to compile changed raw sources into structured Markdown source summaries and concept pages, updating the graph topology.
4.  **Indexing & Retrieval (Tasks 11-12):** Generates deterministic Maps of Content (MOCs) and builds the field-aware BM25 inverted index to support routing.
5.  **Grounded Q&A & Rendering (Tasks 13-14):** Routes queries using the search tool, synthesizes answers with strict citation enforcement, and renders durable artifacts (Markdown reports, Marp slides, matplotlib figures).
6.  **Filing & Linting (Tasks 15-16):** Files generated artifacts back into the wiki graph. Invokes the LLM to audit the wiki for contradictions and missing data, extracting minimal JSON-formatted patches.
7.  **Optional Fine-Tuning & Evaluation (Tasks 17-20):** Generates synthetic supervision data, splits it using exact integer arithmetic to prevent cluster leakage, executes LoRA fine-tuning with CPU-offloaded merging, and evaluates the system's Citation Correctness Rate (CCR).

## Core Components (Notebook Structure)

The notebook is structured as a logical pipeline with modular orchestrator functions for each of the 20 major tasks. All functions are self-contained, fully documented with strict type hints and comprehensive docstrings, and designed for professional-grade execution.

## Key Callable: `run_knowledge_base_system`

The project is designed around a single, top-level user-facing interface function:

-   **`run_knowledge_base_system`:** This apex orchestrator function runs the entire automated research pipeline from end-to-end. A single call to this function reproduces the entire workflow, managing static initialization, the dynamic operational cycle (ingest $\rightarrow$ compile $\rightarrow$ Q&A $\rightarrow$ file $\rightarrow$ lint), optional ML branches, and the final cryptographic reproducibility audit.

## Prerequisites

-   Python 3.10+
-   Docker (Required for secure sandboxed execution of matplotlib scripts)
-   Core Python dependencies: `openai`, `networkx`, `pyyaml`, `markdown-it-py`, `nltk`, `spacy`, `docker`, `python-frontmatter`.
-   Optional ML dependencies (for Task 18): `torch`, `transformers`, `peft`, `trl`, `datasets`.

## Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/chirindaopensource/llm_knowledge_bases_karpathy.git
    cd llm_knowledge_bases_karpathy
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Python dependencies:**
    ```sh
    pip install openai networkx pyyaml markdown-it-py nltk spacy docker python-frontmatter faker
    ```

4.  **Download NLP Models:**
    ```sh
    python -m spacy download en_core_web_sm
    ```

5.  **Install Optional ML Dependencies (if running fine-tuning):**
    ```sh
    pip install torch transformers peft trl datasets accelerate
    ```

## Input Data Structure

The pipeline requires a configuration dictionary and synthetic (or real) acquisition requests:

1.  **`config.yaml`:** The master configuration file defining directory schemas, LLM decoding settings, prompt templates, and evaluation metrics.
2.  **`RawAcquisitionRequest` (Dataclass):** Defines the intent to ingest a source.
    *   Fields: `source_type` (Enum), `title_or_name`, `source_url`, `local_source_path`, `revision`, `metadata_overrides`.
3.  **`EvalQuestion` (Dataclass):** Defines a benchmark question for Task 20 evaluation.
    *   Fields: `question_id`, `question`, `reference_answer`, `required_citations`, `is_multi_doc`.

## Usage

The notebook provides a complete, step-by-step guide. The primary workflow is to execute the final cell, which demonstrates how to load the configuration, generate synthetic acquisition requests, and use the top-level orchestrator to execute the pipeline:

```python
import os
import yaml
import pathlib
from typing import Dict, Any

# 1. Load the master configuration from the YAML file.
# (Assumes config.yaml is in the working directory)
def load_study_configuration(filepath: str = "config.yaml") -> Dict[str, Any]:
    if not isinstance(filepath, str):
        raise TypeError(f"filepath must be a string, got {type(filepath)}.")
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
        print(f"Successfully loaded configuration from {filepath}")
        return config
    except FileNotFoundError as e:
        print(f"CRITICAL ERROR: {filepath} not found in the working directory.")
        raise e

config = load_study_configuration("config.yaml")

# 2. Generate synthetic inputs (Example using generator provided in the notebook)
# In production, these would be driven by a UI or external event queue.
acq_reqs, user_queries, formats, eval_qs = generate_synthetic_pipeline_inputs()

# 3. Execute the entire knowledge base system.
if __name__ == "__main__":
    repo_root = pathlib.Path("./kb_workspace").resolve()
    
    if config:
        print("\nInitiating LLM Knowledge Base System...")
        
        system_artifacts = run_knowledge_base_system(
            raw_config=config,
            repo_root=repo_root,
            acquisition_requests=acq_reqs,
            queries=user_queries,
            requested_formats=formats,
            eval_questions=eval_qs
        )
        
        # 4. Access results
        print("\n" + "="*80)
        print("SYSTEM EXECUTION COMPLETE")
        print("="*80)
        
        pipeline_manifest = system_artifacts.get("pipeline_manifest")
        if pipeline_manifest:
            run_summary = pipeline_manifest.run_summary
            print(f"\n[Run Summary] ID: {run_summary['run_id']}")
            print(f"  - Wiki Files Written: {run_summary['wiki_files_written']}")
            print(f"  - Lint Issues Found: {run_summary['lint_issues_found_count']}")
            
        eval_bundle = system_artifacts.get("evaluation_bundle")
        if eval_bundle:
            baseline_report = eval_bundle.baseline_report
            print("\n[Baseline Evaluation KPIs]")
            print(f"  - Citation Coverage Rate (CCR): {baseline_report.ccr_coverage:.2%}")
            print(f"  - Citation Correctness Rate: {baseline_report.ccr_correct:.2%}")
            print(f"  - Total Hallucinations Detected: {baseline_report.total_hallucinations}")
```

## Output Structure

The pipeline returns a master dictionary containing two primary artifacts, serialized to disk under `logs/`:
-   **`EndToEndRunManifest`**: Contains the `run_summary` (counts of files written, lint issues, warnings), the `cycle_state` (cryptographic hashes of the graph before and after filing), and the optional `fine_tuning_manifest`.
-   **`FinalEvaluationBundle`**: Contains the `reproducibility_bundle` (deep order-invariant comparisons of repeated runs), the `baseline_report` (KPIs like Citation Correctness Rate), and the `acceptance_decision` for the fine-tuned model.

The filesystem substrate (`kb_workspace/`) will contain:
-   **`raw/`**: The immutable ingested sources and `manifest.yaml`.
-   **`wiki/`**: The compiled Markdown graph, including `sources/`, `concepts/`, `index/`, and `derived/` (reports, slides, figures).

## Project Structure

```
llm_knowledge_bases_karpathy/
│
├── llm_knowledge_bases_karpathy_draft.ipynb    # Main implementation notebook
├── config.yaml                                 # Master configuration file
├── requirements.txt                            # Python package dependencies
│
├── LICENSE                                     # MIT Project License File
└── README.md                                   # This file
```

## Customization

The pipeline is highly customizable via the `config.yaml` file. Users can modify study parameters such as:
-   **LLM Decoding Settings:** Adjust `temperature`, `top_p`, and `max_output_tokens` independently for compilation, Q&A, and linting tasks.
-   **Retrieval Parameters:** Modify the BM25 `field_boosts` (e.g., weighting tags higher than body text) or adjust the `snippet_chars` length.
-   **Linting Policies:** Define custom `what_counts_as_contradiction` rules or alter the `web_imputation_policy` to restrict authoritative domains.
-   **Fine-Tuning Hyperparameters:** Alter the LoRA `rank_r`, `learning_rate`, or `train_val_test_split_policy` fractions.

## Contributing

Contributions are welcome. Please fork the repository, create a feature branch, and submit a pull request with a clear description of your changes. Adherence to PEP 8, strict type hinting, and the 1:1 inline comment-to-code-line ratio is required.

## Recommended Extensions

Future extensions, as suggested by the architectural constraints, could include:
-   **Vector Database Integration:** Transitioning from the deterministic BM25 local search to a hybrid dense/sparse retrieval system (e.g., FAISS, Milvus) to support scaling beyond the ~400K word regime.
-   **Multi-Agent Orchestration:** Replacing the scripted-calls framework with a formal multi-agent framework (e.g., AutoGen, LangGraph) to allow autonomous, multi-step research planning prior to synthesis.
-   **Continuous Pre-Training (CPT):** Exploring CPT instead of SFT/LoRA to more deeply embed the wiki's conceptual lattice into the model's parametric memory.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Citation

If you use this code or the methodology in your research, please cite the original source note:

```bibtex
@misc{karpathy2026llmkb,
  author = {Karpathy, Andrej},
  title = {LLM Knowledge Bases},
  year = {2026},
  howpublished = {X (formerly Twitter) Post},
  url = {https://x.com/karpathy/status/2039805659525644595}
}
```

For the implementation itself, you may cite this repository:
```
Chirinda, C. (2026). LLM-Operated Knowledge Base Architectures: A Formal Specification.
GitHub repository: https://github.com/chirindaopensource/llm_knowledge_bases_karpathy
```

## Acknowledgments

-   Credit to **Andrej Karpathy** for the foundational workflow description that forms the entire basis for this computational formalization.
-   This project is built upon the exceptional tools provided by the open-source community. Sincere thanks to the developers of the scientific Python ecosystem, particularly the **OpenAI**, **NetworkX**, **PyTorch**, **Transformers**, and **PEFT** contributors.

--

*This README was generated based on the structure and content of the `llm_knowledge_bases_karpathy_draft.ipynb` notebook and follows best practices for research software documentation.*
