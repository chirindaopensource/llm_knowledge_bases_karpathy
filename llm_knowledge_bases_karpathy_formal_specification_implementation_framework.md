# LLM-Operated Knowledge Base Architectures: A Formal Specification and Implementation Framework

**Abstract**

This paper presents a formal specification and reference implementation framework for constructing large-language-model-operated personal and organizational knowledge bases from heterogeneous research corpora. The architecture is derived from a workflow description by Karpathy (2026) and extended into a rigorous, reproducible specification suitable for deployment in research and production environments. The system operates on a file-centric substrate wherein raw source materials—scholarly articles, technical documentation, software repositories, structured datasets, and multimedia artifacts—are systematically indexed into a local filesystem hierarchy. A large language model then performs incremental compilation of these materials into a structured knowledge graph encoded as interlinked Markdown documents, augmented with provenance metadata, cross-reference indices, and conceptual categorization. This compiled knowledge base subsequently serves as a queryable substrate for complex information synthesis tasks, artifact generation (technical reports, presentation materials, data visualizations), and automated consistency verification. We formalize the data structures, operational semantics, and algorithmic procedures required for robust implementation, and provide empirical design recommendations derived from first-principles analysis of the workflow's operational constraints.

**Idea Originator**: Andrej Karpathy

**Contributors**: CS Chirinda, GPT 5.4, Claude Sonnet 4.5, Claude Sonnet 4.6

**Date**: 3 April 2026

**Keywords:** knowledge management systems, large language models, file-based knowledge graphs, incremental compilation, provenance tracking, research infrastructure

---

## 1. Introduction

### 1.1 Research Context and Motivation

Contemporary applications of large language models have predominantly focused on code synthesis (Chen et al., 2021; Li et al., 2022) and conversational assistance (OpenAI, 2023). However, an emergent and arguably more consequential paradigm treats these models as **knowledge organization engines** that operate continuously over evolving domain-specific corpora. Rather than processing queries in isolation against a static training distribution, such systems maintain a persistent, locally grounded knowledge substrate and perform targeted information synthesis by consulting that substrate under explicit provenance constraints.

This architectural pattern exhibits several theoretically and practically significant properties. First, it replaces ephemeral interaction with **durable structured memory**, thereby supporting cumulative knowledge accretion rather than stateless question-answering. Second, it grounds synthesis operations in an auditable local corpus, mitigating certain classes of hallucination risk inherent to unconstrained generation. Third, it exposes the knowledge base as a file-based artifact amenable to version control, differential audit, and human supervision. Fourth, it supports a closed-loop workflow wherein model-generated outputs are reintegrated into the knowledge base, thereby increasing the effective information density of the substrate over time.

The present work formalizes this pattern into an implementable specification.

### 1.2 Source Material and Epistemic Framework

This specification is derived from a brief workflow description published by Karpathy (2026) on a public microblogging platform. The source material is not a peer-reviewed research article; rather, it is a concise operational note describing a practitioner workflow. Consequently, the source provides:

- a high-level system architecture,
- a set of operational stages and data modalities,
- qualitative performance claims (e.g., feasibility without sophisticated retrieval augmentation at moderate corpus scale),
- but no formal algorithms, evaluation protocols, hyperparameter specifications, or quantitative benchmarks.

This paper therefore operates under a **dual epistemic framework**:

1. **Source-faithful reconstruction:** we preserve all operationally significant constraints and design principles explicitly stated or strongly implied by the source.
2. **Reference implementation layer:** we introduce formal specifications, data schemas, algorithmic procedures, and default parameter settings necessary to enable disciplined reproduction.

Throughout the document, we distinguish these layers explicitly to maintain scholarly rigor and intellectual honesty.

### 1.3 Problem Formalization

Let $\mathcal{D}_{\text{raw}}$ denote a heterogeneous corpus of raw research artifacts. We seek to construct a system $\mathcal{S}$ that implements the following functional mapping:

$$
\mathcal{S}: \mathcal{D}_{\text{raw}} \times \mathcal{Q} \rightarrow \mathcal{K} \times \mathcal{A}
$$

where:

- $\mathcal{Q}$ is the space of user information needs (queries),
- $\mathcal{K}$ is a compiled, structured knowledge base represented as a directed labeled graph embedded in a filesystem,
- $\mathcal{A}$ is the space of durable output artifacts (reports, presentations, visualizations).

The system must satisfy the following constraints:

1. **Incrementality:** updates to $\mathcal{D}_{\text{raw}}$ propagate to $\mathcal{K}$ without requiring global recompilation.
2. **Traceability:** every assertion in $\mathcal{K}$ or $\mathcal{A}$ is linked to one or more elements of $\mathcal{D}_{\text{raw}}$ via explicit provenance edges.
3. **Cumulativeness:** artifacts $a \in \mathcal{A}$ generated at time $t$ are reintegrated into $\mathcal{K}_{t+1}$.
4. **Consistency:** the system supports automated detection and remediation of contradictions, schema violations, and missing references within $\mathcal{K}$.

### 1.4 Contributions

This paper makes the following contributions:

1. A **formal system architecture** for LLM-operated knowledge bases, distinguishing filesystem substrates, transformation operators, and control flows.
2. A **complete data model** including schemas for raw corpora, compiled knowledge graphs, metadata, and logging artifacts.
3. A **detailed methodology** specifying each operational stage with algorithmic precision, including ingest, compilation, indexing, query synthesis, artifact rendering, integrity verification, and optional model fine-tuning.
4. **Reference prompt templates** and runtime configurations suitable for immediate deployment.
5. A **failure mode analysis** and a set of operational guardrails.
6. An **evaluation framework** enabling reproducible benchmarking of knowledge base quality and query performance.

---

## 2. System Architecture

### 2.1 High-Level Overview

The system architecture can be formalized as a tuple:

$$
\mathcal{S} = \langle \mathcal{D}_{\text{raw}}, \mathcal{K}, \Phi_{\text{compile}}, \Phi_{\text{qa}}, \Phi_{\text{render}}, \Phi_{\text{lint}}, \mathcal{T}, \mathcal{L} \rangle
$$

where:

- $\mathcal{D}_{\text{raw}}$ is the raw corpus store,
- $\mathcal{K}$ is the compiled knowledge base,
- $\Phi_{\text{compile}}: \mathcal{D}_{\text{raw}} \times \mathcal{K}_{t-1} \rightarrow \mathcal{K}_t$ is the incremental compilation operator,
- $\Phi_{\text{qa}}: \mathcal{Q} \times \mathcal{K} \rightarrow \mathcal{A}$ is the query synthesis operator,
- $\Phi_{\text{render}}: \mathcal{A}_{\text{intermediate}} \rightarrow \mathcal{A}_{\text{files}}$ is the artifact rendering operator,
- $\Phi_{\text{lint}}: \mathcal{K} \rightarrow \mathcal{L}_{\text{issues}} \times \mathcal{K}_{\text{patches}}$ is the integrity verification operator,
- $\mathcal{T}$ is a set of external tools (search engines, web retrievers),
- $\mathcal{L}$ is a structured log repository.

### 2.2 Filesystem Substrate

The system employs a **file-centric architecture** wherein all persistent state resides in a version-controllable directory hierarchy. Let $\mathcal{F}$ denote the root filesystem namespace. The canonical directory schema is:

$$
\mathcal{F} = \{ \text{raw/}, \text{wiki/}, \text{assets/}, \text{logs/}, \text{eval/} \}
$$

with the following semantics:

- **raw/**: immutable (append-only) storage for ingested source artifacts,
- **wiki/**: LLM-managed Markdown files representing the compiled knowledge base,
- **assets/**: shared multimedia resources,
- **logs/**: structured execution logs for reproducibility and audit,
- **eval/**: benchmark question sets and evaluation artifacts.

The **wiki/** subdirectory is further structured as:

$$
\text{wiki/} = \{ \text{index/}, \text{concepts/}, \text{sources/}, \text{derived/} \}
$$

This stratification supports:

- **index/**: navigational scaffolding (maps of content, global indices),
- **concepts/**: cross-source thematic synthesis,
- **sources/**: per-artifact summaries maintaining direct provenance,
- **derived/**: LLM-generated reports, slides, and figures filed back into the knowledge base.

### 2.3 Knowledge Graph Representation

The compiled knowledge base $\mathcal{K}$ can be formalized as a directed labeled multigraph:

$$
\mathcal{K} = (V, E, \lambda_V, \lambda_E)
$$

where:

- $V$ is the set of nodes (wiki pages),
- $E \subseteq V \times V$ is the set of directed edges (hyperlinks and backlinks),
- $\lambda_V: V \rightarrow \Sigma^*$ assigns content (Markdown text) to each node,
- $\lambda_E: E \rightarrow \mathcal{L}_{\text{edge}}$ assigns labels to edges (e.g., "supports," "contradicts," "extends").

Each node $v \in V$ is associated with metadata:

$$
\text{meta}(v) = \langle \text{path}(v), \text{type}(v), \text{provenance}(v), \text{timestamp}(v), \text{tags}(v) \rangle
$$

where:

- $\text{type}(v) \in \{ \mathit{source\_summary}, \mathit{concept}, \mathit{index}, \mathit{derived\_artifact} \}$,
- $\text{provenance}(v) \subseteq \mathcal{D}_{\text{raw}}$ is the set of raw sources supporting $v$,
- $\text{timestamp}(v)$ records creation and modification times.

### 2.4 Operational Cycle

The system executes a recurring transformation cycle:

$$
\mathcal{D}_{\text{raw}}^{(t)} \xrightarrow{\Phi_{\text{compile}}} \mathcal{K}^{(t)} \xrightarrow{\Phi_{\text{qa}}} \mathcal{A}^{(t)} \xrightarrow{\text{file}} \mathcal{K}^{(t+1)} \xrightarrow{\Phi_{\text{lint}}} \mathcal{K}_{\text{corrected}}^{(t+1)}
$$

This cycle exhibits **cumulative dynamics**: the knowledge base $\mathcal{K}^{(t+1)}$ is strictly more informative than $\mathcal{K}^{(t)}$ provided that:

1. integrity constraints are maintained,
2. semantic drift is controlled,
3. provenance links remain valid.

---

## 3. Data Model and Formal Specifications

### 3.1 Raw Corpus Schema

The raw corpus $\mathcal{D}_{\text{raw}}$ is a finite set of typed artifacts. Each artifact $d \in \mathcal{D}_{\text{raw}}$ is represented as:

$$
d = \langle \text{path}(d), \text{type}(d), \text{content}(d), \text{metadata}(d) \rangle
$$

where:

- $\text{path}(d)$ is the relative filesystem path from the raw/ directory,
- $\text{type}(d) \in \{ \text{pdf}, \text{html}, \text{code}, \text{dataset}, \text{video}, \text{audio} \}$,
- $\text{content}(d)$ is the raw byte sequence or extracted text representation,
- $\text{metadata}(d)$ includes source URL, timestamp, file hash, and optional domain-specific annotations.

**Ingestion Protocol:**

```
procedure INGEST(source_url, target_path):
    content ← FETCH(source_url)
    metadata ← EXTRACT_METADATA(content, source_url)
    WRITE(target_path, content)
    LOG(target_path, metadata, timestamp)
    return target_path
```

### 3.2 Compiled Knowledge Base Schema

The knowledge base $\mathcal{K}$ consists of structured Markdown files with enforced conventions.

**Page Schema:**

A wiki page $p \in \mathcal{K}$ must conform to:

```markdown
---
title: {page_title}
type: {source_summary | concept | index | derived_artifact}
created: {ISO8601_timestamp}
modified: {ISO8601_timestamp}
tags: [{tag1}, {tag2}, ...]
provenance: [{raw/path1}, {raw/path2}, ...]
---

# {Page Title}

## Summary
{One-paragraph executive summary}

## Content
{Structured content with wikilinks [[target_page]]}

## Citations
{Explicit references to raw sources or wiki pages}

## Related
- [[Related Page 1]]
- [[Related Page 2]]

## Open Questions
- {Question 1}
- {Question 2}
```

**Wikilink Syntax:**

- Internal links: `[[wiki/concepts/neural_architectures]]`
- Backlinks are computed dynamically via static analysis.

**Provenance Links:**

Every substantive assertion must include:

```markdown
According to [source](../raw/papers/karpathy2026.pdf#page=3), ...
```

### 3.3 Metadata and Logging

**Structured Logs:**

Each system operation generates a structured log entry:

```json
{
  "timestamp": "2026-04-03T14:32:01Z",
  "operation": "compile",
  "inputs": ["raw/papers/attention_paper.pdf"],
  "outputs": ["wiki/sources/attention_paper.md", "wiki/concepts/transformers.md"],
  "model": "gpt-5.2-high",
  "tokens": {"prompt": 8432, "completion": 2156},
  "latency_ms": 3421,
  "status": "success",
  "warnings": []
}
```

**Provenance Graph:**

The system maintains a directed acyclic graph:

```
raw/paper.pdf → wiki/sources/paper.md → wiki/concepts/topic.md → wiki/derived/report.md
```

---

## 4. Operational Stages

### 4.1 Stage 1: Ingestion and Local Indexing

**Objective:** Systematically acquire and catalog raw research materials.

**Procedure:**

```
procedure INGESTION_PHASE():
    sources ← LOAD_SOURCE_LIST()
    for each source in sources:
        if source.type == "url":
            path ← DOWNLOAD(source.url, raw/)
        else if source.type == "file":
            path ← COPY(source.path, raw/)
        INDEX_ENTRY(path, source.metadata)
    BUILD_BM25_INDEX(raw/)
```

**Index Structure:**

The system builds an inverted index $\mathcal{I}$ mapping:

$$
\mathcal{I}: \text{token} \rightarrow \{ (d, \text{tf}(d, t), \text{pos}(d, t)) \mid d \in \mathcal{D}_{\text{raw}} \}
$$

### 4.2 Stage 2: Incremental Compilation

**Objective:** Transform raw sources into structured wiki pages.

**Algorithmic Specification:**

```
procedure INCREMENTAL_COMPILE(changed_files):
    context ← LOAD_WIKI_STATE()
    for each file in changed_files:
        raw_content ← READ(file)
        source_page ← GENERATE_SOURCE_SUMMARY(raw_content, context)
        concepts ← EXTRACT_CONCEPTS(raw_content, context)
        
        WRITE(wiki/sources/{file_id}.md, source_page)
        
        for each concept in concepts:
            if concept NOT IN context.concepts:
                concept_page ← CREATE_CONCEPT_PAGE(concept, [file])
                WRITE(wiki/concepts/{concept}.md, concept_page)
            else:
                UPDATE_CONCEPT_PAGE(concept, file)
        
        UPDATE_INDEX_PAGES(context, file, concepts)
    
    VERIFY_LINKS()
    COMMIT_CHANGES()
```

**LLM Invocation:**

```python
def generate_source_summary(raw_content, context):
    prompt = f"""
    Analyze this source and create a structured summary:
    
    SOURCE: {raw_content[:8000]}
    
    EXISTING WIKI CONTEXT: {context.relevant_pages}
    
    OUTPUT: Markdown page following schema with:
    - Executive summary
    - Key concepts (link to existing wiki pages where applicable)
    - Novel contributions
    - Citations
    """
    response = llm.complete(prompt, temperature=0.2, max_tokens=8000)
    return parse_markdown(response)
```

### 4.3 Stage 3: Global Index Construction

**Objective:** Build navigational scaffolding.

**Index Types:**

1. **Alphabetical Index:** `wiki/index/alphabetical.md`
2. **Concept Map:** `wiki/index/concept_map.md` (graph visualization)
3. **Source Timeline:** `wiki/index/timeline.md` (chronological)
4. **Topic Hierarchies:** `wiki/index/topics/{domain}.md`

**Construction Algorithm:**

```
procedure BUILD_GLOBAL_INDEX():
    pages ← ENUMERATE(wiki/)
    
    # Alphabetical
    sorted_pages ← SORT(pages, key=title)
    RENDER(wiki/index/alphabetical.md, sorted_pages)
    
    # Concept graph
    graph ← BUILD_LINK_GRAPH(pages)
    clusters ← DETECT_COMMUNITIES(graph)
    RENDER(wiki/index/concept_map.md, clusters)
    
    # Timeline
    timeline ← SORT(pages, key=timestamp)
    RENDER(wiki/index/timeline.md, timeline)
```

### 4.4 Stage 4: Grounded Query-Answer Synthesis

**Objective:** Respond to user information needs by synthesizing wiki content.

**Protocol:**

```
procedure GROUNDED_QA(query):
    # Step 1: Retrieve relevant pages
    candidates ← BM25_SEARCH(query, wiki/, top_k=20)
    shortlist ← LLM_RERANK(query, candidates, top_k=10)
    
    # Step 2: Load full content
    pages ← [READ(p) for p in shortlist]
    
    # Step 3: Synthesize answer
    prompt ← CONSTRUCT_QA_PROMPT(query, pages)
    answer ← LLM_COMPLETE(prompt)
    
    # Step 4: Verify citations
    citations ← EXTRACT_CITATIONS(answer)
    if not VERIFY_CITATIONS(citations, pages):
        answer ← REQUERY_WITH_CITATION_REQUIREMENT()
    
    return answer, citations, shortlist
```

**Prompt Template:**

```
QUERY: {user_query}

RELEVANT WIKI PAGES:
{page_1_content}
{page_2_content}
...

TASK: Answer the query using ONLY the information in the provided pages.
Every claim must cite a specific page path.

FORMAT:
## Answer
{grounded synthesis}

## Evidence
- Claim 1: [wiki/path/to/page.md]
- Claim 2: [wiki/path/to/other.md]

## Caveats
{limitations, unknowns, or contradictions}
```

### 4.5 Stage 5: Artifact Rendering

**Objective:** Generate durable outputs (reports, slides, figures).

**Supported Formats:**

1. **Markdown Reports:** `.md` files with academic structure
2. **Presentation Decks:** `.pptx` or reveal.js HTML
3. **Data Visualizations:** Python/R scripts with reproducible plots

**Rendering Procedure:**

```
procedure RENDER_ARTIFACT(spec):
    if spec.format == "report":
        outline ← LLM_GENERATE_OUTLINE(spec.topic, wiki/)
        sections ← [LLM_WRITE_SECTION(s, wiki/) for s in outline]
        report ← ASSEMBLE_DOCUMENT(sections)
        WRITE(wiki/derived/{spec.title}.md, report)
    
    else if spec.format == "slides":
        deck ← LLM_GENERATE_SLIDES(spec.topic, wiki/, num_slides=spec.count)
        WRITE(wiki/derived/{spec.title}.pptx, deck)
    
    # Reintegrate into knowledge base
    ADD_TO_INDEX(spec.output_path)
```

**Cumulative Integration:**

Generated artifacts are filed in `wiki/derived/` and become queryable for future synthesis tasks. This creates a **positive feedback loop** wherein the knowledge base becomes progressively richer.

### 4.6 Stage 6: Integrity Verification (Linting)

**Objective:** Detect and remediate inconsistencies.

**Verification Targets:**

1. **Structural Issues:**
   - Broken wikilinks
   - Orphan pages (unreachable from index)
   - Missing required sections
   - Malformed metadata

2. **Semantic Issues:**
   - Contradictory assertions
   - Undefined technical terms
   - Unsupported claims (missing citations)
   - Duplicated content across pages

**Linting Procedure:**

```
procedure LINT_KNOWLEDGE_BASE():
    issues ← []
    
    # Structural checks
    all_pages ← ENUMERATE(wiki/)
    for page in all_pages:
        links ← EXTRACT_LINKS(page)
        for link in links:
            if not EXISTS(link):
                issues.append({"type": "broken_link", "page": page, "target": link})
    
    # Semantic checks
    contradictions ← LLM_DETECT_CONTRADICTIONS(all_pages)
    issues.extend(contradictions)
    
    # Generate patches
    patches ← LLM_PROPOSE_FIXES(issues)
    
    return issues, patches
```

**LLM-Assisted Linting:**

```python
def detect_contradictions(pages):
    prompt = f"""
    Analyze these wiki pages for contradictions:
    
    {pages}
    
    OUTPUT JSON:
    [
      {{
        "claim_1": "...",
        "page_1": "wiki/...",
        "claim_2": "...",
        "page_2": "wiki/...",
        "conflict": "explanation"
      }}
    ]
    """
    return llm.complete(prompt, temperature=0.1)
```

### 4.7 Stage 7: Optional Model Fine-Tuning

**Objective:** Train a domain-specialized model on the knowledge base.

**Synthetic Data Generation:**

```
procedure GENERATE_TRAINING_DATA():
    pages ← SAMPLE(wiki/, n=500)
    examples ← []
    
    for page in pages:
        # Question generation
        questions ← LLM_GENERATE_QUESTIONS(page, num=5)
        
        for q in questions:
            answer ← LLM_ANSWER(q, [page])
            examples.append({
                "instruction": q,
                "input": "",
                "output": answer,
                "citations": [page.path]
            })
    
    WRITE(eval/synthetic_dataset.jsonl, examples)
```

**Fine-Tuning Configuration:**

```python
FINE_TUNE_CONFIG = {
    "method": "SFT_LoRA",
    "base_model": "gpt-oss-20b",
    "data": "eval/synthetic_dataset.jsonl",
    "hyperparameters": {
        "learning_rate": 1e-4,
        "batch_size": 128,
        "epochs": 2,
        "max_seq_len": 8192
    }
}
```

---

## 5. Empirical Design Recommendations

### 5.1 Corpus Scale and Retrieval Strategy

**Karpathy's Observation:**

> "For moderate-scale personal knowledge bases (hundreds of papers, documents), sophisticated vector retrieval or embedding schemes are not strictly necessary. A well-structured file hierarchy with keyword-based search suffices."

**Formalization:**

Define corpus scale regimes:

- **Small:** $|\mathcal{D}_{\text{raw}}| < 500$ documents → BM25 sufficient
- **Medium:** $500 \leq |\mathcal{D}_{\text{raw}}| < 10{,}000$ → BM25 + LLM reranking
- **Large:** $|\mathcal{D}_{\text{raw}}| \geq 10{,}000$ → Dense retrieval (e.g., FAISS) required

**Recommendation:** Start with BM25; measure retrieval recall at $k=20$. If recall drops below 0.85, upgrade to hybrid retrieval.

### 5.2 Model Selection

**Compilation and Indexing:** Use reasoning-optimized models with high context windows (e.g., GPT-5.2, Claude Opus 4.5). Temperature: 0.1–0.2 for deterministic output.

**Q&A Synthesis:** Moderate temperature (0.3–0.5) for nuanced explanations. Require explicit citation syntax.

**Artifact Rendering:** Higher temperature (0.5–0.7) for creative outputs (slides, visualizations).

### 5.3 Incremental Compilation Frequency

**Trade-off:**

- **Frequent updates** (daily): Low-latency incorporation of new sources, but higher API costs.
- **Batch updates** (weekly): Cost-efficient, but delayed knowledge availability.

**Recommendation:** Adaptive scheduling:

```
if new_sources.importance == "urgent":
    compile_immediately()
else:
    schedule_batch_update()
```

### 5.4 Provenance Granularity

**Options:**

1. **Document-level:** Link to source file only
2. **Section-level:** Link to specific sections/pages
3. **Sentence-level:** Inline citations for each assertion

**Recommendation:** Section-level provenance balances precision and maintainability. Implement via anchor links:

```markdown
According to [Vaswani et al., 2017, §3.2](../raw/papers/attention.pdf#page=4), ...
```

### 5.5 Handling Multimodal Sources

**Strategy:**

- **PDFs:** Extract text with `pdfplumber`; render figures as separate image files in `assets/`
- **Videos:** Transcribe audio with Whisper; extract keyframes for visual indexing
- **Code Repositories:** Index at file granularity; link function definitions to concept pages

---

## 6. Failure Modes and Guardrails

### 6.1 Common Failure Modes

**1. Citation Drift:**

- **Symptom:** Wiki assertions lose connection to raw sources over multiple edits.
- **Mitigation:** Periodic provenance audits; reject edits without citations.

**2. Concept Fragmentation:**

- **Symptom:** Multiple wiki pages cover overlapping topics.
- **Mitigation:** Semantic deduplication during compilation; merge similar pages.

**3. Index Staleness:**

- **Symptom:** Global indices don't reflect recent wiki changes.
- **Mitigation:** Automatic reindexing on every compilation cycle.

**4. Hallucinated Cross-References:**

- **Symptom:** LLM generates plausible but non-existent wikilinks.
- **Mitigation:** Post-generation link validation; require LLM to verify links exist before inserting them.

### 6.2 Quality Assurance Protocols

```
procedure QUALITY_ASSURANCE():
    # Coverage check
    uncovered_sources ← FIND_SOURCES_WITHOUT_SUMMARIES()
    if len(uncovered_sources) > 0.05 * len(raw/):
        ALERT("High uncovered source ratio")
    
    # Freshness check
    stale_pages ← FIND_PAGES_NOT_UPDATED(days=90)
    RECOMMEND_REFRESH(stale_pages)
    
    # Citation density
    for page in wiki/:
        citation_ratio ← COUNT_CITATIONS(page) / WORD_COUNT(page)
        if citation_ratio < 0.01:
            FLAG(page, "Low citation density")
```

---

## 7. Evaluation Framework

### 7.1 Knowledge Base Quality Metrics

**Structural Metrics:**

- **Coverage:** $C = \frac{|\text{sources with summaries}|}{|\mathcal{D}_{\text{raw}}|}$
- **Connectivity:** Average in-degree and out-degree of wiki pages
- **Index Completeness:** Fraction of pages reachable from main index

**Content Metrics:**

- **Citation Density:** Citations per 1000 words
- **Concept Granularity:** Number of concept pages per 100 source documents

### 7.2 Query Performance Evaluation

**Protocol:**

1. Compile a gold-standard Q&A set: 100 questions with ground-truth answers
2. For each question:
   - Retrieve top-$k$ pages
   - Generate answer
   - Measure:
     - **Retrieval Recall@k:** Fraction of gold pages retrieved
     - **Answer Accuracy:** Human judgment (correct / partially correct / incorrect)
     - **Citation Precision:** Fraction of citations that actually support claims

**Benchmark Suite:**

```json
{
  "question_id": "Q042",
  "question": "What are the computational bottlenecks in transformer training?",
  "gold_pages": ["wiki/concepts/transformers.md", "wiki/sources/attention_paper.md"],
  "gold_answer_key_points": ["Attention O(n^2)", "Memory for activations"],
  "difficulty": "MEDIUM"
}
```

### 7.3 System Performance Metrics

- **Compilation Latency:** Time to process $n$ new sources
- **Query Latency:** Time from query to answer (including retrieval + synthesis)
- **API Cost per Query:** Token usage for end-to-end Q&A cycle

**Target Benchmarks:**

- Compilation: <5 minutes per source (for typical research papers)
- Query: <30 seconds (for moderate complexity questions)
- Cost: <$0.10 per query (at GPT-5 pricing)

---

## 8. Related Work and Theoretical Foundations

### 8.1 Personal Knowledge Management

The system draws on established paradigms in PKM:

- **Zettelkasten** (Luhmann, 1992): Atomic notes with dense cross-links
- **Obsidian/Roam Research**: Graph-based wiki tools
- **Memex** (Bush, 1945): Associative information retrieval

**Novelty:** LLM-driven automation of note creation, indexing, and synthesis.

### 8.2 Retrieval-Augmented Generation (RAG)

Canonical RAG (Lewis et al., 2020):

$$
P(y | x) = \sum_{d \in \text{retrieve}(x)} P(y | x, d) P(d | x)
$$

**Distinction:**

- RAG: Stateless retrieval over static corpus
- This system: Stateful knowledge base with cumulative compilation

### 8.3 Symbolic Knowledge Graphs

Contrasts with RDF/OWL-based KGs:

- **Representation:** Markdown files vs. triple stores
- **Schema:** Flexible vs. rigid ontologies
- **Reasoning:** LLM-driven synthesis vs. logical inference

**Advantage:** Lower implementation complexity; human-readable artifacts.

---

## 9. Limitations and Future Directions

### 9.1 Current Limitations

1. **Scalability:** File-based architecture may degrade at $>50{,}000$ pages
2. **Multimodal Integration:** Limited support for videos, interactive simulations
3. **Collaborative Editing:** No built-in conflict resolution for concurrent updates
4. **Evaluation:** Lacks standardized benchmarks for knowledge base quality

### 9.2 Future Research Directions

**1. Automated Knowledge Consolidation:**

- Periodic global reviews to merge redundant pages
- Concept drift detection and remediation

**2. Interactive Knowledge Graphs:**

- Visual graph browsers with semantic zoom
- Highlighted provenance trails

**3. Multi-Agent Workflows:**

- Specialist agents for different domains (code, math, biology)
- Peer review loops between agents

**4. Federated Knowledge Bases:**

- Protocols for sharing and merging knowledge bases across users
- Privacy-preserving provenance

---

## 10. Conclusion

This paper formalizes the architecture for LLM-operated knowledge bases as described by Karpathy (2026), extending a practitioner workflow into a rigorous, reproducible specification. The system combines file-centric storage, incremental compilation, provenance tracking, and cumulative artifact integration to create a self-improving knowledge substrate.

**Key Takeaways:**

1. **File-based knowledge graphs** offer a pragmatic middle ground between unstructured notes and formal semantic systems.
2. **Incremental compilation** with LLM-driven synthesis enables continuous knowledge accretion without manual curation overhead.
3. **Provenance enforcement** mitigates hallucination risks while preserving auditability.
4. **Cumulative dynamics** wherein generated artifacts become future source material create a positive feedback loop.

The specification supports immediate deployment in research and production environments, with empirical guidelines for model selection, retrieval strategies, and quality assurance.

**Broader Impact:**

This architecture demonstrates a consequential paradigm shift: from LLMs as **conversational assistants** to LLMs as **persistent knowledge organization engines**. By grounding synthesis in local, version-controlled corpora, such systems offer a path toward more reliable, auditable, and domain-specialized AI workflows.

---

# Appendix A. Reference Prompt Templates

## A.1 System Prompt

```
You are a rigorous research assistant maintaining a local Markdown knowledge base.

Operational constraints:
1. Do not fabricate assertions.
2. Every substantive claim must be grounded in provided local sources
   (wiki pages or raw documents) and accompanied by explicit citations.
3. When information is unavailable, respond with "Unknown" and propose
   specific sources to ingest.
4. Adhere strictly to requested output schemas.
```

## A.2 Incremental Compilation Prompt

```
TASK: Perform incremental compilation of changed raw sources into the wiki.

INPUTS:
- changed_raw_files: [{path, type, content}]
- existing_wiki_context: [{path, content_excerpt}]
- conventions: {directory_targets, link_syntax, required_sections}

REQUIREMENTS:
1. For each changed raw source, create or update a source summary page.
2. Extract domain concepts; create or update concept pages with cross-source synthesis.
3. Maintain bidirectional links (wikilinks and backlinks).
4. Update global index pages with brief one-sentence entries.
5. Do not modify pages unrelated to the changed sources.
6. Ground all assertions in raw excerpts or existing wiki pages.

OUTPUT SCHEMA (JSON):
{
  "files_to_write": [{"path": str, "content_markdown": str}],
  "files_to_edit": [{"path": str, "patch_instructions": str}],
  "new_concepts": [{"title": str, "rationale": str}],
  "open_questions": [str]
}
```

## A.3 Grounded Q&A Prompt

```
TASK: Answer the user query by consulting the local wiki.

PROTOCOL:
1. Use provided index excerpts to propose a shortlist of relevant pages.
2. After pages are provided, synthesize an answer grounded exclusively in those pages.
3. Cite wiki page paths for every substantive assertion.
4. If the wiki lacks necessary information, state this explicitly and recommend
   specific sources to ingest.

OUTPUT FORMAT (Markdown):
## Conclusion
## Evidence (with citations)
## Caveats / Unknowns
## Pages Consulted (paths)
## Recommended Wiki Improvements
```

## A.4 Markdown Report Rendering

```
TASK: Render a structured technical report as a Markdown file.

CONSTRAINTS:
- Output must be valid Markdown suitable for Obsidian.
- All citations must be wiki links or explicit repo-relative paths.
- Do not introduce assertions beyond grounded findings.

STRUCTURE:
# {Title}
## Executive Summary
## Main Analysis
## Evidence & Citations
## Limitations
## Next Questions
```

## A.5 Linting Prompt

```
TASK: Perform integrity verification over the wiki.

DETECTION TARGETS (with exact evidence and page paths):
1. Contradictions or semantic inconsistencies
2. Missing citations, undefined terms, or placeholder content
3. Structural issues: broken links, orphan pages, duplicate concepts
4. Opportunities: high-value cross-links, candidate concept pages

OUTPUT (Markdown):
## Summary Table (Issue Type, Severity, Page)
## Detailed Findings (with quoted excerpts)
## Proposed Minimal Patches
## Suggested New Pages / Follow-Up Questions
```

## A.6 Synthetic Data Generation Prompt

```
TASK: Generate synthetic supervised examples grounded in wiki pages.

RULES:
- Every example must be answerable using ONLY the provided pages.
- Include citations (repo-relative paths) for each example.
- Do not introduce external facts.

OUTPUT: JSONL with schema:
{
  "instruction": str,
  "input": str,
  "output": str,
  "citations": [str],
  "difficulty": "EASY" | "MEDIUM" | "HARD",
  "notes": str
}
```

---

# Appendix B. Reference Runtime Configuration

```python
CONFIG = {
    "provider": "openai",
    "api_base": "https://api.openai.com/v1",
    "api_key_env": "OPENAI_API_KEY",

    "models": {
        "reasoning": "gpt-5.2-high",
        "fine_tune_base": "gpt-oss-20b",
    },

    "decoding": {
        "compile": {"temperature": 0.2, "top_p": 0.9, "max_tokens": 12000},
        "index": {"temperature": 0.1, "top_p": 0.9, "max_tokens": 8000},
        "qa": {"temperature": 0.4, "top_p": 0.9, "max_tokens": 6000},
        "lint": {"temperature": 0.1, "top_p": 0.9, "max_tokens": 10000},
        "synthetic": {"temperature": 0.5, "top_p": 0.95, "max_tokens": 12000},
    },

    "directories": {
        "raw": "raw/",
        "wiki": "wiki/",
        "index": "wiki/index/",
        "concepts": "wiki/concepts/",
        "sources": "wiki/sources/",
        "derived": "wiki/derived/",
        "logs": "logs/",
        "eval": "eval/",
    },

    "search": {
        "method": "BM25",
        "k1": 1.5,
        "b": 0.75,
        "top_k": 20,
        "max_llm_results": 10,
    },

    "fine_tuning": {
        "method": "SFT_LoRA",
        "lr": 1e-4,
        "batch_size": 128,
        "epochs": 2,
        "weight_decay": 0.01,
        "warmup_ratio": 0.03,
        "max_seq_len": 8192,
        "optimizer": "AdamW",
        "seed": 42,
    },
}
```

---

**End of Document**