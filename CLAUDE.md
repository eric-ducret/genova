# CLAUDE.md — Variant Analysis Pipeline

This document is the authoritative reference for Claude Code when working on this project.
Read it fully before writing, editing, or refactoring any file.

---

## Project Overview

This is a clinical genomics pipeline that takes a list of genetic variants and a patient
phenotype description, gathers evidence from multiple sources per variant, and produces
a structured clinical report using a Small Language Model (SLM).

The pipeline is designed to run on a SLURM HPC cluster (CURNAGL, A100 40GB GPU node)
and is exposed as a FastAPI server — making it accessible interactively from a browser,
programmatically via HTTP, or as an sbatch job. The server is the single universal
entry point regardless of how compute resources were allocated.

**Core philosophy:**
- The framework handles orchestration. Tools handle one thing each.
- Adding a new tool = one `.py` file + one `.yaml` manifest. Nothing else changes.
- The SLM is a shared resource. Tools request it via context; they do not own it.
- Errors are informative, not silent. Failures appear in the report with context.

---

## Repository Structure

```
variant-pipeline/
│
├── CLAUDE.md                        ← this file
├── MIGRATION.md                     ← migration notes from original codebase
├── README.md                        ← user-facing quickstart and architecture
├── TOOL_DEVELOPMENT.md              ← guide for writing new tools
├── launch.sh                        ← SLURM sbatch script
├── client.py                        ← CLI client for submitting jobs from login node
├── server.py                        ← FastAPI entry point (thin, no business logic)
│
├── pipeline/
│   ├── __init__.py
│   ├── pipeline.py                  ← top-level orchestrator
│   ├── config.py                    ← global pipeline configuration (GENOME_BUILD, EXECUTION_MODE)
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── manifest.py              ← manifest loading, validation, gate resolution
│   │   ├── context.py               ← ToolContext dataclass
│   │   ├── executor.py              ← tool fan-out, error handling, mini-doc assembly
│   │   ├── normalizer.py            ← CSV/Excel ingestion and schema normalization
│   │   └── errors.py                ← typed pipeline exceptions
│   │
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── base.py                  ← LLMClient abstract base class
│   │   ├── vllm_client.py           ← vLLM HTTP client (primary, for A100 node)
│   │   ├── hf_client.py             ← HuggingFace pipeline client (fallback/dev)
│   │   └── registry.py              ← model registry, instantiate by name
│   │
│   ├── stages/
│   │   ├── __init__.py
│   │   ├── retrieval.py             ← per-variant tool fan-out stage
│   │   ├── compression.py           ← context compression stage
│   │   ├── reasoning.py             ← SLM step-by-step reasoning stage
│   │   └── conclusion.py            ← SLM structured report stage
│   │
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── base.py                  ← Tool, NetworkTool, SLMTool, RAGTool, BotTool
│   │   ├── litvar2.py               ← LitVar2 literature search (rsID + gene fallback)
│   │   ├── autopvs1.py              ← AutoPVS1 PVS1 criterion evaluation (LoF variants)
│   │   ├── spliceai.py              ← SpliceAI splice impact delta scores (Broad API)
│   │   ├── websearch.py
│   │   ├── websearch_agent.py
│   │   └── ncbi.py
│   │
│   └── manifests/
│       ├── litvar2.yaml
│       ├── autopvs1.yaml
│       ├── spliceai.yaml
│       └── websearch_agent.yaml
│
└── prompts/
    ├── retrieval.txt
    ├── compression.txt
    ├── reasoning.txt
    └── conclusion.txt
```

---

## Variant Schema

The pipeline normalizes any input CSV or Excel file to the following canonical schema.
Unknown columns are ignored. Missing columns are filled with `"NA"`.

### Canonical fields

| Field | Type | Example | Notes |
|---|---|---|---|
| `Variant` | str | `chr6:100896130 T>C` | Free-text genomic identifier |
| `Chromosome` | str | `chr6` | With or without `chr` prefix |
| `Position` | str | `100896130` | 1-based, hg19 |
| `RS_ID` | str | `rs28936388` | `NA` if not in dbSNP |
| `Ref_seq` | str | `T` | Reference allele; `NA` for complex indels |
| `Var_seq` | str | `C` | Alternate allele |
| `Type` | str | `SNV` | SNV, Insertion, Deletion, Indel, CNV… |
| `HGVS` | str | `NM_005068.3:c.744-2A>G p.?` | Full HGVS notation |
| `Zygosity` | str | `Heterozygous` | Heterozygous / Homozygous / Hemizygous |
| `Gene` | str | `SIM1` | HGNC gene symbol |
| `OMIM_phenotype` | str | `Gitelman syndrome` | Associated OMIM disease name |
| `OMIM_inheritance` | str | `Autosomal recessive` | Full inheritance string |
| `Inheritance` | str | `AR` | Short code: AR, AD, XL, … |
| `ClinVar_class` | str | `Pathogenic` | ClinVar clinical significance |
| `Allelic_balance` | float | `0.5393` | VAF / allele balance |
| `Frequency` | float | `1.19E-05` | Population allele frequency (gnomAD) |
| `CADD_score` | float | `26` | CADD PHRED score |

### Real example rows (from test dataset)

```
Variant=chr19:40730680_1 insG, Chromosome=chr19, Position=40730681, RS_ID=NA,
Ref_seq=NA, Var_seq=G, Type=Insertion, HGVS=NM_024877.4:c.305dup p.(Val103CysfsTer25),
Zygosity=Heterozygous, Gene=CCNP, OMIM_phenotype=NA, OMIM_inheritance=NA,
Inheritance=NA, ClinVar_class=NA, Allelic_balance=0.5417, Frequency=0, CADD_score=NA

Variant=chr6:100896130 T>C, Chromosome=chr6, Position=100896130, RS_ID=NA,
Ref_seq=T, Var_seq=C, Type=SNV, HGVS=NM_005068.3:c.744-2A>G p.?,
Zygosity=Heterozygous, Gene=SIM1, OMIM_phenotype=NA, OMIM_inheritance=NA,
Inheritance=AD, ClinVar_class=NA, Allelic_balance=0.383, Frequency=0, CADD_score=34
```

### Schema extensibility

The normalizer maps any incoming column name to a canonical field via `COLUMN_ALIASES`
in `core/normalizer.py`. To support a new input format, add aliases there.
The canonical schema is the single source of truth — tools always receive a normalized
variant dict, never raw CSV rows.

Additional computed or externally-fetched fields (e.g. gnomAD v4 AF, SpliceAI score)
can be injected by a tool and will appear in `context.all_outputs` for downstream tools.

---

## LLM Layer

### Design

All SLM access goes through a single abstract `LLMClient`. Tools never instantiate
models directly. The client is created once at server startup and injected into
`ToolContext`.

```python
# pipeline/llm/base.py
from abc import ABC, abstractmethod

class LLMClient(ABC):
    @abstractmethod
    def generate(
        self,
        system: str,
        user: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        enable_thinking: bool = False,   # always False for Qwen3.5 in this pipeline
    ) -> str: ...

    @abstractmethod
    def is_available(self) -> bool: ...
```

### Implementations

**`vllm_client.py`** — primary client for A100 node.
Calls the vLLM OpenAI-compatible HTTP endpoint (`/v1/chat/completions`).
Supports async via `httpx.AsyncClient`. Used in production.

**`hf_client.py`** — fallback for local dev without GPU.
Wraps `transformers.pipeline("text-generation")`.
Same interface, slower. Useful for testing tool logic without a running vLLM server.

### Model registry

```python
# pipeline/llm/registry.py
MODELS = {
    "qwen3.5-9b": {
        "client": "vllm",
        "model_id": "Qwen/Qwen3.5-9B",
        "enable_thinking": False,      # MUST be False for Qwen3.5 in this pipeline
        "max_tokens_default": 1024,
    },
    # add new models here
}

def get_client(model_name: str, **kwargs) -> LLMClient:
    ...
```

**Qwen3.5 note:** `enable_thinking` must always be `False` in this pipeline.
The thinking mode produces chain-of-thought tokens that bloat the context and
break downstream parsing. This is enforced at the registry level, not per-tool.

---

## Tool System

### Tool types

| Class | Use when |
|---|---|
| `Tool` | Pure deterministic logic, no network, no SLM |
| `NetworkTool` | HTTP fetch, no SLM |
| `SLMTool` | Calls the SLM for summarization, filtering, or judgment |
| `ReActTool` | Full ReAct loop with tool registry (current websearch agent) |
| `RAGTool` | Vector DB retrieval (interface ready, implementation pluggable) |
| `BotTool` | Browser automation (interface ready, Playwright/Selenium pluggable) |

All inherit from `Tool`. Subclasses add capabilities; they do not change the contract.

### Tool contract

Every tool must implement:

```python
class MyTool(Tool):                      # or NetworkTool, SLMTool, etc.
    name        = "my_tool"              # matches manifest name
    description = "One sentence."        # shown in logs and TOOL_DEVELOPMENT.md

    def gate(self, variant: dict, context: "ToolContext") -> bool:
        """Return False to skip this tool for this variant. Default: always run."""
        return True

    def run(self, variant: dict, context: "ToolContext") -> str | None:
        """
        Execute the tool for one variant.
        Return a string block to include in the mini-doc, or None if no result.
        Raise a PipelineError subclass on failure (never return an error string).
        """
        raise NotImplementedError
```

### ToolContext

Every tool receives a `ToolContext` instance. This is the only way tools access
shared resources.

```python
@dataclass
class ToolContext:
    # Variant being processed
    variant: dict                        # normalized variant dict

    # Patient information
    patient_phenotype: str               # free-text phenotype from user

    # Pipeline state
    all_outputs: dict[str, str | None]   # {tool_name: output} for tools that ran before this one
    variant_report: str                  # accumulated mini-doc so far (for compression tools)
    variant_index: int                   # 0-based index of this variant
    total_variants: int                  # total number of variants in this run

    # Shared resources (injected at startup, never instantiated by tools)
    llm: LLMClient                       # call context.llm.generate(system, user, max_tokens)
    db: VectorStore | None               # None if no DB configured
    browser: BrowserSession | None       # None if no bot configured

    # Convenience
    def field(self, name: str) -> str:
        """Extract a field from the variant dict. Returns 'NA' if missing."""
        return self.variant.get(name, "NA")
```

### Gate system

Gates control whether a tool runs for a given variant. They are evaluated by the
executor before calling `run()`. A gate can be:

**1. Python method** (most flexible):
```python
def gate(self, variant: dict, context: ToolContext) -> bool:
    return context.field("Type").lower() in {"insertion", "deletion", "frameshift"}
```

**2. Manifest expression** (simple cases, no custom code needed):
```yaml
gate:
  field: Type
  operator: in
  value: [Insertion, Deletion, frameshift, nonsense, splice_site]
```

**3. SLM gate** (ambiguous cases — calls SLM with max_tokens=5):
```yaml
gate:
  type: slm
  prompt: "Should PVS1 be evaluated for this variant? Type={Type} HGVS={HGVS}. Answer YES or NO."
```

The Python method always takes precedence over the manifest gate if both are defined.

Manifest gates only support single-field conditions with a single operator. For OR logic
or multi-field conditions a Python `gate()` method is required, and the `gate:` block
should be omitted from the manifest entirely. `litvar2_summary` is the current example:
it gates on RS_ID valid **or** Gene present — this cannot be expressed in a single YAML
expression, so `litvar2.yaml` has no `gate:` block and the logic lives entirely in the
Python `gate()` method.

### Compression

A tool can declare that its output should be compressed when the variant count
exceeds a threshold. This is configured in the manifest:

```yaml
compress:
  enabled: true
  threshold: 8          # compress if total_variants > 8
  strategy: slm         # slm | truncate | first_n_lines
  max_tokens: 300       # target length after compression
```

Alternatively, the threshold can be inferred from the variant table at runtime
by setting `threshold: auto` — the executor will compute a per-tool token budget
based on total variant count and available context window.

Compression is applied by the executor after `run()` returns, before appending
to the mini-doc. The tool itself does not need to handle this.

### Error handling

Tools must raise typed exceptions from `pipeline/core/errors.py`.
Never return error strings. Never raise bare `Exception`.

```python
# pipeline/core/errors.py

class PipelineError(Exception):
    """Base class for all pipeline errors."""

class ToolFetchError(PipelineError):
    """Network request failed. Include URL and status code."""

class ToolParseError(PipelineError):
    """Could not parse response from external source."""

class ToolGateError(PipelineError):
    """Gate evaluation failed (not a skip — an actual error)."""

class SLMError(PipelineError):
    """SLM call failed or returned unparseable output."""

class ManifestError(PipelineError):
    """Manifest file is missing, malformed, or references unknown tool."""

class NormalizationError(PipelineError):
    """Input CSV/Excel could not be parsed or normalized."""
```

The executor catches all `PipelineError` subclasses per tool, logs the error,
and writes a failure note into the mini-doc:

```
AUTOPVS1 [FAILED — ToolFetchError]: Connection timeout after 15s (https://autopvs1.bgi.com/...)
```

This failure note is visible to the SLM in the reasoning stage, which can then
note the missing evidence in the report rather than reasoning from a gap.

### Implemented tools

| Tool | Class | Order | Gate |
|---|---|---|---|
| `litvar2_summary` | `SLMTool` | 1 (parallel) | RS_ID valid **or** Gene present (Python gate) |
| `spliceai` | `NetworkTool` | 1 (parallel) | Python gate — skips synonymous, intergenic, UTR, unresolvable coords |
| `autopvs1` | `NetworkTool` | 2 (parallel) | Python gate — LoF/frameshift/splice variants only |
| `websearch_agent` | `ReActTool` | 3 (serial) | None — internal pre-loop checkpoint decides whether to search |
| `ncbi` | `NetworkTool` | — | — |

**`litvar2_summary`** runs two paths depending on what variant fields are available:
- **rsID present and valid** (`rs\d+`) — variant-level search via the LitVar2 publications endpoint; fetches PMIDs, filters titles with the SLM using the resolved disease query, fetches abstracts, and returns a summary with inline PMID citations prefixed `LitVar2 evidence summary for {rsid}:`.
- **rsID absent, Gene present** — gene-level fallback via `GET /variant/search/gene/{gene}`; collects and deduplicates PMIDs across the first 10 variants returned for that gene, then runs the same filter → abstracts → summary pipeline. Output is prefixed `LitVar2 gene-level evidence for {gene} (no rsID):` so downstream reasoning stages can distinguish the two.

`_disease_query` is resolved once per pipeline run via a single SLM call (`_resolve_disease_query`), cached as an instance attribute, and reused for all subsequent variants. Both paths use this resolved query for title filtering; the full `patient_phenotype` string is passed to the final summarisation step for richer output. `_gene_fallback` resolves `_disease_query` itself if called directly (e.g. from a unit test) without requiring `run()` to have been called first.

**`spliceai`** queries the Broad Institute SpliceAI API and returns delta scores
(DS_AG, DS_AL, DS_DG, DS_DL) for acceptor/donor gain and loss, along with the
`sai10kPredictions` clinical interpretation block (aberration type, confidence,
frameshift description). Only the MANE Select transcript is reported. Coordinates
and genome build are sourced from the canonical variant dict fields; build is read
from `pipeline/config.py` (`GENOME_BUILD`). Skips synonymous, intergenic, and UTR
variants via Python gate. Output labels are enriched for SLM readability
(plain-English aberration names, explicit delta score scale).

---

## Manifest System

Each tool has a corresponding YAML manifest in `pipeline/manifests/`.
The manifest declares metadata, gate, ordering, and compression config.
The executor loads all manifests at startup and resolves tool instances by `name`.

### Manifest schema

```yaml
# pipeline/manifests/example.yaml

name: my_tool                    # must match Tool.name in the Python class
description: "One sentence."     # used in logs and documentation
enabled: true                    # set false to disable without deleting

order: 2                         # execution order (lower = earlier); ties broken alphabetically
parallel: true                   # whether this tool can run concurrently with others at same order

gate:                            # optional; omit to always run, or when gate() is in Python
  field: RS_ID                   # variant field to evaluate
  operator: not_na               # operators: equals, not_equals, in, not_in, not_na, na, slm
  value: ~                       # not needed for not_na / na

compress:                        # optional compression of this tool's output
  enabled: false
  threshold: auto                # integer or "auto"
  strategy: slm                  # slm | truncate | first_n_lines
  max_tokens: 300

timeout: 15                      # seconds before ToolFetchError is raised

retry:
  attempts: 2
  backoff: 2.0                   # seconds between retries
```

### Execution order

Tools at the same `order` value run concurrently if `parallel: true` and the
executor is in async mode. Tools at different order values run sequentially
(i.e. order 1 tools all complete before order 2 tools start).

This allows expressing dependencies: a compression tool at order 99 sees the
full accumulated `variant_report` from all earlier tools.

### Parallelism modes

Configured globally in `pipeline/config.py`:

```python
EXECUTION_MODE = "async"   # "async" | "sequential"
```

- `sequential` — variants processed one by one, tools within each variant run in order.
  Safest, easiest to debug, sufficient for ≤10 variants.
- `async` — variants processed concurrently via `asyncio.gather`.
  Tools marked `parallel: true` at the same order level run concurrently within each variant.
  Recommended for production on A100 node with vLLM serving async requests.

Switching between modes requires no changes to tool code.

---

## Pipeline Stages

### Stage 1 — Retrieval (`stages/retrieval.py`)

For each variant, loads all enabled manifests, evaluates gates, sorts by order,
and executes tools. Assembles outputs into a per-variant mini-doc string.

```
variant dict
    → gate evaluation (skip if False)
    → tool.run(variant, context)
    → compression (if configured and threshold met)
    → append to mini-doc
```

All mini-docs are concatenated into the augmented context document passed to Stage 2.

### Stage 2 — Compression (`stages/compression.py`)

Optional global compression stage. Runs after all per-variant mini-docs are assembled.
Triggered when total assembled context exceeds a token threshold (configurable).
Summarizes each mini-doc independently to fit within the SLM context window.

This is separate from per-tool compression (which compresses individual tool outputs).
Global compression compresses entire per-variant sections.

### Stage 3 — Reasoning (`stages/reasoning.py`)

Single SLM call. Takes the full augmented context and produces step-by-step
clinical reasoning for each variant. Does not produce a classification.
Prompt loaded from `prompts/reasoning.txt`.

### Stage 4 — Conclusion (`stages/conclusion.py`)

Single SLM call. Takes augmented context + reasoning output and produces the
structured clinical report. Prompt loaded from `prompts/conclusion.txt`.

Output format per variant:
```
# Variant [N] — [GENE] ([HGVS])
**Molecular mechanism:** ...
**Phenotype fit:** match | mismatch | uncertain — ...
**Inheritance check:** consistent | inconsistent | insufficient data
**Evidence strength:** strong | moderate | weak | absent
**ACMG criteria:** ...
**Comment:** ...
```

Followed by a `# Clinical Conclusion` section.

---

## Deployment

### SLURM (production)

`launch.sh` starts two processes in the same job allocation:
1. vLLM server on the GPU (serves the model via OpenAI-compatible HTTP)
2. FastAPI server on CPU (pipeline logic, calls vLLM for SLM steps)

```bash
#!/bin/bash
#SBATCH --job-name=variant-pipeline
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00

# Start vLLM server
vllm serve Qwen/Qwen3.5-9B \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --port 8001 &

# Wait for vLLM to be ready
sleep 60

# Start FastAPI server
uvicorn server:app --host 0.0.0.0 --port 8000
```

### Local / dev

Use `hf_client` by setting `LLM_BACKEND=hf` in config. No vLLM needed.
Slower but identical pipeline behavior.

### Client (login node)

```bash
python client.py \
    --server http://<node>:8000 \
    --csv variants.csv \
    --phenotype "Obesity, intellectual disability, hyperphagia" \
    --output report.txt
```

---

## Adding a New Tool — Quick Reference

Full guide in `TOOL_DEVELOPMENT.md`. Minimum steps:

1. Create `pipeline/tools/my_tool.py` inheriting from the appropriate base class
2. Implement `gate()` (optional) and `run()`
3. Create `pipeline/manifests/my_tool.yaml` with `name`, `order`, `enabled`
4. Register in `pipeline/tools/__init__.py`

No other files need to change.

### Tool template

```python
# pipeline/tools/my_tool.py
from pipeline.tools.base import SLMTool          # or Tool, NetworkTool, RAGTool, BotTool
from pipeline.core.context import ToolContext
from pipeline.core.errors import ToolFetchError, ToolParseError


class MyTool(SLMTool):
    name        = "my_tool"
    description = "One sentence describing what this tool fetches or computes."

    def gate(self, variant: dict, context: ToolContext) -> bool:
        # Return False to skip. Access variant fields via context.field("FieldName").
        return context.field("RS_ID") != "NA"

    def run(self, variant: dict, context: ToolContext) -> str | None:
        rsid = context.field("RS_ID")

        # Network call example
        try:
            response = self._get(f"https://example.com/api/{rsid}", timeout=self.timeout)
        except Exception as e:
            raise ToolFetchError(f"Failed to fetch {rsid}: {e}") from e

        # Parse
        try:
            data = response.json()
        except Exception as e:
            raise ToolParseError(f"Could not parse response for {rsid}: {e}") from e

        if not data:
            return None

        # Optional SLM call (only available in SLMTool subclasses)
        summary = context.llm.generate(
            system="Summarize this evidence in 2 sentences.",
            user=str(data),
            max_tokens=200,
        )

        return f"MY_TOOL EVIDENCE ({rsid}):\n{summary}"
```

### Manifest template

```yaml
# pipeline/manifests/my_tool.yaml
name: my_tool
description: "One sentence."
enabled: true
order: 3
parallel: true
gate:
  field: RS_ID
  operator: not_na
compress:
  enabled: true
  threshold: auto
  strategy: slm
  max_tokens: 200
timeout: 15
retry:
  attempts: 2
  backoff: 2.0
```

---

## Interface-Ready Tool Types

These base classes exist and are importable but have no concrete implementations yet.
They define the interface contract so tools can be written without knowing the
underlying library.

### RAGTool

```python
class RAGTool(SLMTool):
    """
    Base class for tools that retrieve from a vector database.
    context.db is a VectorStore instance (Qdrant by default).
    
    The VectorStore interface:
        context.db.search(query: str, collection: str, top_k: int, filters: dict) -> list[Chunk]
        context.db.collections() -> list[str]
    
    Chunk fields: text, score, metadata (dict with source, gene, rsid, etc.)
    
    For variant queries, always pass gene and/or rsid as filters to avoid
    pure semantic drift on short identifiers.
    """
```

### BotTool

```python
class BotTool(Tool):
    """
    Base class for tools that use browser automation.
    context.browser is a BrowserSession instance.
    
    The BrowserSession interface:
        context.browser.get(url: str) -> str          # returns page text
        context.browser.click(selector: str)
        context.browser.fill(selector: str, value: str)
        context.browser.wait_for(selector: str, timeout: int)
    
    Underlying implementation (Playwright or Selenium) is injected at startup.
    BotTool subclasses never import Playwright or Selenium directly.
    
    Example use case: Silicon (https://silicon.com) or any JS-rendered
    clinical database that does not expose an API.
    """
```

---

## Key Design Rules

These rules must be respected in all new code:

1. **Tools are stateless with respect to variant data.** Tools must never store
   per-variant results on `self` between calls — all variant-scoped state lives in
   `ToolContext`. The one accepted exception is run-level caching of expensive
   one-time computations that are identical across all variants (e.g. `_disease_query`
   in `litvar2_summary`, derived solely from `patient_phenotype`). Such cached values
   must be pipeline-run-scoped, variant-independent, and initialised in `__init__`
   as `None`.

2. **Tools never load models.** The LLM client is always accessed via `context.llm`.

3. **Tools never open DB connections.** Always via `context.db`.

4. **Errors are typed.** Always raise from `pipeline/core/errors.py`.
   Never return error strings. Never raise bare `Exception`.

5. **Prompts live in files.** All SLM prompts longer than one sentence belong in
   `prompts/`. Load them with `Path("prompts/my_prompt.txt").read_text()`.
   Do not hardcode multi-line prompts in Python.

6. **Manifests are the source of truth for ordering and gating.**
   Do not hardcode tool order or conditions in stage code.

7. **`enable_thinking` is always False.** Enforced at the LLM registry level.
   No tool or stage should pass `enable_thinking=True`.

8. **Output is free-form string or None.** Tools do not need to produce structured
   output. The SLM in stages 3 and 4 handles synthesis.

9. **Compression is the executor's job.** Tools return their full output.
   The executor applies compression based on the manifest config.

10. **The server knows nothing about variants.** `server.py` parses the request,
    calls `pipeline.run()`, and returns the result. All logic is in `pipeline/`.

---

## Dependencies

### Core runtime
```
torch>=2.0
transformers>=4.40
fastapi
uvicorn
pydantic
httpx           # vLLM async client
vllm            # production inference (A100 node)
```

### Pipeline tools
```
requests
beautifulsoup4
ddgs            # DuckDuckGo search (was duckduckgo-search)
pandas
openpyxl        # Excel support in normalizer
```

### Optional (future tools)
```
qdrant-client   # RAG / vector DB
sentence-transformers  # embeddings for RAG
playwright      # BotTool browser automation
```

### Notes
- `torch` and `transformers` are only required by `hf_client.py` (dev/fallback mode)
- In production (vLLM mode), the FastAPI server process does NOT load torch directly —
  the model runs in the separate vLLM subprocess
- `vllm` must be installed in the same environment as the model server, not necessarily
  the FastAPI server environment
