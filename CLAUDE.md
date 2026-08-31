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
- Adding a new tool = one `.py` file + one `.yaml` manifest + one import line in `pipeline/tools/__init__.py` + one instance line in `Pipeline._tools` (`pipeline/pipeline.py`).
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
├── batch.sh                         ← SLURM sbatch script for batch folder processing
├── client.py                        ← CLI client for submitting jobs from login node
├── batch.py                         ← batch runner: process a folder of variant files
├── server.py                        ← FastAPI entry point (thin, no business logic)
│
├── analyse_results/
│   ├── analyse_results.py           ← post-processing: per-case CSV from report.txt files
│   ├── analyse_results.sh           ← SLURM wrapper for analyse_results.py
│   ├── analysis_notebook.ipynb      ← Jupyter notebook for metric visualisation
│   └── augment_litvar2.py           ← (auxiliary) LitVar2 enrichment for result tables
│
├── batch_pheno.py                   ← cohort batch runner: pheno_prompts.csv + jobs_folder → per-patient reports
├── batch_pheno_vllm.sh              ← SLURM sbatch script for one batch_pheno.py batch (Phase 2, run N times in parallel)
├── warmup_vllm.sh                   ← SLURM sbatch script: Phase 1.5 sequential NCBI/disease-query pre-fetch
│
├── pipeline/
│   ├── __init__.py
│   ├── pipeline.py                  ← top-level orchestrator
│   ├── config.py                    ← global pipeline configuration (GENOME_BUILD, EXECUTION_MODE, DEFAULT_MODEL, FORCE_GENERIC_PHENOTYPE)
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── manifest.py              ← manifest loading, validation, gate resolution
│   │   ├── context.py               ← ToolContext dataclass
│   │   ├── executor.py              ← tool fan-out, error handling, mini-doc assembly, retry
│   │   ├── normalizer.py            ← CSV/Excel ingestion, SLM-driven column mapping, parental AB extraction
│   │   ├── errors.py                ← typed pipeline exceptions
│   │   ├── moi.py                   ← mode-of-inheritance classification (gene mode, chromosome, phenotype pertinence)
│   │   ├── segregation.py           ← deterministic allelic-balance segregation/phase/zygosity classifiers
│   │   ├── acmg_sf.py                ← ACMG Secondary Findings v3.2 (81-gene) actionable-variant detection
│   │   ├── citations.py             ← shared anti-hallucination citation validator (used by conclusion + all MOI stages)
│   │   ├── acmg_points.py           ← shared point-arithmetic regex utility (classify(), adjust_points_line()) for the validators below
│   │   ├── acmg_pp2_bp1.py          ← post-hoc validator: strips PP2/BP1 if the SLM's claim doesn't match gene-level ClinVar consequence counts
│   │   ├── acmg_pp3.py              ← post-hoc validator: strips PP3 if fewer than 2 in-silico tools actually cross their documented threshold
│   │   ├── acmg_pp4.py              ← post-hoc validator: strips PP4 if its own justification names a hardcoded heterogeneous condition
│   │   ├── acmg_ps1_pm5.py          ← post-hoc validator: inserts a missed PS1/PM5 line when residue-level evidence supports it but the SLM omitted it
│   │   ├── acmg_bp6.py              ← post-hoc validator: strips BP6 if its own justification text self-contradicts (names VUS/pathogenic instead of benign)
│   │   ├── build_detection.py       ← genome-build (hg19/hg38) auto-detection via ClinGen majority vote over a variant sample — utility, not yet called from pipeline.py
│   │   ├── clinvar_reference.py     ← deterministically appends a ClinVar Variation-ID/URL line to conclusion/MOI blocks
│   │   └── protein_change.py        ← shared HGVS protein-change parsing (extract_residue_queries), used by websearch_agent.py and clinvar_residue_search.py
│   │
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── base.py                  ← LLMClient abstract base class
│   │   ├── vllm_client.py           ← vLLM HTTP client (primary, for A100 node) — multi-model, Gemma artifact stripping
│   │   ├── hf_client.py             ← HuggingFace pipeline client (fallback/dev)
│   │   └── registry.py              ← model registry, instantiate by name (MODELS dict, DEFAULT_MODEL env var)
│   │
│   ├── stages/
│   │   ├── __init__.py
│   │   ├── retrieval.py             ← per-variant tool fan-out stage (+ ALLELIC BALANCE block when parental_ab present)
│   │   ├── first_triage.py          ← fast SLM keep/discard decision per variant
│   │   ├── reasoning.py             ← run_reasoning() (Call 1) + run_second_triage() (Call 2, INCLUDE/EXCLUDE)
│   │   ├── cross_analysis.py        ← gene-level compound-het / interaction analysis
│   │   ├── conclusion.py            ← Layer 2: per-variant BASE ACMG score (Tavtigian 2020 point system)
│   │   ├── moi_denovo.py            ← Layer 3: PS2/PM6 de novo delta (AD/AD_AR/XLD/XLR/XLD_XLR/XL genes)
│   │   ├── moi_dominant.py          ← Layer 4: PP1 cosegregation delta (AD/AD_AR/XLD genes, inherited not de novo)
│   │   ├── moi_recessive.py         ← Layer 5: PM3 delta — run_pair() compound-het, run_solo() confirmed homozygous
│   │   ├── moi_xlinked.py           ← Layer 6: PP1 delta via XLR/XLD allelic-balance pattern (chrX genes)
│   │   ├── actionable.py            ← ACMG-SF actionable-findings report section (independent of phenotype/MOI)
│   │   └── final_conclusion.py      ← Layer 8: cross-MOI synthesis, two-pass draft (clinical_conclusion.txt) + revise (final_conclusion_revise.txt)
│   │
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── base.py                  ← Tool, NetworkTool, SLMTool, RAGTool, BotTool
│   │   ├── litvar2.py               ← literature search: gene-first three-track PubMed + LitVar2 rsID; CGD inheritance lookup
│   │   ├── autopvs1.py              ← AutoPVS1 PVS1 criterion evaluation (LoF variants)
│   │   ├── spliceai.py              ← SpliceAI splice impact delta scores (Broad API)
│   │   ├── gnomad_constraint.py     ← gnomAD gene constraint (pLI, LOEUF) — gene-cached, for PVS1 mechanism plausibility
│   │   ├── gnomad_frequency.py      ← live variant-level gnomAD AF + het/hom carrier counts (PM2/BS1 fallback when CSV Frequency is blank)
│   │   ├── clinvar_gene_stats.py    ← ClinVar gene-level P/LP consequence counts (PP2/BP1) + per-variant submission tally (PS3)
│   │   ├── clinvar_residue_search.py ← deterministic residue-level ClinVar query — primary PS1/PM5 evidence source
│   │   ├── clingen_allele.py        ← ClinGen Allele Registry cross-reference (CAid → ClinVar/dbSNP/gnomAD/ExAC)
│   │   ├── websearch.py             ← WebSearchTool + WebFetchTool (sub-tools for ReAct agent)
│   │   ├── websearch_agent.py       ← WebSearchAgentTool: full ReAct web-search loop + residue-level PS1/PM5 search fallback
│   │   └── ncbi.py                  ← NCBIFetchTool (sub-tool for ReAct agent)
│   │
│   └── manifests/
│       ├── litvar2.yaml             ← order 1
│       ├── spliceai.yaml            ← order 1
│       ├── gnomad_constraint.yaml   ← order 1
│       ├── gnomad_frequency.yaml    ← order 1
│       ├── clinvar_gene_stats.yaml  ← order 1
│       ├── clinvar_residue_search.yaml ← order 1
│       ├── clingen_allele.yaml      ← order 1
│       ├── autopvs1.yaml            ← order 2
│       └── websearch_agent.yaml     ← order 3 (serial — consumes all order-1/2 outputs via context.all_outputs)
│
└── prompts/
    ├── retrieval.txt
    ├── compression.txt
    ├── first_triage.txt
    ├── reasoning.txt
    ├── second_triage.txt
    ├── conclusion.txt               ← full Tavtigian 2020 ACMG point-scoring rubric (base, MOI-agnostic)
    ├── cross_analysis.txt
    ├── moi_denovo.txt
    ├── moi_dominant.txt
    ├── moi_recessive.txt            ← compound-het pair
    ├── moi_recessive_homozygous.txt ← confirmed-homozygous solo
    ├── moi_xlinked.txt
    ├── moi_mode_classification.txt  ← LLM tier-3 gene-inheritance-mode fallback prompt (core/moi.py)
    ├── actionable_variants.txt
    ├── clinical_conclusion.txt      ← Layer 8 draft
    └── final_conclusion_revise.txt  ← Layer 8 fact-check pass
```

**Note:** `launch.sh` / `batch.py` / `client.py` (below, under Deployment) are the original single-shot/server-mode
entry points. For a full cohort run, `batch_pheno.py` + `warmup_vllm.sh` + `batch_pheno_vllm.sh` are the actual
production workflow — see **Deployment → Cohort batch run** below.

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
| `Allelic_balance` | float | `0.5393` | VAF / allele balance (proband only — see Parental allelic balance below) |
| `Frequency` | float | `1.19E-05` | Population allele frequency (gnomAD) |
| `CADD_score` | float | `26` | CADD PHRED score |
| `Transcript` | str | `NM_000303.3` | RefSeq transcript, only when given as its own column separate from HGVS |
| `REVEL_score` | float | `0.82` | REVEL pathogenicity score |
| `SIFT_score` | str/float | `0.01` | SIFT score or prediction |
| `PolyPhen2_score` | str/float | `0.97` | PolyPhen-2 score or prediction |
| `AlphaMissense_score` | float | `0.71` | AlphaMissense pathogenicity score |
| `SpliceAI_score` | float | `0.85` | Pre-computed splice-impact score, if already present in the input |

### Column mapping

Column-to-field mapping is **SLM-driven** (`normalizer.py::_map_columns_llm`) when an
`LLMClient` is passed to `normalize_upload()` — the model inspects the header row plus
one sample data row and classifies **every** header column into a canonical field,
including the allelic-balance columns (proband/mother/father), producing a
human-readable `header_mapping_summary` surfaced at the top of the report. When no
`llm` is passed (e.g. the CLI convenience entry point), falls back to the deterministic
`COLUMN_ALIASES` dict mapper.

Allelic-balance columns used to be claimed by a hardcoded regex
(`^allelic\s+balance\s*-`) *before* the SLM mapper ever saw them. That regex was
space-strict and, on a real upload that used `Allelic_balance_1`/`Allelic_balance_2`
naming, matched nothing — silently discarding trio parental AB data for every variant
in the file. The SLM mapper now classifies these columns itself (see the
`Allelic_balance`/`Allelic_balance_mother`/`Allelic_balance_father` entries in
`_FIELD_DESCRIPTIONS`, which describe the matching rules: explicit naming, sample-ID
suffix, or positional fallback when naming gives no parent identity).

### Parental allelic balance

Primary path: `_build_parental_ab_from_mapped()` reads parental AB straight from the
SLM's own column mapping (`Allelic_balance`/`Allelic_balance_mother`/
`Allelic_balance_father`), so it works under whatever naming convention the input file
happens to use. Fallback path (only when the SLM path finds no proband AB column at
all, or when `llm=None`): `extract_parental_ab()` — a regex over raw headers matching
`"allelic balance"` (case/separator-insensitive substring, so `Allelic_balance_1` and
`Allelic balance - proband` both match). Either path returns a `parental_ab: list[dict]`
alongside the normalized variants — one dict per variant, keys
`"proband"`/`"mother"`/`"father"` (3 columns), `"proband"`/`"extra1"` or the named
parent (2 columns), or just `"proband"` (1 column). This is the data source for the
entire segregation/MOI machinery — see Mode-of-Inheritance & ACMG Scoring below.

`normalize_upload(raw, filename, llm=None)` returns
`(variants, raw_rows, parental_ab, header_mapping_summary)` — a 4-tuple; all three
call sites (`server.py`, `batch.py`, `batch_pheno.py`) unpack all four and thread
`parental_ab`/`header_mapping_summary` into `Pipeline.run()`.

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

Every request includes `"chat_template_kwargs": {"enable_thinking": False}` — this
is the vLLM equivalent of the `enable_thinking=False` flag that the HF tokenizer
applies via `apply_chat_template`. Without it, Qwen3.5-9B generates plain-text
"Thinking Process:" preambles in non-thinking mode that contaminate downstream
context and break output parsers. A `_strip_thinking_header()` regex is also applied
to every response as a defensive fallback.

**`hf_client.py`** — fallback for local dev without GPU.
Wraps `transformers.pipeline("text-generation")`.
Same interface, slower. Useful for testing tool logic without a running vLLM server.
Applies `enable_thinking=False` via `tokenizer.apply_chat_template` — this is what
prevents thinking preambles in HF mode and was the reference for the vLLM fix.

### Model registry

```python
# pipeline/llm/registry.py
MODELS = {
    "qwen3.5-9b": {
        "client":             "vllm",
        "model_id":           "Qwen/Qwen3.5-9B",
        "enable_thinking":    False,   # MUST remain False — thinking tokens break downstream parsing
        "max_tokens_default": 1024,
        "vllm_extra_payload": {"chat_template_kwargs": {"enable_thinking": False}},
    },
    "gemma4-e4b": {
        "client":             "vllm",
        "model_id":           "google/gemma-4-E4B-it",
        "enable_thinking":    False,
        "max_tokens_default": 1024,
        "vllm_extra_payload": {},   # -it model has its own chat template + correct EOS handling
    },
    # add new models here following the same pattern
}

def get_client(model_name: str, **kwargs) -> LLMClient:
    ...
```

`vllm_extra_payload` is merged verbatim into every vLLM chat-completions request for that model —
this is how model-specific quirks (Qwen's `enable_thinking` flag, a future model's own extra params)
are handled without branching in `vllm_client.py` itself.

**Model selection:** `pipeline/config.py`'s `DEFAULT_MODEL = os.environ.get("VARIANT_MODEL", "qwen3.5-9b")`.
Override per-submission with `VARIANT_MODEL=gemma4-e4b sbatch batch_pheno_vllm.sh ...` (see
`batch_pheno_vllm.sh`, which resolves the HF model ID from this same registry so the `vllm serve`
command line and the pipeline's client always agree on which model is running).

**Qwen3.5 note:** `enable_thinking` must always be `False` in this pipeline.
The thinking mode produces chain-of-thought tokens that bloat the context and
break downstream parsing. For the HF client this is enforced via
`tokenizer.apply_chat_template(..., enable_thinking=False)`. For the vLLM client
it is enforced via `vllm_extra_payload`'s `chat_template_kwargs` — the two are
equivalent but use different APIs. `vllm_client.py` also applies
`_strip_thinking_header()` (Qwen preamble) and `_strip_gemma_artifacts()`
(Gemma `<start_of_turn>`/`<end_of_turn>` leakage) as defensive post-processing
on every response, regardless of model.

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
    db: VectorStore | None = None        # None if no DB configured
    browser: BrowserSession | None = None  # None if no bot configured

    # Run-level metadata
    genome_build: str = "hg19"           # reference genome build for this run
    raw_fields: dict = field(default_factory=dict)  # full original CSV row for this variant

    # Convenience
    def field(self, name: str) -> str:
        """Extract a field from the variant dict. Returns 'NA' if missing or falsy."""
        return self.variant.get(name, "NA") or "NA"
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
| `gnomad_constraint` | `NetworkTool` | 1 (parallel) | Manifest gate — Gene present |
| `gnomad_frequency` | `NetworkTool` | 1 (parallel) | Manifest gate — resolvable coordinates |
| `clinvar_gene_stats` | `NetworkTool` | 1 (parallel) | Manifest gate — Gene present |
| `clinvar_residue_search` | `NetworkTool` | 1 (parallel) | Manifest gate — a protein-residue position can be parsed from HGVS |
| `clingen_allele` | `NetworkTool` | 1 (parallel) | Python gate — a usable HGVS/genomic query can be built |
| `autopvs1` | `NetworkTool` | 2 (parallel) | Python gate — LoF/frameshift/splice variants only |
| `websearch_agent` | `ReActTool` | 3 (serial) | Python gate — hard-skips ClinVar benign/likely-benign and common variants (AF > 1%); pre-loop checkpoint handles remaining cases |

**`litvar2_summary`** runs a gene-first three-track search:

- **Track 1 — Gene + patient phenotype (primary — always runs when Gene is present)** — queries PubMed `esearch` with `({gene}[tiab] OR {gene}[Gene]) AND {disease_query}`. If no disease co-occurrence is found, retries with gene-only to distinguish "no literature at all" from "gene unrelated to disease". Returns an explicit `"NO DISEASE LINK"` message on zero relevant hits — providing a clear negative signal for triage. Output header includes PubMed's `querytranslation` and pool counts for downstream traceability. When total results exceed `MAX_PMIDS`, a second call sorted by publication date adds recent papers not in the relevance window.
- **Track 2 — Gene + known disease term (supplemental — runs when a disease term is available)** — bridges the terminology gap between how the patient's condition is described and how OMIM/CGD name the canonical disease for that gene. Disease term source priority: (a) `OMIM_phenotype` from the variant dict; (b) NHGRI Clinical Genomic Database (CGD) conditions for the gene — downloaded once per process and cached at the class level (`_cgd_table`). Relevance filtering anchors on the known condition; summarisation is framed on the patient phenotype so output stays clinically grounded. Track silently skipped when no disease term is available or no relevant papers are found.
- **Track 3 — rsID variant-level search (supplemental — runs when RS_ID is valid)** — queries the LitVar2 variant publications endpoint for variant-specific literature. Omitted from output if no relevant papers are found. Output header identifies LitVar2 as source, distinguished from the gene-level PubMed blocks.

All tracks that yield evidence are combined in the output separated by `---`. `_disease_query` is resolved once per pipeline run via a single SLM call (`_resolve_disease_query`), cached as an instance attribute, and reused for all variants. The resolver returns 2–4 PubMed-compatible disease terms as a PubMed OR expression, preferring MeSH-indexed terms for broad recall.

**`autopvs1`** uses a HGVS-first query strategy: when a valid transcript HGVS is available (`NM_xxx:c.xxx`), it queries AutoPVS1 via `/search?q={hgvs}` so AutoPVS1 resolves coordinates internally — immune to stale CSV coords. Falls back to VCF-style coords (`/variant/{build}/{chrom}-{pos}-{ref}-{alt}`) when HGVS is absent or the search path fails. Results are discarded when AutoPVS1 returns `variant_type="Intergenic"` or the returned gene does not match the expected gene. `pvs1_applicable` is `True` only when a flowchart is present in the response and no "incompatible with recommendations" message is found.

**`spliceai`** queries the Broad Institute SpliceAI API and returns delta scores
(DS_AG, DS_AL, DS_DG, DS_DL) for acceptor/donor gain and loss, along with the
`sai10kPredictions` clinical interpretation block (aberration type, confidence,
frameshift description). Only the MANE Select transcript is reported. Coordinates
and genome build are sourced from the canonical variant dict fields; build is read
from `pipeline/config.py` (`DEFAULT_GENOME_BUILD`). Gate is two-layer: Python gate
skips synonymous, intergenic, UTR, and variants with unresolvable coordinates;
manifest gate additionally skips the API call when a `SpliceAI_v13` column is
already present in the input (pre-computed scores from ANNOVAR/hg38 format). Output
labels are enriched for SLM readability (plain-English aberration names, explicit
delta score scale).

**`websearch_agent`** runs a ReAct loop with three sub-tools (WebSearchTool,
WebFetchTool, NCBIFetchTool). Before starting the loop, a pre-loop checkpoint prompt
receives all pre-fetched evidence from earlier tools (LitVar2, AutoPVS1) and decides
whether any primary gaps remain (OMIM/GeneReviews, ClinVar details, functional data,
recent case reports). The loop runs up to `max_steps=4` iterations; after each
observation a mid-loop checkpoint decides whether to continue or stop. The Python
gate hard-skips ClinVar benign/likely-benign and common variants (AF > 1%) before
reaching the pre-loop checkpoint. Before the ReAct loop starts, `run()` also fires a
deterministic (non-LLM) residue-level web search: `_extract_residue_queries()` pulls
every distinct protein-residue position out of the variant's HGVS and searches
`"{gene} p.Cys1321 ClinVar"` (destination amino acid deliberately dropped) so PS1/PM5
in `conclusion.txt` can see every other pathogenic substitution reported at that
position, not just this variant's own record.

**`gnomad_constraint`** fetches pLI and LOEUF (`oe_lof_upper`) per gene from gnomAD's
public GraphQL API (`https://gnomad.broadinstitute.org/api`, no API key). Gene-scoped
and cached at the class level (`_constraint_cache`), so a cohort with many variants
per gene issues one request per unique gene, not per variant. Feeds PVS1's "is LoF a
plausible mechanism for this gene" gate in `conclusion.txt` (pLI≥0.9 or LOEUF<0.35 =
LoF-intolerant) and the report's Gene-Level Evidence Summary table.

**`clinvar_gene_stats`** fetches two independent pieces of ClinVar evidence via NCBI
E-utilities: (1) gene-level counts of Pathogenic/Likely-pathogenic ClinVar variants
split missense vs. nonsense/frameshift (class-cached per gene, `_stats_cache`) —
grounds PP2/BP1; (2) this specific variant's own ClinVar submission-classification
tally (P/LP/VUS/LB/B per individual submitter), **not** gene-cached — resolved and
fetched fresh per variant, tagging each Pathogenic/Likely-pathogenic submission's
rationale with `[mentions functional/experimental evidence]` when it cites an in
vitro/in vivo/assay finding. This tag is what PS3 in `conclusion.txt` requires before
crediting functional evidence — an aggregate P/LP ClinVar label alone is not
sufficient grounding for PS3. This is the single tool whose NCBI call volume scales
with variant count rather than gene count — the one to watch first under rate-limit
pressure on a large cohort run.

**`clingen_allele`** resolves a variant against the ClinGen Allele Registry
(`reg.clinicalgenome.org/allele`, no API key), cross-referencing ClinVar/dbSNP/gnomAD/
ExAC by canonical allele ID (CAid). Tries transcript+cDNA HGVS, then a clean HGVS
string, then genomic-coordinate SNV fallback (trying both hg19/hg38 accessions since
the declared build is frequently wrong for a given upload). Not gene-cached — one
lookup per variant. An empty cross-reference set is itself reported as a meaningful
"novel/unreported" signal. A 429/5xx from the registry raises `ToolFetchError` (so the
manifest's retry engages); a 400 (genuinely unparseable HGVS/coordinates) is reported
as "not resolvable" without retrying, since retrying a malformed request doesn't help.

**`clinvar_residue_search`** queries ClinVar (NCBI esearch + esummary) for every
classified variant reported at this variant's own protein residue(s), then classifies
each hit against this variant's own change: identical destination amino acid + different
nucleotide → PS1 candidate; different destination amino acid → PM5 candidate; identical
nucleotide → this variant's own record (skipped). One esearch + one batched esummary
call per distinct residue (`{gene}[gene] AND {aa3}{pos}[variant name]`), capped at
`_MAX_HITS_PER_RESIDUE=15`. Replaces reliance on the general-purpose web-search agent
for PS1/PM5 grounding — a live comparison found the web search missed a directly-relevant
ClinVar record that this structured esearch found immediately and deterministically.
`prompts/conclusion.txt`'s PS1/PM5 criteria now treat this tool's output as the
*primary* evidence source, falling back to the web-search agent's residue-level query
only when this block is absent for a variant.

**`gnomad_frequency`** fetches live variant-level gnomAD allele frequency (genome + exome
AF, plus homozygote/heterozygote carrier counts) via the same public GraphQL API as
`gnomad_constraint`. Two call sites: (1) as a retrieval-stage tool, gated to run only
when the input CSV's own `Frequency` field is blank — exists because PM2/BS1 were being
applied from "Frequency is blank, therefore presumed rare" reasoning, when a live
lookup can directly contradict that (a real case: CSV `Frequency=NA`, but gnomAD exome
AF=0.0013 — not rare enough for a presumed-recessive disease); (2) called ungated from
`pipeline.py::_build_gene_evidence_table` for the final report's Gene-Level Evidence
Summary, since homozygote/heterozygote carrier counts are never present in the input
CSV at all, regardless of whether a plain AF was supplied. Not gene-cached — variant-scoped,
fetched fresh per variant/table row.

---

## Mode-of-Inheritance (MOI) & ACMG Scoring

Layered on top of the tool/manifest system above. Two new deterministic (no-LLM)
core modules precompute facts *before* any relevant SLM call, so the model states
them as given rather than re-deriving them under uncertainty:

- **`pipeline/core/segregation.py`** — pure functions over allelic-balance floats:
  `classify_ab_ratio` (0.3–0.7→het, >0.8→hom, <0.1→absent), `classify_segregation`
  (proband+mother+father → de_novo/maternal/paternal/both_carriers/homozygous_parent/
  uncertain/insufficient_data, duo-tolerant), `classify_phase` (two variants' own
  segregation results → cis/trans/unknown — a CIS compound-het pair is filtered out
  here, in plain Python, before it can ever reach an LLM call), `classify_xlinked_ab`.
- **`pipeline/core/moi.py`** — `build_gene_mode_cache(variants, kept_indices,
  context_slices, group_by_gene, llm)` resolves each gene's inheritance mode via a
  3-tier fallback: (1) CSV `Inheritance`/`OMIM_inheritance` text; (2) NHGRI CGD lookup
  via `LitVar2SummaryTool.get_cgd_inheritance()`; (3) an LLM call (`_llm_classify_mode`,
  `prompts/moi_mode_classification.txt`) reasoning over the retrieved literature/
  gnomAD-constraint text for genes tiers 1–2 left unresolved — replacing an older
  plain-text substring match. Genes needing tier 3 are resolved concurrently across a
  dedicated pool (`MAX_WORKERS_MODE_CLASSIFICATION`) after tiers 1–2 run serially over
  all genes. Gated throughout by `gene_chromosome()` (a gene confirmed off chrX can
  never classify as X-linked, regardless of what any tier claims). Returns
  `(gene_mode_cache, gene_mode_reasoning_cache)` — the second dict holds the tier-3
  LLM's justification text (empty for tiers 1–2, which are unambiguous structured
  facts and carry no reasoning), surfaced in the report's Gene-Level Evidence Summary
  so a literature-reasoned mode call is auditable, not a silent label.
  `derive_pertinence()` reads LitVar2's own emitted markers (`"NO DISEASE LINK"` /
  `"PubMed gene-disease evidence for"`) to produce a phenotype-pertinence signal
  without a new tool call.
- **`pipeline/core/acmg_sf.py`** — `ACMG_SF_GENES` (81 genes, ACMG SF v3.2) +
  `build_actionable_set()`: a variant in one of these genes classified P/LP (ClinVar
  field, or an SLM fallback check against already-fetched LitVar2 text when
  ClinVar_class is NA) is force-included through triage/second-triage regardless of
  phenotype relevance, and gets its own report section (`stages/actionable.py`) —
  independent of everything else in the pipeline.

**The SLM produces every criterion's initial score — Python only applies narrow,
documented post-hoc corrections, never an independent scorer.**
`stages/conclusion.py` (Layer 2) scores each variant's MOI-agnostic ACMG criteria
(mechanism, frequency, in-silico, functional, phenotype fit — never PS2/PP1/PM3) via
`prompts/conclusion.txt`'s full Tavtigian et al. 2020 Bayesian point-system rubric.
After the SLM call, `conclusion.py::run_one()` chains five deterministic validators —
`validate_citations → validate_pp3 → validate_pp2_bp1 → validate_ps1_pm5 →
validate_bp6 → validate_pp4` (in `pipeline/core/acmg_{pp3,pp2_bp1,ps1_pm5,bp6,pp4}.py`)
— each a pure string transform over the SLM's *own already-generated* conclusion
text: it regex-matches the criterion line the SLM already wrote, checks it against
separately-fetched deterministic evidence (gene-level ClinVar consequence counts,
in-silico threshold crossings, residue-level ClinVar hits, the justification text
itself), and only on a mismatch strips or inserts that one line, shifting the running
point total by the criterion's fixed value via `acmg_points.py::adjust_points_line()`.
None of these independently evaluate a variant from scratch or compute a criterion the
SLM didn't already attempt — they correct five specific, previously-observed SLM
failure modes (PP2/BP1, PP3, PS1/PM5, BP6, PP4), the same "precomputed fact, never
re-derived" spirit as `core/segregation.py`/`core/moi.py` but applied as a correction
pass over generated text rather than a fact injected into the prompt beforehand.

Four MOI-layer stages then each take that (validator-corrected) base block as **read-only input**
("BASE CONCLUSION — already scored, do not recompute") and add exactly one
inheritance-specific delta on top, gated as follows:

| Layer | Stage | Gate | Adds |
|---|---|---|---|
| De Novo | `moi_denovo.py` | mode ∈ {AD, AD_AR, XLD, XLR, XLD_XLR, XL} | PS2 (trio-confirmed) / PM6 (one parent) |
| Dominant-Inherited | `moi_dominant.py` | mode ∈ {AD, AD_AR, XLD} | PP1 (cosegregation, both proband+parent AB confirm het) |
| Recessive (pair) | `moi_recessive.py::run_pair` | ≥2 variants in a recessive-relevant gene, `classify_phase` ≠ cis | PM3, per-variant, based on partner's own classification |
| Recessive (solo) | `moi_recessive.py::run_solo` | confirmed-homozygous, no trans partner | PM3 at reduced weight (+0/+0.5/+1, ClinGen SVI guidance) |
| X-Linked | `moi_xlinked.py` | `gene_chrom_cache[gene] == "X"` (chromosome, not mode) | PP1 via XLR/XLD allelic-balance pattern |

A variant can land in 0, 1, or 2+ layers (e.g. an AD_AR gene's variant appears in both
Dominant and Recessive) — deliberately not deduplicated. Python only reads the
SLM-written `"ACMG points: N → Classification"` line back out via regex
(`pipeline.py::_extract_acmg_points_value`) for sorting/ranking; it never recomputes it.

`stages/final_conclusion.py` (Layer 8) closes the report with a two-pass synthesis:
a draft (`prompts/clinical_conclusion.txt`) across all MOI layers plus an optional
`actionable_variants` list (ACMG-SF findings, built in `pipeline.py` before this call
so the draft's own prose can name them in a dedicated "Actionable findings" section,
not just the separate ACTIONABLE VARIANTS appendix), then a **second, independent**
call (`prompts/final_conclusion_revise.txt`) that fact-checks the draft against the
same source layer blocks for wrong point totals, hallucinated criteria, CIS pairs
mislabeled compound-het, single-heterozygous-in-AR-gene over-claiming, and disease
names bled across genes. Falls back to the unrevised draft on any failure or
malformed output.

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

### Parallelism

All pipeline stages use a `ThreadPoolExecutor` with `MAX_WORKERS = 32` (set in
both `pipeline/stages/retrieval.py` and `pipeline/pipeline.py`). Variants are
processed concurrently; tools within each variant still run in manifest order.

vLLM's continuous batching handles concurrent HTTP requests natively — parallel
threads translate directly into higher GPU throughput without any special async
plumbing in the pipeline.

Thread-safety requirements:
- `executor.run_variant()` returns `(mini_doc, log_entries)` — no shared state mutated.
- `LitVar2SummaryTool._disease_query` and `_cgd_table` are initialized with
  double-checked locking (`threading.Lock()`).
- `WebSearchAgentTool._last_trace` uses `threading.local()` to avoid cross-thread
  contamination.

GPU KV cache math for the A100 40 GB (Qwen3.5-9B bfloat16):
- Model weights: ~18 GB → KV cache pool: ~18 GB (at `--gpu-memory-utilization 0.90`)
- ~112 KB per token; 32 workers × ~3000 tokens each ≈ 10.7 GB — well within budget.
- Beyond ~48 workers vLLM starts queuing rather than crashing; latency increases.

**This estimate predates the ACMG-scoring/MOI-layer merge and is now optimistic for
the heaviest stage.** `prompts/conclusion.txt` (Layer 2, base ACMG score) alone is
~5,350 fixed instruction tokens before any variant evidence is substituted in —
roughly 5–10x the size of the MOI-layer prompts and a meaningful jump from the old
per-call estimate. Per-call token count is unevenly distributed across stages:
Layer 2 conclusion calls run heavier than the 3,000-token estimate above; the four
MOI-layer calls (Layer 3–6, each their own `MAX_WORKERS_LLM=16` pool) are lighter,
since their fixed template is small and they only re-embed one already-generated
base conclusion block (capped at `MAX_NEW_TOKENS_REPORT=1500`), not raw evidence.
**Recompute this before a large cohort run** rather than trusting the number above —
watch `nvidia-smi` / vLLM's own logs during an initial small-batch test
(see Deployment → Cohort batch run) before pointing `MAX_WORKERS_LLM=16` at the
full cohort. If OOM or heavy queuing shows up, reduce `MAX_WORKERS_LLM` in
`pipeline/pipeline.py` first (the LLM-bound pools) — `MAX_WORKERS=32` (retrieval,
network-bound) is unaffected by this and does not need to change.

To change worker count, edit `MAX_WORKERS`/`MAX_WORKERS_LLM` in `pipeline/pipeline.py`
(and `MAX_WORKERS` in `pipeline/stages/retrieval.py` to match, for retrieval specifically).

---

## Pipeline Stages

The full flow, in the order `pipeline.py::Pipeline.run()` actually executes it:

```
retrieval → ACMG-SF detection → first triage → MOI precompute
  → reasoning (Call 1) → second triage (Call 2) → ACMG-SF override
  → cross-analysis → conclusion (Layer 2, base ACMG)
  → segregation/chromosome cache → MOI layers 3–6 (De Novo, Dominant, Recessive, X-Linked)
  → final conclusion (Layer 8, two-pass) → actionable section → report assembly
```

Threading: `MAX_WORKERS=32` for retrieval/first-triage (network/IO-bound);
`MAX_WORKERS_LLM=16` for every LLM-calling stage from reasoning onward, always
`min(MAX_WORKERS_LLM, len(targets))`. Cross-analysis and final conclusion run
sequentially (once per gene / once per run), not threaded.

### Stage 1 — Retrieval (`stages/retrieval.py`)

For each variant, calls `executor.run_variant()` and collects the returned context slice.
`retrieval.py` is a thin loop — manifest loading, gate evaluation, tool ordering,
execution, and per-tool compression all happen inside the Executor.

```
retrieval.py (per variant)
    → executor.run_variant(variant, ...)
          → gate evaluation (skip if False)
          → tool.run(variant, context)
          → compression (if configured and threshold met)
          → append to mini-doc
    ← context slice string: PATIENT DATA [+ ALLELIC BALANCE if parental_ab given] + VARIANT DATA
```

All context slices are kept independently (one per variant) for downstream stages.

### ACMG-SF actionable detection (`pipeline.py`, calls `core/acmg_sf.py`)

Runs once, on **all** variants, immediately after retrieval — before triage, and
independent of it. `acmg_sf.build_actionable_set()` flags any variant in one of the
81 ACMG SF v3.2 genes classified P/LP. Flagged variants are force-exempted from
DISCARD at first triage and force-set to INCLUDE at second triage, regardless of
phenotype relevance — see the MOI & ACMG Scoring section above.

### First Triage (`stages/first_triage.py`)

One small SLM call per variant immediately after retrieval. Decides whether a variant
can be discarded with high confidence given the patient phenotype.
Prompt loaded from `prompts/first_triage.txt`.

Only runs when `n > TRIAGE_ENABLED_THRESHOLD` (configured in `pipeline/config.py`;
current value: 12). When skipped, all variants are implicitly KEEP.

Two automatic exemptions from DISCARD, both overriding to KEEP: compound heterozygous
candidates (two or more kept-eligible variants in the same gene) and ACMG-SF
actionable findings (checked first).

SLM output format:
```
Keep-case: <strongest reason to keep, ≤10 words>
Discard-case: <strongest reason to discard, ≤10 words>
Decision: KEEP or DISCARD
```

Results:
- **KEEP** variants proceed to MOI precompute and reasoning.
- **DISCARD** variants are excluded from all subsequent stages and appear in the
  appendix of the final report with their justification.

### MOI precompute (`pipeline.py`, calls `core/moi.py`)

Runs once, on all kept variants, before reasoning. Builds `gene_mode_cache` +
`gene_mode_reasoning_cache` (inheritance mode per gene, tiers 1–2 no-LLM /
tier 3 one LLM call per unresolved gene, run concurrently), `recessive_gene_groups`
(compound-het candidate groupings), and `pertinence_cache` (per-variant
gene-phenotype literature signal, read from LitVar2's already-emitted evidence) — see
the MOI & ACMG Scoring section above for the exact 3-tier resolution logic. Also
computes `segregation_cache` (over `kept_indices`) so the deterministic CIS/TRANS
phase fact (`_phase_fact()`) can be injected into sibling-context blocks before
reasoning/second-triage, not just for MOI Layers 3–5 later.

### Stage 2 — Reasoning + Second Triage (`stages/reasoning.py`)

Two SLM calls per kept variant:

**Call 1 — `run_reasoning()`** (`prompts/reasoning.txt`): produces step-by-step clinical
reasoning grounded in cited evidence only, plus a backend-fed `{inheritance_mode_block}`
(gene mode + phenotype pertinence, stated as fact) and, for compound-het candidates,
a `{sibling_context_block}` built from the sibling variant's raw evidence (Call 1 runs
for every kept variant in a gene group before any of them proceeds to Call 2, so the
sibling's own reasoning doesn't exist yet at this point). Ends with a mandatory
"Segregation & Inheritance Summary" table. The model is explicitly instructed not
to produce a score or classification and not to introduce external knowledge without
a reference in the provided context.

**Call 2 — `run_second_triage()`** (`prompts/second_triage.txt`): takes the variant
context and Call 1's reasoning and emits a structured inclusion decision — now with
the sibling's *reasoning* (not just raw evidence) injected, plus AD_AR/XLD_XLR
inheritance-mode carve-outs and a low-allelic-balance-confidence exclusion rule.
Decoupling the decision from the narrative prevents the model from forcing a
favourable outcome to justify speculative reasoning.

SLM output format for second triage:
```
Include-case: <strongest reason to include, ≤15 words>
Exclude-case: <strongest reason to exclude, ≤15 words>
Decision: INCLUDE or EXCLUDE
```

The two outputs are concatenated (separated by `SECOND TRIAGE:`) before being stored,
so the REASONING display section in the final output is self-contained.

Any variant in `actionable_indices` (ACMG-SF) that didn't come back INCLUDE is then
force-set to INCLUDE.

Results:
- **INCLUDE** variants proceed to cross-analysis, conclusion, and the MOI layers.
- **EXCLUDE** variants are listed in the report appendix with their reasoning available
  in the REASONING section — they still get a Layer 2 base conclusion (next stage),
  just no MOI-layer analysis.

### Stage 3 — Cross-analysis (`stages/cross_analysis.py`)

Runs after second triage, for each gene that has two or more **included** variants.
Presents the per-variant contexts and reasonings to the SLM and asks it to evaluate
potential compound heterozygous or other gene-level interactions.
Prompt loaded from `prompts/cross_analysis.txt`.

The cross-analysis output is passed into `conclusion.run_one()` and, where
applicable, `moi_recessive.run_pair()`/`run_solo()` for the relevant variants.

### Stage 4 — Conclusion / base ACMG score, Layer 2 (`stages/conclusion.py`)

One SLM call per **kept, non-reasoning-failed** variant — a broader set than just
INCLUDE, so excluded variants still carry a documented base score for the report
appendix. Takes the variant context slice, its reasoning block (+ backend-fed
pertinence/zygosity blocks), and (if available) the cross-analysis for its gene.
Produces the structured base-ACMG report block. Prompt loaded from
`prompts/conclusion.txt` — see the MOI & ACMG Scoring section above for the full
Tavtigian point-system rubric this now implements.

After the SLM call, `core/citations.py::validate_citations()` strips any inline
citations that are not grounded in the variant context: `(PMID:XXXXXXXX)` references
are kept only if that PMID appears in the context; `(https://...)` references are kept
only if the URL appears verbatim in the context. This prevents hallucinated citations
from propagating into the final report. The same function is reused by all four
MOI-layer stages. Citation validation is followed by the five post-hoc criterion
validators (PP3, PP2/BP1, PS1/PM5, BP6, PP4) described in the MOI & ACMG Scoring
section above.

Output format per variant (abridged — see `prompts/conclusion.txt` for the full
criterion-by-criterion rubric):
```
# Variant [N] — [GENE] ([HGVS])
**Molecular mechanism:** ...
**ClinVar:** ...
**Gene constraint (gnomAD):** ...
**ClinVar gene-level P/LP counts:** ...
**Phenotype fit:** match | mismatch | uncertain — ...
**Inheritance check:** consistent | inconsistent | insufficient data
**Segregation:** de novo | inherited from mother | inherited from father | both parents carriers | uncertain | no parental data
**Evidence strength:** strong | moderate | weak | absent
**ACMG criteria:** [criterion list, each with strength/points/justification]
**ACMG points:** [sum] → [classification]
**Comment:** ...
```

### MOI layers 3–6 (`stages/moi_denovo.py`, `moi_dominant.py`, `moi_recessive.py`, `moi_xlinked.py`)

Run after Layer 2, over **included** variants only, each independently gated (see
the table in the MOI & ACMG Scoring section above). Each is one SLM call (recessive
pair: one call per gene-pair combination) that takes the Layer 2 block as read-only
input and adds exactly one inheritance-specific ACMG delta on top. A variant can hit
0, 1, or 2+ layers — not deduplicated.

### Stage 5 — Final Conclusion, Layer 8 (`stages/final_conclusion.py`)

Two sequential SLM calls, once per run (not per variant):

**Draft** (`prompts/clinical_conclusion.txt`) — takes all non-empty MOI-layer output
lists plus any Unclassified (included-but-no-layer-fired) base conclusions, each under
its own `=== LAYER ===` header, and the patient phenotype.

**Revise** (`prompts/final_conclusion_revise.txt`) — a second, independent call that
fact-checks the draft against the same source layer blocks (wrong point totals,
hallucinated criteria, CIS pairs mislabeled compound-het, single-heterozygous-in-AR-
gene over-claiming, disease names bled across genes). Replaces the draft only if it
succeeds and the output contains `"# Clinical Conclusion"`; falls back to the
unrevised draft otherwise.

Output format (two sections):
```
# Reasoning
[layer-by-layer summary, phase-check lines for recessive genes, then comparison across layers]

# Clinical Conclusion
[per-causative-variant paragraphs with ACMG classification + criteria list, then a closing summary sentence]
```

The combined layer-blocks text is truncated to 50 000 characters before either LLM
call (`_MAX_CONCLUSIONS_CHARS`) to prevent a `400 Bad Request` from vLLM when a
large number of variants are included and the combined text exceeds
`--max-model-len 16384` tokens.

### Actionable Variants section (`stages/actionable.py`)

One SLM call, once per run, batched over all ACMG-SF-flagged variants that reached
a Layer 2 conclusion (skipped entirely if none). Independent of phenotype fit and
MOI layers by design.

### Report assembly (`pipeline.py`, no LLM calls)

`_build_gene_evidence_table(variants, genome_build, gene_mode_cache,
gene_mode_reasoning_cache)` builds the Gene-Level Evidence Summary, one block per
gene: ClinVar P/LP missense:nonsense/FS ratio + gnomAD pLI/LOEUF (read from the
tools' own class-level caches, not re-fetched) + inheritance mode (with reasoning
text when tier-3-LLM-resolved) + a **live** gnomAD variant-level AF/carrier-count
lookup for the gene's representative variant (`gnomad_frequency.py` — never present
in the input CSV regardless of whether a plain AF column existed) → each non-empty
MOI-layer section (blocks sorted Pathogenic → Likely Pathogenic → VUS → Likely Benign
→ Benign) → Unclassified appendix (same sort, each block additionally carrying a
`clinvar_reference.py`-appended ClinVar Variation-ID/URL line) → the Layer 8 Clinical
Conclusion → Actionable Variants (if any) → exclude/discard appendix.

---

## Post-processing — `analyse_results/`

### Purpose

`analyse_results.py` reads the `report.txt` files produced by batch pipeline runs and
computes a per-case analysis CSV (`results_table_{YYYYMMDD}.csv`). It is a standalone
script — it does **not** import from the `pipeline/` package and has no external
dependencies beyond the Python standard library.

**Known gap since the ACMG-scoring/MOI-layer merge:** only received a light
compatibility pass — `_parse_report_funnel`'s `conclusion_genes` regex was updated to
also match MOI-layer headers (`# De Novo Analysis — GENE`, etc.), not just the old
flat `# Variant N — GENE` header, so `n_in_conclusion` doesn't silently collapse to
just the Unclassified section. It does **not** yet parse ACMG points/classification,
per-layer breakdowns, or the Actionable Variants section — that's a separate,
not-yet-done task once the new report format has been seen with real output.

### What it does (per patient)

1. **Final conclusion extraction** (`_extract_final_conclusion`) — locates the
   `# Reasoning` / `# Clinical Conclusion` synthesis block in the FINAL REPORT section.
   Accepts both `#` and `##` header variants (SLM output can vary). Falls back to the
   last `---`-separated block when neither header pattern is found — covers cases where
   the SLM outputs plain text or an unexpected header level.

2. **Variant ranking** — calls the vLLM server to ask the SLM to rank the included
   variants from most to least probable causative (JSON array of `"GENE (HGVS)"` strings).
   `max_tokens=400` is used to avoid truncation on large lists.

3. **Anti-hallucination anchoring** (`_filter_ranked_variants`) — after the SLM ranks the
   variants, the ranked list is anchored to the set of genes confirmed by regex
   (`conclusion_genes` from `_parse_report_funnel`):
   - Step 1: remove ranked identifiers whose leading gene token is not in `conclusion_genes`
     (eliminates SLM hallucinations).
   - Step 2: append any `conclusion_genes` the SLM omitted (ensures completeness, preserving
     report order).
   When the synthesis block is missing entirely, `conclusion_genes` is used directly as the
   ranking fallback (no SLM call needed, no hallucination risk).

4. **Stage funnel parsing** (`_parse_report_funnel`) — pure-regex extraction of stage-level
   statistics: n_input, first triage results, second triage decisions, included variants,
   and genes with conclusion blocks. Runs before the SLM ranking so `conclusion_genes` can
   anchor the ranked list.

5. **Expert gene cross-reference** — looks up the expert solution gene from `pheno_prompts.csv`
   and determines whether it survived each pipeline stage (`_expert_stage_info`) and its
   rank in the SLM-ranked list (`_find_expert_rank`).

### Standalone vLLM client

`_VLLMClient` is a minimal vLLM HTTP client using only `urllib.request` (stdlib). It
replaces the previous import of `pipeline.llm.vllm_client` so the script can run
on the cluster login node or in any environment without the full pipeline package
installed. It sends `"chat_template_kwargs": {"enable_thinking": False}` in every
request, consistent with the pipeline's vLLM client.

### Output

`results_table_{YYYYMMDD}.csv` — one row per case. Key columns:

| Column | Description |
|---|---|
| `patient_id` | Patient stem (directory name with timestamp stripped) |
| `n_variants` | Number of input variants |
| `triage_ran` | Whether first triage ran (n > threshold) |
| `n_discarded_triage` | Variants discarded at first triage |
| `n_excluded_reasoning` | Variants excluded at second triage |
| `n_included` | Variants included in conclusion |
| `ranked_variants` | SLM-ranked list, anchored to conclusion genes |
| `expert_gene` | Solution gene from pheno_prompts.csv |
| `expert_rank` | 1-indexed rank of expert gene in `ranked_variants` (None if absent) |
| `expert_survived_*` | Boolean per stage: did the expert gene pass? |

### Running

```bash
# On the cluster (after sbatch launch.sh has produced results):
bash analyse_results/analyse_results.sh

# Directly:
python analyse_results/analyse_results.py \
    --results-dir /path/to/results \
    --pheno-csv   /path/to/pheno_prompts.csv \
    --vllm-url    http://localhost:8001 \
    --workers     8 \
    --output-dir  analyse_results/
```

The script requires the vLLM server to be running (same server used by the pipeline).

---

## Deployment

### SLURM (production)

`launch.sh` starts two processes in the same job allocation:
1. vLLM server on the GPU (serves the model via OpenAI-compatible HTTP)
2. FastAPI server on CPU (pipeline logic, calls vLLM for SLM steps)

```bash
#!/bin/bash
#SBATCH --job-name=variant_batch
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00

# Start vLLM server
vllm serve Qwen/Qwen3.5-9B \
    --dtype bfloat16 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.90 \
    --port 8001 \
    --enable-prefix-caching \
    > vllm_startup.log 2>&1 &

# Poll until ready (Qwen3.5 Mamba architecture takes several minutes to initialize)
for i in $(seq 1 60); do
    curl -sf http://localhost:8001/v1/models > /dev/null 2>&1 && break
    sleep 5
done

# Run batch pipeline
python batch.py "$JOBS_FOLDER" --backend vllm
```

`--max-model-len 16384` — keeps KV cache within the 40 GB A100 budget.
`--gpu-memory-utilization 0.90` — reserves headroom to avoid OOM during concurrent requests.
`--enable-prefix-caching` — reuses KV cache for shared prompt prefixes across requests.
Startup polling uses 60 attempts × 5 s = 5 minutes; Qwen3.5's hybrid Mamba
architecture needs this much time on first load.

### Cohort batch run (production, current)

For a full patient cohort (e.g. `pheno_prompts.csv`), the actual workflow is
`warmup_vllm.sh` → N × `batch_pheno_vllm.sh`, not `launch.sh`/`batch.py` above:

```bash
# Phase 1.5 — one sequential job, resolves disease queries + NCBI Track 1 for
# every patient into external_prefetch's cache, so Phase 2 jobs below make as
# few live NCBI calls as possible.
sbatch warmup_vllm.sh <jobs_folder>

# Phase 2 — N parallel jobs, one per batch (n-batches must match across all of them)
for i in $(seq 1 6); do
    sbatch batch_pheno_vllm.sh "$i" <jobs_folder> 6
done
```

`batch_pheno_vllm.sh` resolves the HF model ID from `pipeline.llm.registry.MODELS`
so the `vllm serve` command and the pipeline's client always agree on which model
is running; override with `VARIANT_MODEL=gemma4-e4b sbatch batch_pheno_vllm.sh ...`.

**`external_prefetch` is not part of this repo.** `batch_pheno.py` imports it inside
a `try/except ImportError: pass` — if present on the cluster filesystem, it caches
LitVar2's disease-query resolution and NCBI Track-1 results so the parallel Phase 2
jobs mostly avoid live NCBI calls. **It predates the ACMG-scoring merge and does not
cover `clinvar_gene_stats`'s NCBI traffic** (gene-level ClinVar counts + this specific
variant's own submission tally) or the two other new tools (`gnomad_constraint` hits
gnomAD, not NCBI; `clingen_allele` hits the ClinGen Allele Registry, not NCBI) — none
of these are pre-fetched/cached by the existing warmup design. Extending the warmup
to cover `clinvar_gene_stats` the same way it covers LitVar2 Track 1 would meaningfully
reduce live-NCBI pressure during Phase 2, but that's cluster-side code this repo
doesn't have visibility into.

**Recommended before pointing this at the full cohort:** run `warmup_vllm.sh` +
a single `batch_pheno_vllm.sh` batch against a small `--n-batches` slice first (a
handful of patients), and watch for `429`/rate-limit warnings in the job log and
GPU memory headroom in `nvidia-smi`, before fanning out to the full parallel run.

### Local / dev

Set `LLM_BACKEND=hf` to use `hf_client`. No vLLM needed.
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
   `ToolContext`. Two accepted exceptions exist: (a) instance-level caching of
   expensive one-time computations that are identical across all variants (e.g.
   `_disease_query` in `litvar2_summary`, derived solely from `patient_phenotype`) —
   initialised in `__init__` as `None`; (b) class-level caching of data fetched once
   per process and shared across all instances and variants (e.g. `_cgd_table` in
   `litvar2_summary`). Both forms must be pipeline-run-scoped and variant-independent.

2. **Tools never load models.** The LLM client is always accessed via `context.llm`.

3. **Tools never open DB connections.** Always via `context.db`.

4. **Errors are typed.** Always raise from `pipeline/core/errors.py`.
   Never return error strings. Never raise bare `Exception`.

5. **Prompts live in files.** All SLM prompts longer than one sentence belong in
   `prompts/`. Load them with:
   `Path(__file__).parent.parent.parent / "prompts" / "my_prompt.txt"`.
   This resolves relative to the source file, not the working directory.
   Do not hardcode multi-line prompts in Python.

6. **Manifests are the source of truth for ordering and gating.**
   Do not hardcode tool order or conditions in stage code.

7. **`enable_thinking` is always False.** Enforced at the LLM registry level.
   No tool or stage should pass `enable_thinking=True`.

8. **Output is free-form string or None.** Tools do not need to produce structured
   output. The SLM in the reasoning and conclusion stages handles synthesis.

9. **Compression is the executor's job.** Tools return their full output.
   The executor applies compression based on the manifest config.

10. **The server knows nothing about variants.** `server.py` parses the request,
    calls `pipeline.run()`, and returns the result. All logic is in `pipeline/`.

11. **Deterministic facts are computed in Python, never re-derived by the SLM.**
    Anything expressible as a pure function over already-known data (segregation
    pattern, CIS/TRANS phase, gene inheritance mode, X-linked AB pattern — see
    `core/segregation.py`/`core/moi.py`) is computed once in plain Python and handed
    to the prompt as a stated, backend-determined fact ("do not re-derive"). This is
    not an optimization — a CIS compound-het pair, for example, is filtered out in a
    Python loop before it can ever reach an LLM call, rather than relying on the model
    to catch it. Reserve the LLM for judgment calls that genuinely require reading
    unstructured evidence (is this literature relevant, does this comment describe
    functional data, is the phenotype specific enough for PP4) — never for arithmetic
    or classification a `re.match`/threshold check could do reliably. A related but
    distinct pattern: the `core/acmg_{pp3,pp2_bp1,ps1_pm5,bp6,pp4}.py` validators
    (see MOI & ACMG Scoring above) apply this same "Python owns the deterministic
    check" principle *after* the SLM has already generated a criterion line, as a
    narrow correction pass for five specific, previously-observed failure modes —
    never as an original scorer, and never expanded to cover a criterion without a
    documented, recurring SLM mistake behind it.

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
