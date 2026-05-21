# Variant Analysis Pipeline

Clinical genomics pipeline that takes a list of genetic variants and a patient
phenotype description, gathers evidence from multiple sources per variant, and
produces a structured clinical report using a Small Language Model (SLM).

Designed for SLURM HPC clusters (CURNAGL, A100 40 GB GPU node). Exposed as a
FastAPI server — accessible from a browser, programmatically via HTTP, or as an
sbatch job.

---

## Architecture overview

```
CSV / Excel  ──►  normalize_upload()
                        │
                        ▼
             Pipeline.run(variants, phenotype)
                        │
        ┌─────────────────────────────────────────┐
        │  Stage 1 — Retrieval (all variants)      │
        │                                          │
        │  order 1 parallel: LitVar2  SpliceAI     │
        │  order 2 parallel: AutoPVS1              │
        │  order 3 serial:   WebSearchAgentTool    │
        └──────────────┬──────────────────────────┘
                       │  one context slice per variant
                       ▼
              Triage  (if n > threshold)
              KEEP / DISCARD per variant
              (compound-het candidates always KEEP)
                       │  kept variants only
                       ▼
        ┌─────────────────────────────────────────┐
        │  Stage 2 — Reasoning                     │
        │  one SLM call per kept variant           │
        │  → step-by-step reasoning                │
        │  → RELEVANCE score 1–3                   │
        └──────────────┬──────────────────────────┘
                       │
              Tier selection
              Tier 1: score-3 first, backfill score-2
              Tier 2: kept but not Tier 1
                       │  Tier 1 only below
                       ▼
        ┌─────────────────────────────────────────┐
        │  Stage 3 — Cross-analysis                │
        │  (genes with ≥ 2 Tier 1 variants)        │
        │  compound-het / interaction assessment   │
        └──────────────┬──────────────────────────┘
                       │
                       ▼
        ┌─────────────────────────────────────────┐
        │  Stage 4 — Per-variant Conclusion        │
        │  one SLM call per Tier 1 variant         │
        └──────────────┬──────────────────────────┘
                       │
                       ▼
              Stage 5 — Final Conclusion
              one SLM call across all Tier 1 blocks
```

---

## Running on the cluster (SLURM)

```bash
sbatch launch.sh
```

`launch.sh` starts two processes in the same job allocation:
1. **vLLM server** on the GPU (serves Qwen/Qwen3.5-9B via OpenAI-compatible HTTP)
2. **FastAPI server** on CPU (pipeline logic, calls vLLM for SLM steps)

---

## Submitting a job from the login node

```bash
python client.py \
    --server http://<node>:8000 \
    --csv variants.csv \
    --phenotype "Obesity, intellectual disability, hyperphagia" \
    --output report.txt
```

The `--server` URL is printed in the SLURM job output when the FastAPI server starts.

---

## Local / dev mode (no GPU, no vLLM)

Set `LLM_BACKEND=hf` to use HuggingFace `transformers.pipeline` directly.
Slower, but identical pipeline behaviour.

```bash
LLM_BACKEND=hf uvicorn server:app --port 8000
```

---

## Input format

Any CSV or Excel file. Unknown columns are ignored. Missing columns are filled
with `"NA"`. Common column names are auto-detected via `COLUMN_ALIASES` in
`pipeline/core/normalizer.py`.

| Field | Example |
|---|---|
| `Variant` | `chr6:100896130 T>C` |
| `Chromosome` | `chr6` |
| `Position` | `100896130` |
| `RS_ID` | `rs28936388` |
| `Ref_seq` | `T` |
| `Var_seq` | `C` |
| `Type` | `SNV` |
| `HGVS` | `NM_005068.3:c.744-2A>G p.?` |
| `Zygosity` | `Heterozygous` |
| `Gene` | `SIM1` |
| `ClinVar_class` | `Pathogenic` |
| `Frequency` | `1.19E-05` |
| `CADD_score` | `26` |

---

## Output format

Plain text with four sections separated by `=`×60 banners:

```
============================================================
PROCESS DETAILS
============================================================

────────────────────────────────────────────────────────
VARIANT 1 — GENE  NM_xxx.x:c.xxx
────────────────────────────────────────────────────────

[LITVAR2_SUMMARY]  GATE: PASS
  Query     : <disease query>
  Articles  : N kept from M found
  Kept articles:
    [PMID:12345678] Title of article

[SPLICEAI]  GATE: PASS
  Raw output:
    SpliceAI scores ...

[AUTOPVS1]  GATE: SKIPPED  (variant type not LoF)

[WEBSEARCH_AGENT]  GATE: PASS
  ReAct trace:
    Step 1: ...
  Raw output:
    ...

============================================================
AUGMENTED CONTEXT
============================================================
PATIENT DATA:
<patient phenotype>

VARIANT DATA:

VARIANT 1:
Variant=..., Chromosome=..., Gene=..., ...

<tool outputs for variant 1>

VARIANT 2:
...

============================================================
REASONING
============================================================
<step-by-step clinical reasoning per variant>

============================================================
FINAL REPORT
============================================================
# Variant 1 — GENE (HGVS)
**Molecular mechanism:** ...
**Phenotype fit:** match / mismatch / uncertain — ...
**Inheritance check:** consistent / inconsistent / insufficient data
**Evidence strength:** strong / moderate / weak / absent
**ACMG criteria**: ...
**Comment:** ...

---

# Clinical Conclusion
...
```

**PROCESS DETAILS** logs each tool's gate outcome (`PASS` or `SKIPPED` with reason)
and its metadata: LitVar2 keeps article titles and PMIDs; WebSearchAgent records the
full ReAct trace; all tools append their raw string output.

**AUGMENTED CONTEXT** is the raw evidence document assembled by Stage 1 (retrieval).
It covers all variants and is passed as-is to the SLM reasoning and conclusion stages.

**REASONING** is the per-variant step-by-step output of Stage 2. Covers all kept
(non-discarded) variants; each block is tagged `[TIER 1]` or `[TIER 2]`. Not shown
in the clinical report — used only as input to Stage 4.

**FINAL REPORT** contains Tier 1 structured blocks (Stage 4), the overall
`# Clinical Conclusion` paragraph (Stage 5), and — when triage ran — an appendix
listing Tier 2 variants (reasoning available) and Tier 3 variants (discarded at triage).

---

## Adding a new tool

See `TOOL_DEVELOPMENT.md` for the full guide. Minimum steps:

1. Create `pipeline/tools/my_tool.py` inheriting from the appropriate base class
2. Implement `gate()` (optional) and `run()`
3. Create `pipeline/manifests/my_tool.yaml` with `name`, `order`, `enabled`
4. Register in `pipeline/tools/__init__.py`

No other files need to change.

---

## Dependencies

### Core runtime
```
torch>=2.0
transformers>=4.40
fastapi
uvicorn
pydantic
httpx
vllm
```

### Pipeline tools
```
requests
beautifulsoup4
ddgs
pandas
openpyxl
yaml
```
