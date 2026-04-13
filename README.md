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
            ┌───────────┼───────────┐
            │           │           │
        LitVar2      AutoPVS1   WebSearch
        (order 1)   (order 2)   Agent (3)
            │           │           │
            └───────────┴───────────┘
                        │ mini-docs
                        ▼
              [Stage 2 — optional compression]
                        │
                        ▼
              Stage 3 — step-by-step reasoning
                        │
                        ▼
              Stage 4 — structured report
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

Plain text with three sections:

```
============================================================
AUGMENTED CONTEXT
============================================================
PATIENT DATA:
...

VARIANT DATA:
VARIANT 1:
...

============================================================
REASONING
============================================================
...

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

# Clinical Conclusion
...
```

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
