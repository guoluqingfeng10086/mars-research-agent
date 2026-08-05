# Mars Research Agent

Mars Research Agent is a research workflow for Mars-related literature
retrieval, relevance screening, survey generation, research-map construction,
and evidence-gap analysis. The complete system supports adaptive full-text
reading and optional integration of local geologic products.
multi-source data products can be found in the appendix of the article.

## Review-Package Scope

The complete paper corpus is not included in this review package. The released
serialized corpus contains approximately **2,235 paper records**, including
bibliographic metadata, `abstract_note`, `research_content`, and precomputed
embeddings. It supports lightweight literature retrieval, Survey generation,
Research Memory construction, and Discovery analysis.

The reviewer-facing lightweight version does not open the source PDF/Markdown
papers or inspect their full references and supplementary files. Compared with
the complete version described in the manuscript, it therefore does not perform
adaptive full-text reading. Survey and Discovery results produced in this mode
are provided for reference only.

## Reviewer Quick Start

The reviewer-facing lightweight workflow requires Python 3.10. Install its
dependencies from the project root:

```bash
python -m pip install -r requirements.txt
```

Before running the scripts, configure the LLM endpoint and the local
`bge-large-en-v1.5` model path in `config.py`:

```python
MODEL_PATH = r"<path-to-bge-large-en-v1.5>"

QUERY_LLM_API_KEY = "<your-api-key>"
QUERY_LLM_BASE_URL = "<your-OpenAI-compatible-endpoint>"
QUERY_LLM_MODEL = "<model-name>"
```

Set `DEVICE = "cpu"` in `config.py` when CUDA is unavailable. GPU users should
install the PyTorch build appropriate for their CUDA environment and retain
`DEVICE = "cuda"`.

Do not commit a real API key to the repository. The released
`mmcorpus_bge_merged.pkl` is used by default through `CORPUS_PKL_PATH`.
`PDF_CORPUS_DIR` and `MD_CORPUS_DIR` are not required for the reviewer-facing
`lightweight` mode.

Also disable the optional supplement round in `config.py`:

```python
ENABLE_SUPPLEMENT_ROUND = False
```

### 1. Run the Lightweight Survey

The Survey stage can be run independently to retrieve papers, screen relevance,
and generate the lightweight Research Memory:

```bash
python scripts/run_survey_stage.py \
  --query "Does the evidence support an ancient ocean in Chryse Planitia?" \
  --top_m 20 \
  --batch_size 5 \
  --survey_mode lightweight
```

This produces `survey.txt`, `MindMap.txt`, and `research_map.json` without
opening source PDF/Markdown papers.

### 2. Run the Discovery Agent

After the Survey has been generated, run the public Discovery entrypoint with
the same or a closely related question:

```bash
python scripts/run_discovery_agent.py \
  --question "Does the evidence support an ancient ocean in Chryse Planitia?" \
  --evidence-sources survey \
  --no-gap-filling \
  --no-new-survey \
  --prompt-mode hypothesis_evaluation
```

This configuration uses only the released lightweight records and the Research
Memory generated from them. It performs no full-text reading and no additional
local or Web evidence acquisition.

### Example Scientific Questions

The question in the commands above can be replaced with related Mars-science
questions, for example:

- Did an ancient ocean exist in Chryse Planitia?
- What evidence supports aqueous alteration in Noachian terrains on Mars?
- How did valley networks in the Martian highlands form?
- Is the reported evidence for subsurface water ice credible?
- At 34.84 N, 102.57 E in Chryse Planitia, does the evidence support an ancient ocean?

Use `hypothesis_evaluation` for hypothesis questions,
`detection_confidence` for reported detections, and `generic` for broader
evidence-synthesis questions.

### Mineral-Detection Confidence Example

In the complete local research environment, a location-specific mineral
confidence analysis can combine Survey memory, literature expansion, and local
geologic products:

```bash
python scripts/run_discovery_agent.py \
  --question "At latitude 34.84 N, longitude 102.57 E in Utopia Planitia, a possible Fe/Mg-smectite signature was reported. How credible is this detection?" \
  --prompt-mode detection_confidence \
  --evidence-sources survey,local,web,geo \
  --max-discovery-rounds 2
```

The coordinates identify the location for geologic-product retrieval. The
reported mineral signature is treated as a hypothesis to assess, not as an
accepted observation. In the reviewer-facing lightweight configuration, the
same question can be evaluated only from the released Research Memory, so its
result remains for reference only.

## Workflow

```text
Scientific question
  -> query analysis
  -> vector + BM25 retrieval
  -> RRF ranking
  -> relevance screening
  -> paper records
  -> Survey report
  -> text mind map + structured Research Memory
  -> completeness assessment
  -> Discovery report
```

The Discovery Agent reads `survey.txt`, `MindMap.txt`, and `research_map.json` to
trace claims, evidence, conflicts, alternatives, and evidence gaps. In the
complete local environment it can request additional literature reading and
geologic evidence. In the reviewer configuration above, it operates only on the
lightweight Research Memory.

## Reading Modes

| Mode | Information used | Full-text corpus | Review package |
| --- | --- | --- | --- |
| `lightweight` | Metadata, `abstract_note`, `research_content` | Not required | Supported |
| `middle` | Query-centered PDF/Markdown reading | Required | Not reproducible |
| `full` | Adaptive multi-round full-text analysis | Required | Not reproducible |

The `middle` and `full` implementations are included in the code and are used
with the complete corpus in the authors' local environment.

## Project Structure

```text
config.py                                  model, corpus, and API configuration
geo_data_config.py                         optional geologic-product paths
scripts/run_discovery_agent.py             public entrypoint
scripts/run_survey_with_supplement.py      internal Survey workflow
src/lit_agent/agents/discovery_agent.py    Discovery orchestration
src/lit_agent/agents/retrieval_agent.py    local literature retrieval
src/lit_agent/agents/relevance_agent.py    relevance screening
src/lit_agent/agents/paper_digest_agent.py paper reading and record generation
src/lit_agent/agents/survey_agent.py       Survey generation
src/lit_agent/agents/mindmap_agent.py      text mind-map generation
src/lit_agent/agents/research_map_agent.py structured Research Memory
src/lit_agent/agents/discovery_gap_agent.py completeness and gap assessment
src/lit_agent/agents/discovery_evidence_agent.py optional evidence acquisition
src/lit_agent/retrieval/                    vector, BM25, Web, and geo retrieval
src/lit_agent/prompts/discovery_prompts.py complete Discovery prompt set
```

## Configuration

The main local paths are defined in `config.py`:

```text
CORPUS_PKL_PATH   serialized retrieval corpus
MODEL_PATH        BGE embedding model
PDF_CORPUS_DIR    source PDF directory
MD_CORPUS_DIR     converted full-text Markdown directory
OUTPUT_ROOT_DIR   Survey output root
```

Optional geologic-product paths are defined in `geo_data_config.py`.

The core `requirements.txt` covers the reviewer-facing lightweight workflow.
The complete local system additionally requires `PyMuPDF` for PDF reading. Its
optional geologic-product retrieval requires `pandas`, `geopandas`, `rasterio`,
`shapely`, `fiona`, `geopy`, `requests`, and `lxml`, together with the local
datasets configured in `geo_data_config.py`.

## Outputs

Survey outputs:

```text
<OUTPUT_ROOT_DIR>/<survey_run>/
  run_config.json
  search_stage/top_m_papers.json
  survey_stage/selected_papers.json
  survey_stage/paper_records_*.json
  survey_stage/survey.txt
  survey_stage/MindMap.txt
  survey_stage/research_map.json
```

Discovery outputs:

```text
discovery_stage/<question_timestamp>/
  initial_research_map.json
  final_research_map.json
  final_completeness.json
  discovery_rounds.json
  discovery_result.json
  discovery_report.txt
```

## Data Availability

The complete literature corpus used by the full system will be released after the paper is accepted.
You can also directly obtain paper sources from the .pkl file to build a local corpus.
