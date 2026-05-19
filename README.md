![Mars Research Agent](title.svg)

![Mars Research Agent Workflow](workflow.png)

![Section Divider](divider.svg)

## Overview

Mars Research Agent connects literature retrieval, relevance screening, survey generation, and mind map construction into a reusable research pipeline.

| Core Capability | Description |
| --- | --- |
| Literature retrieval | Search and rank Mars-related papers from the local corpus. |
| Status analysis | Extract and organize evidence around a scientific question. |
| Survey generation | Produce structured research summaries from selected papers. |
| Mind map output | Generate a mind map-style representation of the research context. |

![Section Divider](divider.svg)

## Data and Modes

The current corpus contains approximately **2,000 Mars-related papers**, mainly covering Martian mineral research and related geological questions.

| Mode | Best For | PDF Required |
| --- | --- | --- |
| `lightweight` | Fast exploration and initial screening | No |
| `middle` | Query-centered evidence extraction | Yes |
| `full` | Deeper survey preparation and analysis | Yes |

![Section Divider](divider.svg)

## Quick Start

Run an interactive retrieval demo:

```bash
python scripts/search_demo.py
```

Run the survey pipeline:

```bash
python scripts/run_survey_stage.py \
  --query "What is the evidence for aqueous alteration in Noachian terrains on Mars?" \
  --top_m 30 \
  --batch_size 5 \
  --survey_mode full
```

For a faster lightweight run:

```bash
python scripts/run_survey_stage.py \
  --query "Mars clay minerals in Gale crater" \
  --top_m 20 \
  --survey_mode lightweight
```

![Section Divider](divider.svg)

## Main Outputs

Outputs are saved under `outputs/` with a timestamped run directory.

| Output | Path |
| --- | --- |
| Retrieved papers | `search_stage/top_m_papers.json` |
| Selected papers | `survey_stage/selected_papers.json` |
| Survey draft | `survey_stage/survey.txt` |
| Mind map | `survey_stage/MindMap.txt` |

![Section Divider](divider.svg)

## Project Layout

```text
config.py                # Configuration
scripts/                 # Entry scripts
src/lit_agent/agents/    # Agent workflow
src/lit_agent/retrieval/ # Retrieval modules
src/lit_agent/prompts/   # Prompt templates
outputs/                 # Generated results
```

![Section Divider](divider.svg)

## Future Direction

The project will continue toward a full research assistant for Martian geology, connecting literature analysis with multimodal scientific data, mineral detection agents, and discovery-oriented workflows.
