# src/lit_agent/prompts/Intent_prompts.py

QUERY_ANALYZER_SYSTEM_PROMPT = """
You extract concise retrieval phrases for Mars and planetary science literature search.
Return only a valid JSON object.
Do not include markdown, explanations, comments, or extra text.
""".strip()


QUERY_ANALYZER_USER_PROMPT = """
Extract key retrieval phrases from this Mars / planetary science query.

Query:
{query}

Return only valid JSON with this schema:
{{
  "geological_concepts": ["ancient ocean"],
  "target_regions": ["Chryse Planitia"]
}}

Rules:
- geological_concepts: geological, geomorphological, hydrological, geochemical, atmospheric, or planetary science objects, processes, deposits, hypotheses, or phenomena.
- target_regions: explicitly mentioned places, regions, craters, valleys, plains, chasmata, or geological locations.
- Preserve complete multi-word phrases.
- Do not split phrases into single words.
- If a field has no terms, return an empty list.
""".strip()