SUPPLEMENT_DECISION_SYSTEM_PROMPT = """
You are a scientific literature coverage auditor.
Decide whether the current paper records need one small supplementary evidence round.
Return only valid JSON.
""".strip()


SUPPLEMENT_DECISION_USER_PROMPT = """
Scientific question:
{query}

Survey mode:
{survey_mode}

Current core paper records:
{paper_records_json}

Task:
Inspect the current core paper records and decide whether a small supplementary
evidence round is useful.

Important:
- The supplement round is optional and should be used only when it may improve
  coverage of the scientific question.
- Do not request supplement just because more papers could exist.
- Focus on concrete gaps visible from the current records.
- The supplementary round can use both local retrieval expansion and web
  metadata search.
- Web search is metadata/abstract-level only.

Useful gap types:
- missing_recent_work
- weak_evidence_theme
- missing_alternative_interpretation
- missing_method_or_dataset
- sparse_local_coverage
- no_clear_gap

For each gap, generate:
- topic: a concise gap topic
- gap_type
- why_gap: why the current records do not cover it well
- local_query: a short local retrieval query, based on current records and the scientific question
- web_queries: up to {max_web_queries_per_gap} concise ADS/Crossref-friendly queries

Return this exact JSON object:
{{
  "should_supplement": true,
  "reason": "",
  "gaps": [
    {{
      "topic": "",
      "gap_type": "",
      "why_gap": "",
      "local_query": "",
      "web_queries": []
    }}
  ]
}}

Rules:
- If no useful gap is visible, set should_supplement to false and gaps to [].
- Return at most {max_gaps} gaps.
- Return only valid JSON.
- No Markdown.
""".strip()


SUPPLEMENT_SYNTHESIS_SYSTEM_PROMPT = """
You are a scientific literature supplement synthesizer.
Synthesize accepted supplementary candidates into traceable, abstract-level
evidence cards for cautious integration with a core literature survey.
Return only valid JSON.
""".strip()


SUPPLEMENT_SYNTHESIS_USER_PROMPT = """
Scientific question:
{query}

Survey mode:
{survey_mode}

Supplement plan:
{supplement_plan_json}

Current core paper records:
{paper_records_json}

Supplementary local candidates:
{local_candidates_json}

Supplementary web candidates:
{web_candidates_json}

Task:
Determine how the accepted supplementary candidates affect the survey coverage
and create concise, citable evidence cards.

Important:
- Core paper records are the main evidence base.
- Supplementary candidates were screened after reading the provided abstract,
  corpus summary, or external abstract identified by source_read.
- Treat all supplementary findings as abstract-level or summary-level evidence,
  never as full-text analysis.
- A paper may support, qualify, challenge, or contextualize a core survey claim.
- Each accepted evidence card must identify its citation and reading source.
- If the provided content does not justify a usable finding, do not accept it
  as evidence; place it in follow_up_candidates if it may still merit reading.
- Do not force a change if the supplementary candidates are weak.

Return this exact JSON object:
{{
  "supplement_used": true,
  "overall_effect": "none | minor_gap_note | add_context | add_recent_direction | adjust_outline",
  "evidence_level": "abstract_or_summary_level_supplement",
  "summary": "",
  "accepted_supplementary_papers": [
    {{
      "citation_id": "S1",
      "citation": "",
      "title": "",
      "year": "",
      "journal": "",
      "source_type": "local | web",
      "source_read": "abstract_note | research_content | abstract",
      "reading_level": "abstract_level | corpus_summary_level",
      "matched_gap_topic": "",
      "why_selected": "",
      "distilled_finding": "",
      "relation_to_main_survey": "supports | qualifies | challenges | adds_context | identifies_gap",
      "confidence": "low | medium | high",
      "limitations": ""
    }}
  ],
  "integration_analysis": [
    {{
      "main_claim": "",
      "effect": "strengthened | qualified | challenged | contextualized | new_gap",
      "reason": "",
      "citations": ["S1"]
    }}
  ],
  "follow_up_candidates": [
    {{
      "title": "",
      "year": "",
      "source": "",
      "doi": "",
      "why_follow_up": ""
    }}
  ]
}}

Rules:
- If nothing can be supported from the provided content, set supplement_used
  to false, overall_effect to "none", and accepted_supplementary_papers to [].
- Assign citation_id values sequentially as S1, S2, and so on.
- Build citation from provided authors and year; do not invent authors.
- Keep distilled_finding faithful to the supplied content and concise.
- State limitations explicitly, especially geographic scope or reading level.
- Return only valid JSON.
- No Markdown.
""".strip()


SUPPLEMENT_RELEVANCE_SYSTEM_PROMPT = """
You are a scientific literature relevance screener for supplementary candidates.
Judge whether each candidate is useful for the specific gap it was retrieved for.
Return only valid JSON.
""".strip()


SUPPLEMENT_RELEVANCE_USER_PROMPT = """
Scientific question:
{query}

Survey mode:
{survey_mode}

Supplement plan:
{supplement_plan_json}

Supplementary candidates:
{candidates_json}

Task:
Screen each supplementary candidate for usefulness.

Important:
- These candidates are not part of the main paper records.
- They should pass screening only if they help cover one of the supplement gaps.
- Be stricter than ordinary keyword matching.
- Examine the supplied content before deciding usefulness.
- Candidates with source_read = "metadata_only" cannot be accepted as
  evidence and may only be recommended for later reading.
- Web candidates are external abstract-level only, so mark confidence cautiously.

Relevance levels:
- "high": directly helps cover a named supplement gap.
- "medium": useful context for a named supplement gap.
- "low": weak keyword overlap, not useful enough.
- "irrelevant": unrelated or misleading.

For each candidate, return only:
- row_id
- relevance_level
- is_usable
- matched_gap_topic
- reason

Rules:
- is_usable = true only for "high" or "medium".
- is_usable = false for "low" or "irrelevant".
- Return one result for every input row_id.
- Return only a valid JSON list.
- No Markdown.
""".strip()


SUPPLEMENT_SURVEY_INTEGRATION_PROMPT = """

Supplementary abstract- or summary-level evidence:
{supplement_report_json}

Instructions for integrating supplementary evidence:
- The literature records above remain the primary evidence base.
- The accepted supplementary papers have been inspected only at the reading
  level recorded in each evidence card; do not describe them as full-text reads.
- Use accepted evidence cards when they materially support, qualify, challenge,
  or contextualize a survey claim, and cite them using their provided citation.
- Do not mix these papers into the main chronological literature contributions.
- Add a concise section titled "Supplementary Abstract-Level Evidence and Its
  Implications" only if accepted_supplementary_papers is non-empty.
- In that section, explain how the supplementary evidence changes or constrains
  the interpretation, including uncertainty and limitations.
- Add a short "Supplementary References Consulted at Abstract or Summary Level"
  list containing the citation, title, and reading level of accepted papers.
- Follow-up candidates without usable abstract-level evidence may be mentioned
  as future reading needs, not used to support scientific claims.
""".strip()


SUPPLEMENT_MINDMAP_INTEGRATION_PROMPT = """

Supplementary abstract- or summary-level evidence:
{supplement_report_json}

Instructions for supplementary mind-map nodes:
- Keep the main literature-record nodes separate from supplementary evidence.
- Add a branch titled "Supplementary Abstract-Level Evidence" only if accepted
  supplementary papers are present.
- For each supplementary node, show its citation_id and citation, distilled
  finding, effect on the central interpretation, reading level, and limitation.
- Use integration_analysis to connect supplementary evidence to strengthened,
  qualified, challenged, or unresolved claims.
- Do not present supplementary papers as fully read literature records.
""".strip()
