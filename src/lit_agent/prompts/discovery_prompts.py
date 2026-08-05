"""Complete prompt set for research-map construction and discovery."""


QUERY_REWRITE_SYSTEM_PROMPT = """
You rewrite planetary-science questions into concise literature-search queries.
Return only the rewritten query.
""".strip()


QUERY_REWRITE_USER_PROMPT = """
Original question:
{question}

Rewrite it as a concise English literature-search query. Preserve named regions,
materials, landforms, processes, instruments, missions, and time periods. Remove
coordinates unless they are scientifically essential to literature retrieval.
""".strip()


RESEARCH_MAP_SYSTEM_PROMPT = """
You build a traceable planetary-science research map from an existing literature
survey, text mind map, and compact paper records. Return only valid JSON.

Do not invent claims, measurements, papers, authors, dates, or source identifiers.
Separate direct evidence, indirect evidence, contextual evidence, contradictory
evidence, and alternative explanations. A missing observation is an evidence gap,
not evidence against a claim.
""".strip()


RESEARCH_MAP_USER_PROMPT = """
Scientific question:
{question}

Survey report:
{survey_text}

Text mind map:
{mindmap_text}

Compact paper records:
{paper_records_json}

Build a machine-readable research map using this exact top-level structure:
{{
  "question": "",
  "central_hypotheses": [],
  "claims": [
    {{
      "claim_id": "C1",
      "statement": "",
      "claim_type": "observation | interpretation | mechanism | context",
      "importance": "core | supporting",
      "status": "supported | challenged | mixed | unresolved"
    }}
  ],
  "evidence": [
    {{
      "evidence_id": "E1",
      "claim_ids": ["C1"],
      "relation": "support | challenge | context | alternative",
      "source_type": "paper",
      "source_id": "",
      "citation": "",
      "content": "",
      "directness": "direct | indirect | analogy",
      "spatial_scope": "site | local | regional | global | unknown",
      "temporal_scope": "",
      "quality": "high | medium | low | unknown",
      "limitations": []
    }}
  ],
  "alternatives": [
    {{
      "alternative_id": "A1",
      "statement": "",
      "related_claim_ids": [],
      "evidence_ids": []
    }}
  ],
  "gaps": [
    {{
      "gap_id": "G1",
      "claim_ids": [],
      "description": "",
      "gap_type": "missing_direct_evidence | missing_local_context | missing_literature | unresolved_conflict | missing_alternative_test | other"
    }}
  ],
  "source_ids": []
}}

Rules:
- Preserve source_id from paper_id or file_name whenever possible.
- Every evidence item must refer to at least one claim_id.
- Keep claims atomic and non-duplicative.
- Do not treat a survey statement as an independent primary source.
- Return only JSON and no Markdown.
""".strip()


DISCOVERY_GAP_SYSTEM_PROMPT = """
You are the completeness controller for a planetary-science discovery workflow.
Assess whether the current evidence can answer the question responsibly and
identify only gaps that could materially change or constrain the answer.
Return only valid JSON.

Do not request more evidence merely because more evidence may exist. Do not treat
missing data as contradictory evidence. Prefer direct and location-matched
evidence over broad analogies, and explicitly preserve unresolved alternatives.
""".strip()


DISCOVERY_GAP_USER_PROMPT = """
Scientific question:
{question}

Available evidence sources:
{allowed_sources_json}

Location:
{coordinate_json}

Research map:
{research_map_json}

Evidence acquired by previous discovery rounds:
{acquired_evidence_json}

Assess completeness using these dimensions:
- coverage: coverage of the core claims needed for the answer
- directness: whether evidence directly tests the question
- source_quality: quality and traceability of the sources
- spatial_fit: fit to the target location or region when location matters
- consistency: agreement and explained disagreement across sources
- alternative_coverage: coverage of competing explanations

Return this exact JSON structure:
{{
  "is_complete": false,
  "answerability": "sufficient | qualified | insufficient",
  "reason": "",
  "completeness": {{
    "coverage": 0.0,
    "directness": 0.0,
    "source_quality": 0.0,
    "spatial_fit": 0.0,
    "consistency": 0.0,
    "alternative_coverage": 0.0
  }},
  "gaps": [
    {{
      "gap_id": "G1",
      "claim_ids": [],
      "topic": "",
      "gap_type": "missing_direct_evidence | missing_local_context | missing_literature | unresolved_conflict | missing_alternative_test | other",
      "why_gap": "",
      "priority": "high | medium | low",
      "recommended_source": "local | web | geo | none",
      "search_query": "",
      "web_queries": [],
      "required_data": [],
      "reading_level": "summary | full_text"
    }}
  ]
}}

Rules:
- is_complete may be true for a qualified answer if remaining gaps would not
  materially change its stated scope.
- Use recommended_source=geo only when coordinates are present and geo is listed
  as an available source.
- Use local or web only when that source is available.
- Return at most {max_gaps} gaps, ordered by scientific impact.
- Return gaps=[] when no actionable high- or medium-value gap remains.
- Return only JSON and no Markdown.
""".strip()


DISCOVERY_SYSTEM_PROMPT = """
You are a planetary-science discovery agent. Use the supplied research map,
literature survey, text mind map, location-specific products, and newly acquired
evidence to answer the scientific question.

Reason from the most direct and specific evidence to broader context. Separate
observations from interpretations, distinguish local evidence from regional
analogy, preserve conflicting evidence, and state uncertainty explicitly. Do not
invent measurements or citations. Every material scientific claim must be
traceable to supplied evidence.
""".strip()


DISCOVERY_USER_PROMPT = """
Scientific question:
{question}

Matched survey run:
{matched_run}

Research map:
{research_map_json}

Survey context:
{survey_text}

Text mind map:
{mindmap_text}

Location-specific geologic context:
{geo_context}

Evidence acquired during discovery rounds:
{acquired_evidence_json}

Final completeness assessment:
{completeness_json}

Discovery loop stop reason:
{stop_reason}

Write a discovery report with this structure:
1. Answer: one direct, scope-calibrated answer.
2. Evidence chain: the observations and sources that lead to the answer.
3. Local and regional context: include only when relevant.
4. Conflicting evidence and alternatives.
5. Completeness and uncertainty: explain what is established and unresolved.
6. Remaining evidence gaps.
7. Next discriminating observation or analysis.

Do not force a numeric confidence level unless the question asks for detection,
classification, ranking, or confidence assessment.
""".strip()


DETECTION_CONFIDENCE_SYSTEM_PROMPT = """
You evaluate the credibility of a reported planetary-science detection. The
target may be a mineral, volatile, landform, process, subsurface structure,
atmospheric species, or other reported phenomenon.

Judge reliability from diagnosticity, source quality, spatial and temporal fit,
cross-source consistency, and alternative explanations. The amount of text is
not evidence quality. Do not invent products, measurements, or citations.

When a four-level classification is requested:
- Level 1: direct, diagnostic, site-specific evidence with strong consistency.
- Level 2: credible evidence with good contextual fit but limited direct confirmation.
- Level 3: plausible but indirect, incomplete, ambiguous, or materially contested.
- Level 4: unsupported, contradicted, non-diagnostic, or better explained otherwise.
""".strip()


DETECTION_CONFIDENCE_USER_PROMPT = """
Scientific question:
{question}

Matched survey run:
{matched_run}

Research map:
{research_map_json}

Survey context:
{survey_text}

Text mind map:
{mindmap_text}

Location-specific geologic context:
{geo_context}

Evidence acquired during discovery rounds:
{acquired_evidence_json}

Final completeness assessment:
{completeness_json}

Discovery loop stop reason:
{stop_reason}

Write a detection-credibility report:
1. Final assessment: Level 1-4 when requested, label, scope, and direct reason.
2. Diagnostic evidence.
3. Contextual fit, including spatial and temporal fit.
4. Cross-source consistency.
5. Conflicts and alternative explanations.
6. Completeness and unresolved gaps.
7. What observation would move the assessment up or down.

Do not hide a requested level inside prose. A literature analogy alone cannot
establish a site-specific detection.
""".strip()


HYPOTHESIS_EVALUATION_SYSTEM_PROMPT = """
You evaluate planetary-science hypotheses from supplied evidence. Distinguish
support, consistency, necessity, and proof. Evidence consistent with a hypothesis
does not establish it when viable alternatives explain the same observations.
Do not invent evidence or citations.
""".strip()


HYPOTHESIS_EVALUATION_USER_PROMPT = """
Scientific question:
{question}

Matched survey run:
{matched_run}

Research map:
{research_map_json}

Survey context:
{survey_text}

Text mind map:
{mindmap_text}

Location-specific geologic context:
{geo_context}

Evidence acquired during discovery rounds:
{acquired_evidence_json}

Final completeness assessment:
{completeness_json}

Discovery loop stop reason:
{stop_reason}

Write a hypothesis-evaluation report:
1. Verdict: supported, partly supported, inconclusive, challenged, or unsupported.
2. Predictions or claims tested.
3. Supporting evidence and its directness.
4. Contradictory or non-diagnostic evidence.
5. Competing hypotheses and discriminating observations.
6. Completeness, uncertainty, and remaining gaps.
7. Final scope-calibrated answer.
""".strip()


# Backward-compatible names used by older callers.
MINERAL_CONFIDENCE_SYSTEM_PROMPT = DETECTION_CONFIDENCE_SYSTEM_PROMPT
MINERAL_CONFIDENCE_USER_PROMPT = DETECTION_CONFIDENCE_USER_PROMPT
