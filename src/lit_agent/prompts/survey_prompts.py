# src/lit_agent/prompts/survey_prompts.py

RELEVANCE_SYSTEM_PROMPT = """
You are a scientific literature relevance screening assistant.
Judge whether each candidate paper is relevant to the given scientific question.
Return only valid JSON.
""".strip()





RELEVANCE_USER_PROMPT = """
Scientific question:
{query}

Candidate papers:
{papers_json}

Task:
For each candidate paper, judge whether it is relevant to the scientific question.

Each candidate contains only:
- row_id
- paper_id
- research_region
- content_source
- content_text

content_text is selected by the program:
- use abstract_note if it is available;
- otherwise use research_content.

Use only research_region and content_text to judge relevance.

When judging relevance, infer from the scientific question:
- the target body, region, or geological setting;
- the scientific object, process, hypothesis, or phenomenon;
- whether the paper may provide useful evidence, context, observations, or constraints for answering the question.

Relevance levels:
- "high": directly relevant to answering the question.
- "medium": useful contextual or background evidence for the question.
- "low": weakly related but not useful enough.
- "irrelevant": unrelated to the question.

Selection rule:
- is_selected = true only when relevance_level is "high" or "medium".
- is_selected = false when relevance_level is "low" or "irrelevant".

For each candidate, return only:
- row_id
- relevance_level
- is_selected
- reason

Return only a valid JSON list.
No Markdown.
""".strip()



PAPER_QUERY_CONTENT_SYSTEM_PROMPT = """
You are a planetary science paper-reading assistant.
Read one full paper and write one query-centered research-status note.
Do not write a generic paper summary.
Return only valid JSON.
""".strip()


PAPER_QUERY_CONTENT_USER_PROMPT = """
Scientific question:
{query}

Paper metadata:
{paper_json}

Full paper text:
{pdf_text}

Task:
Read this paper and write one query-centered research-status note.

Important:
- Paper metadata contains content_source and content_text.
- content_text is abstract_note if available; otherwise research_content.
- Use content_text only as orientation.
- Base the note mainly on the full paper text and the scientific question.
- Do not summarize the paper generally.
- Focus on what this paper contributes to the current research status around the scientific question.
- Mention uncertainty or limitation if important.
- Preserve paper_id.

Return this structure as valid JSON:
{{
  "paper_id": "",
  "query_centered_content": ""
}}

Return only valid JSON.
No Markdown.
""".strip()



PAPER_DEEP_ANALYSIS_SYSTEM_PROMPT = """
You are an expert planetary science literature analyst.
Analyze one paper with respect to the given scientific question.
Be evidence-based, cautious, and concise.
Return valid JSON when requested.
""".strip()


PAPER_DEEP_ANALYSIS_ROUND1_PROMPT = """
Scientific question:
{query}

Paper metadata:
{paper_json}

Full paper text:
{pdf_text}

Round 1 task: initial synthesis and follow-up questions.

Read the paper with the scientific question in mind.

Your task:
1. Summarize the paper's main contribution to the scientific question.
2. Identify the most important evidence, data, observations, methods, or interpretations that should be checked more carefully.
3. Formulate follow-up questions for the next reasoning round.
4. Decide whether the paper needs deeper reasoning in Round 2.

Important:
- Do not write a generic paper summary.
- Focus only on the relation between this paper and the scientific question.
- The follow-up questions should guide a deeper second-round analysis.

Return this structure as valid JSON:
{{
  "paper_id": "",
  "round1_summary": "",
  "key_relevance_to_query": "",
  "important_evidence_to_check": [],
  "follow_up_questions": [],
  "continue_reasoning": true
}}

Return only valid JSON.
No Markdown.
""".strip()


PAPER_DEEP_ANALYSIS_ROUND2_PROMPT = """
Scientific question:
{query}

Paper metadata:
{paper_json}

Round 1 result:
{round1_json}

Full paper text:
{pdf_text}

Round 2 task: answer the follow-up questions through deeper evidence reasoning.

Use the full paper text to answer the follow-up questions from Round 1.

Focus on:
1. What evidence in the paper is most relevant to the scientific question.
2. Whether the evidence supports, constrains, complicates, or challenges the question.
3. Which interpretations are justified, and which would be overclaims.
4. What uncertainty or limitation remains.
5. Whether a third focused reasoning round is still necessary.

Critical rule for Round 3:
- If needs_round3 is true, round3_focus_questions must be a non-empty list of specific unresolved questions.
- If needs_round3 is false, round3_focus_questions must be an empty list [].

Return this structure as valid JSON:
{{
  "paper_id": "",
  "round2_deep_reasoning": "",
  "answered_questions": [],
  "remaining_uncertainties": [],
  "evidence_role": "supports | constrains | complicates | challenges | background",
  "needs_round3": false,
  "round3_focus_questions": []
}}

Return only valid JSON.
No Markdown.
""".strip()


PAPER_DEEP_ANALYSIS_ROUND3_PROMPT = """
Scientific question:
{query}

Paper metadata:
{paper_json}

Round 1 result:
{round1_json}

Round 2 result:
{round2_json}

Full paper text:
{pdf_text}

Round 3 task: focused expert reasoning.

Only perform this round because Round 2 indicated that one more focused reasoning step is needed.

Unresolved focus questions:
{round3_focus_questions}

Rules:
1. Do not repeat the whole paper summary.
2. Do not repeat Round 1 or Round 2.
3. Focus only on resolving the remaining scientific uncertainty as far as the paper allows.
4. If the paper cannot resolve a question, state that clearly.

Return this structure as valid JSON:
{{
  "paper_id": "",
  "round3_focused_reasoning": "",
  "resolved_points": [],
  "remaining_uncertainties": []
}}

Return only valid JSON.
No Markdown.
""".strip()


PAPER_DEEP_ANALYSIS_FINAL_PROMPT = """
Scientific question:
{query}

Paper metadata:
{paper_json}

Round 1 result:
{round1_json}

Round 2 result:
{round2_json}

Round 3 result, if available:
{round3_json}

Task:
Write the final expert-level deep analysis for this paper.

Requirements:
1. Do not write a generic paper summary.
2. Focus only on this paper's contribution to the current research status around the scientific question.
3. Explain the relevance, evidence, interpretation, uncertainty, and limitation.
4. Clearly distinguish what the paper supports from what it does not prove.
5. Be scientifically cautious and concise.
6. Preserve paper_id.

Return this structure as valid JSON:
{{
  "paper_id": "",
  "expert_level_deep_analysis": {{
    "round1_summary": "",
    "round2_deep_reasoning": "",
    "round3_focused_reasoning": "",
    "final_deep_analysis": ""
  }}
}}

Return only valid JSON.
No Markdown.
""".strip()


FINAL_SURVEY_SYSTEM_PROMPT = """
You are a planetary geology survey writer.
Write a coherent, evidence-based, query-centered research status report.
Use concise scientific English.
""".strip()


FINAL_SURVEY_USER_PROMPT = """
Scientific question:
{query}

Literature records:
{paper_records_json}

Task:
Write survey.txt.

Each literature record has already been compacted by the program.

For each record:
- Use record_content as the main paper-level information.
- If reasoning_trace is provided, use it only to explain evidence chains, uncertainty, and limitations.
- Use metadata such as year, title, journal, authors, citation, file_name, research_region, relevance_level, and record_source only for organization and attribution.

Citation requirements:
1. Use inline author-year citations in the narrative, for example:
   Early studies (Fairén et al., 2003) proposed episodic flood inundations...
   Subsequent work (Pajola et al., 2016; Adler et al., 2019) identified fluvial terraces...
2. Use the provided citation field. Do not invent author names or years.
3. Cite papers when introducing evidence, interpretations, or research claims.
4. It is acceptable to group multiple papers in one citation.
5. Do not cite every sentence mechanically. One citation can support a short connected claim when the source is clear.

Writing requirements:
1. Write a query-centered research-status report, not isolated paper summaries.
2. Organize the report around evidence themes that emerge from the selected papers.
3. Preserve file_name when discussing individual literature contributions.
4. Organize individual literature contributions chronologically.
5. Distinguish direct evidence, contextual evidence, uncertainty, and alternative interpretations when they are present.
6. If a paper does not clearly provide uncertainty or limitation, do not invent one.
7. In full-mode records, use reasoning_trace only when it helps explain evidence chains, unresolved issues, or limits of interpretation.

Required overall structure:

Scientific Question:
...

1. Overall Research Status
...

2. Evidence Framework
Create suitable subsections based on the selected papers.
Each subsection should synthesize evidence rather than list papers.
Use inline author-year citations.

3. Chronological Literature Contributions
For each paper, write a concise paragraph or short structured entry.
Each entry should include:
- [Year] Title citation
- Authors:
- Journal:
- File:
- A query-centered contribution statement

When relevant, also include:
- Key evidence or data:
- Interpretation for the scientific question:
- Uncertainty, limitation, or alternative explanation:

Do not force all optional fields if they are not supported by the record.

4. Synthesis Toward the Scientific Question
...

5. Remaining Research Gaps
...
""".strip()



FINAL_MINDMAP_SYSTEM_PROMPT = """
You are a scientific mind-map generator.
Convert a survey and compact literature records into a concise hierarchical text mind map.
""".strip()

FINAL_MINDMAP_USER_PROMPT = """
Scientific question:
{query}

Survey report:
{survey_text}

Literature records:
{paper_records_json}

Task:
Generate MindMap.txt.

Each literature record has already been compacted by the program.
Use record_content as the paper-level knowledge node.
Use metadata such as year, title, file_name, research_region, relevance_level, record_source, and citation only for organization and attribution.

Requirements:
1. Use a concise hierarchical outline.
2. Organize around the scientific question.
3. Reflect evidence themes, hypothesis relations, uncertainties, and literature nodes.
4. Preserve file_name for each literature node.
5. Avoid long paragraphs.
6. Avoid duplicate nodes.
7. Do not force empty categories. If a category has no evidence, omit it or mark it briefly as unresolved.

Suggested structure. Adapt section names to the actual literature:

Scientific Question:
...

1. Central Question / Hypothesis
   1.1 Main interpretation
   1.2 Alternative interpretations, if present

2. Evidence Framework
   2.1 Evidence theme
       - Key evidence:
       - Supporting papers:
       - Uncertainty or limitation, if present:
   2.2 Evidence theme
       - Key evidence:
       - Supporting papers:
       - Uncertainty or limitation, if present:

3. Chronological Development
   3.1 [Year] Title citation
       - File:
       - Relevance level:
       - Record source:
       - Query-centered contribution:
       - Limitation or unresolved issue, if present:

4. Conflicting or Uncertain Evidence, if present

5. Research Gaps
""".strip()