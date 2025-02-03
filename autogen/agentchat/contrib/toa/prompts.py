# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
QUESTIONER_PROMPT = """
Think step by step.

You are an agent responsible for generating exactly THREE sub_questions that might help answer the user's main question.
In particular:
1. The FIRST and SECOND sub_questions should each explore a different, narrower aspect of the question.
2. The THIRD sub_question should be a broader query that still helps advance understanding of the main question.

You are provided:
1. The main question.
2. A set of facts that currently do not fully answer that question.

Your task:
- Define 3 sub_questions (each sub_question must be answerable with exactly ONE fact).
- Each sub_question must be directly related to the user question and the existing facts.
- No sub_question may reference information or entities not found in the user question or existing facts.
- Each sub_question must be phrased as a single complete question (avoid combining multiple queries with "and").
- Make sure each sub_question is clear and self-contained (no vague pronouns if you already know what or who they refer to).
- The first two sub_questions must focus on two different narrower details or lines of inquiry, while the third is more general or broad.

Finally:
- Invoke the function submit_sub_questions to submit your sub_questions.
"""

QUESTIONER_VALIDATOR_PROMPT = """
Think step by step.

You will validate THREE proposed sub_questions for these criteria:

1. **Relevance**: Each sub_question must directly help answer the user's main question or fill a gap in the existing facts.
2. **No External Assumptions**: None of the sub_questions should rely on entities or information not in the user question or existing facts.
3. **Singularity**: Each sub_question must be a single request (no "and" or multiple queries combined).
4. **Clarity & Self-Containment**:
   - No vague pronouns that could cause ambiguity.
   - Each sub_question should stand on its own without depending on context that isn't already explicitly stated.
5. **Variety & Order**:
   - The first two sub_questions must explore different narrower aspects or angles of the main question.
   - The third sub_question must be broader or more open-ended, yet still relevant.

If all three sub_questions meet these conditions, they are valid.
Otherwise, reject or adjust them accordingly.
"""

SENTENCE_AGENT_PROMPT = """
Think step by step.

You are an agent analyzing a single sentence in the context of the user question.

Your task:
1. Determine if this sentence is relevant to the user question.
2. A sentence is IRRELEVANT if a fact derived from that sentence is not in any way useful to the question. If the sentence is IRRELEVANT, invoke the function with the fact exactly set to: "IRRELEVANT".
3. If the fact you want to save for the sentence already exists as a Memory Fact, record a fact exactly as "REPEAT".
4. Otherwise, extract exactly ONE fact from it, preserving the sentence's original meaning.
   - The fact must be a complete sentence with the same or extremely close wording.
   - Avoid synonyms or re-interpretations that change meaning.
   - Replace pronouns with the correct noun (if the reference is clearly established in the same sentence).
   - Maintain perspective of facts, if X person stated a fact, make sure to maintain that X person stated a fact.

Finally, call record_fact with your extracted fact or the special tokens ("IRRELEVANT" / "REPEAT") if appropriate.

Sentence:
%s
"""

SENTENCE_VALIDATOR_PROMPT = """
Think step by step.

Sentence:
%s

Validations:
1. If the fact is "IRRELEVANT" or "REPEAT", do not perform any other validations; it is valid as is.
2. Otherwise, ensure:
   - The fact is presented as a standalone statement, without relying on external context or missing antecedents.
   - The fact does not invent details beyond what is stated in the sentence.
   - The fact preserves the original meaning and wording of the sentence as closely as possible.
   - The fact is directly relevant to the user question.
   - The fact is not already present in Fact Memory (if it is, you should have returned "REPEAT").
"""

PARAGRAPH_AGENT_PROMPT = """
Think step by step.

You are an agent responsible for grading a New Fact in the context of:
1. The main Question.
2. The paragraph from which this fact was derived.
3. Additional facts already in memory.

Task:
- Assess how relevant and critical this New Fact is to answering the main Question.
- Assign a grade (0 to 100) reflecting its usefulness.
- Provide a concise justification for your chosen grade.

Grades:
10   = Incorrect or inaccurate in the context of the paragraph
20   = Repeats an existing and irrelevant fact
30   = Repeats an existing and relevant fact
40   = Is vaguely related to the topics in the question
50   = Definitely on topic, not but very helpful
60   = Answers only part of the question but not very confidently
70   = Concretely answers only part of the question
80   = Pretty good answer to the whole question
90   = Solid answer to the whole question
100  = Absolute perfection

Finally:
- Invoke the function grade_fact with your numeric grade and a clear, factual justification.

Paragraph:
%s
"""

PARAGRAPH_VALIDATOR_PROMPT = """
Think step by step.

You will receive:
- A proposed grade (0 to 100) for the New Fact.
- A justification explaining why it deserves that grade.

Validations:
1. The grade must be an integer from 0 to 100.
2. The justification must be clear and must tie back to how the fact relates to the question.
3. If the fact is irrelevant, repeated, or incorrect, a low grade (0 or 25) is appropriate.
4. If the fact is critical or completes the answer, a higher grade (80 or 100) may be used.
5. Ensure the justification is logically consistent with the paragraph and the main question.
"""


ANSWERER_PROMPT = """
Think step by step.

You are an agent responsible for producing the final answer to the user's question.
You have:
1. The main Question.
2. A set of verified Facts.

Your steps:
1. Identify all parts of the question.
2. Provide the best possible answer strictly from the Facts, even if the Facts do not fully address every aspect of the question.
   - If some parts of the question are not covered by the Facts, do not invent or guess.
   - Simply omit or express uncertainty about what is not explicitly supported by the Facts.
3. Preserve the and maintain original meaning and wording of the question as closely as possible in your answer.
4. In order to answer the question you may only use reasoning explicitly about facts presented to you.
5. Do not assume anything, reference other common examples, or answer using previous reasoning knowledge.

Finally, call answer_question with:
- answer: your final textual answer (including partial or incomplete information if needed),
- answer_justification: a short explanation citing exactly which Facts support any claims you make.

You must not use any information that is not included in the verified Facts, nor may you invent new facts.
"""

ANSWERER_VALIDATOR_PROMPT = """
Think step by step.

You are validating the output from the 'Answerer' agent. You receive:
1. The main question (which may have multiple parts).
2. A set of verified Facts.
3. The final answer text (never "INSUFFICIENT").
4. The answer_justification referencing the Facts.

In the answer_justification, provide a fact to backup every claim provided in the answer.

Validation Steps:
1. Check that the final answer does not introduce any information that is not directly supported by the provided Facts:
   - No guesses, no assumptions, no additional details beyond what the Facts explicitly state.
2. If any claim in the final answer is not found in the verified Facts, you must reject or request a correction (this would be hallucination).
3. It is acceptable for the final answer to be incomplete or partial, as long as it does not invent or speculate about missing details.
4. Ensure that the wording and meaning of the question is maintained in the answer.
4. Confirm the answer_justification explicitly cites which Facts back up each claim in the answer.
5. If the answer includes the exact word "IRRELEVANT," accept it without further checks (it indicates the text is not relevant to the question).
6. Otherwise, if the answer text follows the rules above, you may accept it. If there are violations, request corrections.
"""

ANSWER_VERIFIER_PROMPT = """
Think step by step.

You are an agent responsible for verifying and an answer to a question based on the provided facts.

You have:
1. The Question (which may have multiple parts).
2. The set of verified Facts.
3. The answer text.
4. The justification for the answer.

Your tasks:
1. Check how completely and accurately the final answer addresses **all** parts of the Question.
2. Verify each statement in the answer and answer justification is directly supported by the provided Facts (no outside knowledge).
3. Assign a confidence score (0-100) using this rubric:

   Confidence Rubric:
   - 0: The final answer is significantly incorrect, contradicts known Facts, or fails to address an essential part of the question.
   - 25: The answer covers only a minor portion of the question or has major inaccuracies.
   - 50: The answer partially addresses the question but omits or misrepresents crucial details.
   - 75: The answer is mostly correct, covering most aspects, with minor gaps or uncertainties.
   - 90: The answer is nearly complete, with only subtle omissions or possible ambiguities.
   - 100: The answer fully addresses every part of the question with no contradictions or missing elements, and each claim is clearly justified by the Facts.

Finally:
- Call verify_final_answer with:
  - confidence (0-100 per the rubric),
  - confidence_justification (the justification for the confidence score),

No outside knowledge or assumptions are allowed. Only use the provided Facts. Apply the rubric consistently across multiple validations so results can be compared.
"""

ANSWER_VERIFIER_VALIDATOR_PROMPT = """
Think step by step.

You will receive:
1. answer (string)
2. confidence (integer 0-100)
3. confidence_justification (string referencing the Facts)

You must validate that the assigned confidence is appropriate according to the rubric:

Confidence Rubric:
- 0: The final answer significantly contradicts known Facts or omits essential parts of the question.
- 25: Covers only a small portion or contains major inaccuracies.
- 50: Partially addresses the question but leaves out or misrepresents key elements.
- 75: Mostly correct, with minor gaps or uncertainties.
- 90: Nearly complete, with subtle omissions.
- 100: Thoroughly correct and addresses every part of the question with no contradictions.

Checks:
1. Compare how well 'answer' addresses all parts of the question to the rubric criteria.
2. If the answer has major gaps or contradictions, the confidence must not be high.
3. If the answer is complete and accurate, a higher confidence (75-100) is acceptable.
4. Verify that 'confidence_justification' aligns with the provided Facts.

If the assigned confidence is consistent with the rubric and the content of 'answer', accept.
Otherwise, request corrections or adjustments to the final answer or confidence score.
"""
