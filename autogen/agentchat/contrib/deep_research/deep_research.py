# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
from typing import Annotated

import autogen
from autogen.agentchat.contrib.deep_research.prompts import ANSWER_VERIFIER_PROMPT, ANSWER_VERIFIER_VALIDATOR_PROMPT, ANSWERER_PROMPT, ANSWERER_VALIDATOR_PROMPT, QUESTIONER_PROMPT, QUESTIONER_VALIDATOR_PROMPT
from autogen.agentchat.contrib.deep_research.reliable_function_swarm import ReliableFunctionSwarm

from autogen.agentchat.contrib.deep_research.prompts import (
    FACT_GRADER_PROMPT,
    FACT_GRADER_VALIDATOR_PROMPT,
    FACT_EXTRACTOR_VALIDATOR_PROMPT,
    FACT_EXTRACTOR_PROMPT,
    SURFER_PROMPT,
    SURFER_VALIDATOR_PROMPT
)
from autogen.tools.dependency_injection import Field as AG2Field
from autogen.tools.experimental.browser_use.browser_use import BrowserUseResult, BrowserUseTool

key_map = {"gpt-4o-mini": "OPENAI_API_KEY"}

config_list = autogen.config_list_from_json(
    env_or_file="./OAI_CONFIG_LIST")

TASK_PROMPT = "You are a Deep Researcher. You are extremely inquisitive and are responsible for submitting questions to perform deep research on the internet to find out more about a topic. "

# local_llm_config = {
#     "config_list": [
#         {
#             "model": "qwen2.5-7b-instruct",
#             "base_url": "http://192.168.0.169:1234/v1",
#             "price" : [0.01, 0.01]
#         },
#     ],
#     "cache_seed": None,
# }

# openai_llm_config = {
#     "config_list": [
#         {
#             "model": "qwen2.5-7b-instruct",
#             "base_url": "http://192.168.0.169:1234/v1",
#             "price" : [0.01, 0.01]
#         },
#     ],
#     "cache_seed": None,
# }

local_llm_config = {
    # "cache_seed": 42,
    "temperature": 1,
    "config_list": config_list,
    "timeout": 120,
}

openai_llm_config = {
    # "cache_seed": 42,
    "temperature": 1,
    "config_list": config_list,
    "timeout": 120,
}

def answer_question(
    context_variables: dict,
    answer: Annotated[
        str, AG2Field(
            description="The answer to the question created strictly from facts presented to you")
    ],
    answer_justification: Annotated[
        str,
        AG2Field(
            description="The justification of the answer, the justification should explicitly cite and reference the facts used in the answer."
        ),
    ],
):
    """
    Invoke this function to answer a question strictly using the facts presented to you.
    """
    res_str = f"""Answer:
{answer}
Answer Justification:
{answer_justification}"""
    return ((answer, answer_justification), res_str)


answerer = ReliableFunctionSwarm(
    name="Answerer",
    researcher_llm_config=openai_llm_config,
    result_validator_llm_config=openai_llm_config,
    agent_system_message=ANSWERER_PROMPT % "",
    validator_system_message=ANSWERER_VALIDATOR_PROMPT % "",
    functions=[answer_question],
    silent=False,
)


def verify_final_answer(
    context_variables: dict,
    confidence: Annotated[int, AG2Field(description="A confidence score from 0-100 based on the confidence rubric.")],
    confidence_justification: Annotated[str, AG2Field(description="The justification for the confidence score")],
):
    """
    Use this function to finalize the confidence score of the answer.
    """
    res_str = f"""
Confidence: {confidence}
Confidence Justification: {confidence_justification}
"""
    return ((confidence, confidence_justification), res_str)


answer_verifier = ReliableFunctionSwarm(
    name="AnswerVerifier",
    researcher_llm_config=openai_llm_config,
    result_validator_llm_config=openai_llm_config,
    agent_system_message=ANSWER_VERIFIER_PROMPT,
    validator_system_message=ANSWER_VERIFIER_VALIDATOR_PROMPT,
    functions=[verify_final_answer],
    silent=False,
)


async def submit_sub_questions(
    context_variables: dict,
    sub_questions: Annotated[
        list[str],
        AG2Field(
            description="Based on the facts presented to you, break down the question into a list of specific sub_questions."
        ),
    ],
):
    """
    Use this function to submit sub_questions.
    """
    out = "\n"
    for i, sub_question in enumerate(sub_questions):
        out += f"    {i}: {sub_question}\n"

    return sub_questions, out


questioner = ReliableFunctionSwarm(
    name="Questioner",
    researcher_llm_config=openai_llm_config,
    result_validator_llm_config=openai_llm_config,
    agent_system_message=QUESTIONER_PROMPT % TASK_PROMPT,
    functions=[submit_sub_questions],
    validator_system_message=QUESTIONER_VALIDATOR_PROMPT % TASK_PROMPT,
    silent=False,
)

async def browse(
    context_variables: dict,
    task: Annotated[str, AG2Field(description="A task describing how to use the internet to research a topic.")],
):
    """
    Use this function to perform a task using the browser.
    """

    browser_use_tool = BrowserUseTool(
        llm_config=openai_llm_config,
        browser_config={"headless": False}
    )
    res: BrowserUseResult = await browser_use_tool(task)

    extracted_content_str = '\n'.join(res.extracted_content)
    res_str = f"""
Research Result:
{res.final_result}

Extracted Content:
{extracted_content_str}
"""
    
    return res, res_str
    
surfer = ReliableFunctionSwarm(
    name="Surfer",
    researcher_llm_config=openai_llm_config,
    result_validator_llm_config=openai_llm_config,
    agent_system_message=SURFER_PROMPT,
    validator_system_message=SURFER_VALIDATOR_PROMPT,
    functions=[browse],
    silent=False,
)

def record_fact(
    context_variables: dict,
    fact: Annotated[
        str,
        AG2Field(
            description="""A fact which provides information and is directly relevant to the provided question.
The fact must follow these instructions:
- If the provided Research Result is unrelated to the question, publish a fact with the text exactly as: "IRRELEVANT"
- The fact must be derived directly from the Research Result provided and the context of the question. No background information can be used.
- Write the fact as a complete and standalone statement. It must be meaningful without knowing what the question was.
- If the fact you want to write is already present as a Memory Fact, then submit exactly the text "REPEAT"
"""
        ),
    ],
) -> str:
    """
    Use this function to record a fact.
    """
    return fact, fact


def grade_fact(
    context_variables: dict,
    grade: Annotated[
        int,
        AG2Field(
            description="""The grade for the fact.
Any integer from 0 - 100. Below are examples of grades, but the grade can and should be in between the example numbers
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
100  = Absolute perfection"""
        ),
    ],
    grade_justification: Annotated[str, AG2Field(description="The justification for the grade.")],
):
    """
    Use this function to grade a fact.
    """
    res_str = f"""Grade: {grade}
Grade Justification:
{grade_justification}"""
    return (grade, grade_justification), res_str


fact_extractor = ReliableFunctionSwarm(
    name="FactExtractor",
    researcher_llm_config=openai_llm_config,
    result_validator_llm_config=openai_llm_config,
    agent_system_message=FACT_EXTRACTOR_PROMPT,
    validator_system_message=FACT_EXTRACTOR_VALIDATOR_PROMPT,
    functions=[record_fact],
    silent=False,
)

fact_grader = ReliableFunctionSwarm(
    name="FactExtractor",
    researcher_llm_config=openai_llm_config,
    result_validator_llm_config=openai_llm_config,
    agent_system_message=FACT_GRADER_PROMPT,
    validator_system_message=FACT_GRADER_VALIDATOR_PROMPT,
    functions=[grade_fact],
    silent=False,
)


confidence = 0
# question = "In the endnote found in the second-to-last paragraph of page 11 of the book with the doi 10.2307/j.ctv9b2xdv, what date in November was the Wikipedia article accessed? Just give the day of the month."
question = "On a leap day before the year 2008, a joke was removed from the Wikipedia page for “Dragon”. What was the phrase that was removed? Give the phrase as it appeared on the page, but without punctuation."
test_case_actual_answer = "tbd"

level_num = 0
grade_required = 50

good_facts = []
# Min 3 levels, at least 90 confidence, max 5 levels will just return answer
while (level_num < 3 or int(confidence) < 90) and level_num < 5:
    level_num += 1
    sub_questions, sub_questions_memory = questioner.auto_gen_func(
        user_input=f"""Question: {question}""")

    for i, sub_question in enumerate(sub_questions):
        pre = f"Level {str(level_num)} : Beam {i + 1}  :"
        facts_to_keep = []
        # Go through every paragraph every loop
        paragraph_count = 0
        all_sentence_results = []

        surf_res, surf_res_memory = surfer.auto_gen_func(user_input=sub_question)

        fact, fact_memory = fact_extractor.auto_gen_func(
            user_input=f"""Sub Question:
{sub_question}

{surf_res_memory.result_str}
""")
        (grade, grade_justification), fact_grade_memory = fact_grader.auto_gen_func(user_input=f"""
Sub Question:
{sub_question}

New Fact:
{fact}
""")   
        if grade > 50:
            good_facts.append(fact_memory.to_memory_message())
    (answer, answer_justification), _ = answerer.auto_gen_func(
            user_input=question, messages=good_facts)
    (confidence, confidence_justification), _ = answer_verifier.auto_gen_func(
            user_input=f"""
Question:
{question}
Answer:
{answer}
Answer Justification:
{answer_justification}
""", messages=good_facts)

print(f"""
Answer:
{answer}
Question:
{question}
Answer Justification:
{answer_justification}
Confidence:
{confidence}
Confidence Justification:
{confidence_justification}
""")
