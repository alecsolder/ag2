# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
import json
from typing import Annotated

import autogen
from autogen.agentchat.contrib.learning.knowledge_function_swarm_agent import KnowledgeFunctionSwarmAgent
from autogen.agentchat.contrib.learning.knowledge_sharing_swarm_agent import (
    get_all_agent_memories,
    restore_agent_memories,
)
from autogen.agentchat.contrib.toa.prompts import (
    ANSWERER_PROMPT,
    ANSWERER_VALIDATOR_PROMPT,
    ANSWER_VERIFIER_PROMPT,
    ANSWER_VERIFIER_VALIDATOR_PROMPT,
    PARAGRAPH_AGENT_PROMPT,
    PARAGRAPH_VALIDATOR_PROMPT,
    QUESTIONER_PROMPT,
    QUESTIONER_VALIDATOR_PROMPT,
    SENTENCE_AGENT_PROMPT,
    SENTENCE_VALIDATOR_PROMPT,
)
from autogen.agentchat.contrib.toa.toa_logging import (
    Level,
    f_text,
    log,
    log_beam_answer,
    log_beam_answers,
    log_levels,
    log_sentence_result,
    log_sub_questions,
)
from autogen.tools.dependency_injection import Field as AG2Field

key_map = {"gpt-4o-mini": "OPENAI_API_KEY"}

config_list = autogen.config_list_from_json(env_or_file="/workspaces/ag2/OAI_CONFIG_LIST")

local_llm_config = {
    "config_list": [
        {
            "model": "qwen2.5-7b-instruct",
            "base_url": "http://192.168.0.169:1234/v1",
        },
    ],
    "cache_seed": None,
}

openai_llm_config = {
    "cache_seed": 42,
    "temperature": 1,
    "config_list": config_list,
    "timeout": 120,
}


def answer_question(
    context_variables: dict,
    answer: Annotated[
        str, AG2Field(description="The answer to the question created strictly from facts presented to you")
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


answerer = KnowledgeFunctionSwarmAgent(
    name="Answerer",
    researcher_llm_config=openai_llm_config,
    result_validator_llm_config=openai_llm_config,
    agent_system_message=ANSWERER_PROMPT,
    validator_system_message=ANSWERER_VALIDATOR_PROMPT,
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


answer_verifier = KnowledgeFunctionSwarmAgent(
    name="AnswerVerifier",
    researcher_llm_config=openai_llm_config,
    result_validator_llm_config=openai_llm_config,
    agent_system_message=ANSWER_VERIFIER_PROMPT,
    validator_system_message=ANSWER_VERIFIER_VALIDATOR_PROMPT,
    functions=[verify_final_answer],
    silent=False,
)


def submit_sub_questions(
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
        out += f"    {i}: {f_text(sub_question, 2)}\n"

    return sub_questions, out


questioner = KnowledgeFunctionSwarmAgent(
    name="Questioner",
    researcher_llm_config=openai_llm_config,
    result_validator_llm_config=openai_llm_config,
    agent_system_message=QUESTIONER_PROMPT,
    functions=[submit_sub_questions],
    validator_system_message=QUESTIONER_VALIDATOR_PROMPT,
    silent=False,
)


def record_fact(
    context_variables: dict,
    fact: Annotated[
        str,
        AG2Field(
            description="""A fact which provides information and is directly relevant to the provided question.
The fact must follow these instructions:
- If the provided sentence is unrelated to the question, publish a fact with the text exactly as: "IRRELEVANT"
- The fact must be derived directly from the sentence provided and the context of the question. No background information can be used.
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


test_case_json = json.load(open("/workspaces/ag2/autogen/agentchat/contrib/toa/test_case.json", "r"))

test_case = test_case_json[0]
question = test_case["question"]
test_case_actual_answer = test_case["answer"]
context = test_case["context"]

agent_map = {}
all_sentence_agents = []
for i, paragraph in enumerate(context):
    paragraph_name = paragraph[0]
    sentences = paragraph[1]
    paragraph_text = "\n".join(sentences)
    paragraph_agent = KnowledgeFunctionSwarmAgent(
        name="ParagraphAgent",
        researcher_llm_config=local_llm_config,
        result_validator_llm_config=local_llm_config,
        agent_system_message=PARAGRAPH_AGENT_PROMPT % (paragraph_text),
        validator_system_message=PARAGRAPH_VALIDATOR_PROMPT,
        max_rounds=10,
        functions=[grade_fact],
        silent=False,
    )

    sentence_agents = []

    for sentence in sentences:
        sentence_agent = KnowledgeFunctionSwarmAgent(
            name="SentenceAgent",
            researcher_llm_config=local_llm_config,
            result_validator_llm_config=local_llm_config,
            agent_system_message=SENTENCE_AGENT_PROMPT % (sentence),
            validator_system_message=SENTENCE_VALIDATOR_PROMPT % (sentence),
            max_rounds=10,
            functions=[record_fact],
            silent=False,
        )
        sentence_agent.sentence = sentence
        sentence_agents.append(sentence_agent)
    paragraph_agent.register_knowledge_sources(sentence_agents)
    all_sentence_agents += sentence_agents

    agent_map[paragraph_name] = (paragraph_agent, sentence_agents)

paragraph_agents = []
for paragraph_name, (paragraph_agent, sentence_agents) in agent_map.items():
    paragraph_agents.append(paragraph_agent)

questioner.register_knowledge_sources(paragraph_agents)
answerer.register_knowledge_sources(paragraph_agents)
answer_verifier.register_knowledge_sources(paragraph_agents)
confidence = 0

levels = []


def record_level(
    level_num, question, answer, confidence=0, confidence_justification="", answer_justification="", sentence_results=[]
):
    levels.append(
        Level(
            level_num=level_num,
            question=question,
            answer=answer,
            answer_justification=answer_justification,
            confidence=confidence,
            confidence_justification=confidence_justification,
            sentence_results=sentence_results,
        )
    )


log("\n\n\n\n\n")
record_level(0, question, test_case_actual_answer)
log_levels(levels)
log("\n\n")
level_num = 0
while level_num < 3 or int(confidence) < 90:
    level_num += 1
    log(f"############# Beginning Level {str(level_num)} ##################")
    sub_questions, sub_questions_memory = questioner.auto_gen_func(user_input=f"""Question: {f_text(question)}""")

    log_sub_questions(sub_questions)

    log("### Beginning Beam Search ###")
    beam_answers = []
    original_memories = get_all_agent_memories(all_sentence_agents)
    for i, sub_question in enumerate(sub_questions):
        pre = f"Level {str(level_num)} : Beam {i + 1}  :"
        log(f"Sub Question: {sub_question}", pre, 1)
        facts_to_keep = []
        # Go through every paragraph every loop
        paragraph_count = 0
        all_sentence_results = []
        for paragraph_name, (paragraph_agent, sentence_agents) in agent_map.items():
            paragraph_count += 1
            sentence_results = []
            log(f"Paragraph {paragraph_count}: " + paragraph_name, pre, 2)
            sentence_num = 0
            for sentence_agent in sentence_agents:
                sentence_num += 1
                fact, fact_memory = sentence_agent.auto_gen_func(user_input=sub_question)

                grade_res = None
                grade = 0
                grade_justification = ""
                if "irrelevant" in fact.lower():
                    log(f"{sentence_num}: is irrelevant to the question.", prefix=pre, indent_num=3)
                elif "repeat" in fact.lower():
                    log(f"{sentence_num}: is a repeat of an existing fact.", prefix=pre, indent_num=3)
                else:
                    q = f"New Fact:\n{fact}\nQuestion:\n{sub_question}"
                    (grade, grade_justification), _ = paragraph_agent.auto_gen_func(user_input=q)
                    sentence_result = (grade, grade_justification, fact, fact_memory, sentence_agent, paragraph_name)
                    log_sentence_result(sentence_result, pre, 3)

                    sentence_results.append(sentence_result)
                    all_sentence_results.append(sentence_result)

            for grade, grade_justification, fact, fact_memory, sentence_agent, paragraph_name in all_sentence_results:
                if int(grade) >= 50:
                    facts_to_keep.append(
                        (grade, grade_justification, fact, fact_memory, sentence_agent, paragraph_name)
                    )

        for (
            best_grade,
            best_grade_justification,
            best_fact,
            best_fact_res,
            best_fact_sentence_agent,
            best_paragraph_name,
        ) in facts_to_keep:
            best_fact_sentence_agent.remember(best_fact_res)
        (answer, answer_justification), _ = answerer.auto_gen_func(user_input=question)
        (confidence, confidence_justification), _ = answer_verifier.auto_gen_func(
            user_input=f"""
Question:
{question}
Answer:
{answer}
Answer Justification:
{answer_justification}
"""
        )
        beam_memories = get_all_agent_memories(all_sentence_agents)
        beam_answers.append(
            (
                answer,
                answer_justification,
                confidence,
                confidence_justification,
                sub_question,
                all_sentence_results,
                beam_memories,
                i,
            )
        )
        restore_agent_memories(all_sentence_agents, original_memories)

    log_beam_answers(beam_answers)

    best_beam_answer = max(beam_answers, key=lambda x: int(x[2]))
    (
        answer,
        answer_justification,
        confidence,
        confidence_justification,
        sub_question,
        all_sentence_results,
        beam_memories,
        best_level_num,
    ) = best_beam_answer
    log("Winning Beam:")
    log_beam_answer(best_beam_answer)
    restore_agent_memories(all_sentence_agents, beam_memories)
    record_level(
        level_num=level_num,
        question=sub_question,
        answer=answer,
        answer_justification=answer_justification,
        sentence_results=all_sentence_results,
        confidence=confidence,
        confidence_justification=confidence_justification,
    )
    log_levels(levels)
log("########################################################")
log("#################### FINAL RESULT ######################")
