# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
from typing import Annotated, Any, Callable

from pydantic import BaseModel, Field

RESULT_DEF = " The result of the function invocation."
HYPOTHESIS_DEF = "A hypothesis for the outcome of invoking this function."
VALIDATION_RESULT_RESULT_DEF = (
    "Whether or not the result of the function invocation satisfies the specified Validations."
)
VALIDATION_RESULT_JUSTIFICATION_DEF = "Justification for validation_result decision."


class FunctionResponse(BaseModel):
    function: Annotated[Callable, Field(description="The function that was called.")]
    hypothesis: Annotated[str, Field(description=HYPOTHESIS_DEF)]
    args: Annotated[list[Any], Field(description="The args for the function invocation")]
    kwargs: Annotated[dict, Field(description="The kwargs for the function invocation")]
    result_data: Annotated[Any, Field(description=RESULT_DEF + " As Any type.")]
    result_str: Annotated[str, Field(description=RESULT_DEF + " As a string.")]

    class Config:
        arbitrary_types_allowed = True

    def __str__(self):
        return self.to_response_str()

    def to_response_str(self, user_input):
        return f"""
User Input:
{user_input}
Result:
{self.result_str}
"""

    def formatted_response(self, user_input, input_title, output_title):
        return f"""
{input_title}:
{user_input}
{output_title}:
{self.result_str}
"""


class ValidationResult(BaseModel):
    validation_result: Annotated[bool, Field(description=VALIDATION_RESULT_RESULT_DEF)]
    justification: Annotated[str, Field(description=VALIDATION_RESULT_JUSTIFICATION_DEF)]

    def __str__(self) -> str:
        return (
            f"Validation Result: {'Passed' if self.validation_result else 'Failed'}\n"
            f"Justification: {self.justification}"
        )


class Memory(BaseModel):
    user_input: str
    result_data: Any
    result_str: str
    validation_justification: str
    # TODO lazy code, make it optional later
    result_only_memory_message: Annotated[bool, Field(default=False)]

    def __str__(self) -> str:
        return self.to_memory_message()

    # TODO lazy code for Fact: here
    def to_memory_message(self, role="user", name="memory", prefix="Memory Fact: "):
        return {
            "role": role,
            "name": name,
            "content": prefix + self.result_str
            if self.result_only_memory_message
            else f"""Memory Question: {self.user_input}
{prefix}{self.result_str}""",
        }


def answer_memory_str(memory):
    return f"""
Question: {memory.user_input}
Answer:
{memory.result_str}"""
