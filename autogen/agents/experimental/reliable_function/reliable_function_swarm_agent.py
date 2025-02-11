
# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
from dataclasses import Field
import traceback

from asyncio import iscoroutinefunction
import asyncio
import copy
from inspect import Parameter, Signature, signature
from typing import Annotated, Any, Callable, Dict, List, Optional, Tuple, Union

from openai import BaseModel

from autogen import ConversableAgent, GroupChat

from autogen.agentchat.agent import Agent
from autogen.agentchat.contrib.swarm_agent import (
    AfterWork,
    AfterWorkOption,
    SwarmAgent,
    SwarmResult,
    initiate_swarm_chat,
    register_hand_off
)
from autogen.tools.dependency_injection import Field as AG2Field

HYPOTHESIS_DEF = "A hypothesis for the outcome of invoking this function."
VALIDATION_RESULT_RESULT_DEF = (
    "Whether or not the result of the function invocation satisfies the specified Validations."
)
VALIDATION_RESULT_JUSTIFICATION_DEF = "Justification for validation_result decision."
    
class ValidationResult(BaseModel):
    validation_result: Annotated[bool, Field(description=VALIDATION_RESULT_RESULT_DEF)]
    justification: Annotated[str, Field(description=VALIDATION_RESULT_JUSTIFICATION_DEF)]

    def __str__(self) -> str:
        return (
            f"Validation Result: {'Passed' if self.validation_result else 'Failed'}\n"
            f"Justification: {self.justification}"
        )
class ReliableFunctionContext(BaseModel):
    task: Annotated[Optional[str], Field(description="The task for this invocation of the agent.")]
    result_data: Annotated[Optional[Any], Field(description="The result of the function invocation as Any type.")]
    result_str: Annotated[Optional[str], Field(description="The result of the function invocation as str type.")]
    hypothesis: Annotated[Optional[str], Field(description=HYPOTHESIS_DEF)]
    args: Annotated[Optional[list[Any]], Field(description="The args for the function invocation")]
    kwargs: Annotated[Optional[dict], Field(description="The kwargs for the function invocation")]
    swarm_result_agent: Annotated[Optional[str], Field(description="If the function returned SwarmResult with an agent, store it here.")]
    validation_result: Annotated[Optional[ValidationResult], Field(description="The result from the validator.")]

    class Config:
        arbitrary_types_allowed = True

    def __str__(self):
        return self.to_response_str()

    def to_response_str(self):
        return f"""
Task:
{self.task}
Result:
{self.result_str}
"""


# TODO: context_variables usage
# Unless something has changed, context_variables must be pickle-able, so this currently does not support
# non-pickleable return types, which is something that will need to be fixed. That is why I previously
# did not store function results in context_variables
# Ah, for example, I think you can't put a ConversableAgent instance into it

# TODO: Multiple registered functions support
# A lot will likely need to happen for this one
# Each one of them needs to be wrapped, do they each have their own unique results validator?
# Maybe just don't do this? 

# TODO: Flag to turn off hypothesis, add more?

# TODO: Better working way to explain what the function passed into this sould return, str or Any,str, or SwarmResults


# This class relies heavily on using context_variables to store state as it is working.
# It uses the key {name}-ReliableFunctionContext as the location it writes to
class ReliableFunctionSwarm(ConversableAgent):
    def __init__(
        self,
        name: str,
        runner_llm_config: dict,
        validator_llm_config: dict,
        agent_system_message: str,
        validator_system_message: str,
        function: Callable,
        max_rounds: int = 10,
        **kwargs,
    ) -> None:
        super().__init__(
            name=name,
            llm_config=None,  
            system_message="",
            **kwargs,
        )

        self._agent_system_message = agent_system_message
        self._validator_system_message = validator_system_message
        self.max_rounds = max_rounds
        self._context_variables_key = f'{self.name}-ReliableFunctionContext'
        self._validator = ConversableAgent(
            name=name+"-Validator",
            llm_config=_configure_structured_output(validator_llm_config, ValidationResult),
            **kwargs
        )
        self._runner = ConversableAgent(
            name=name+"-Runner", llm_config=runner_llm_config, functions=self._function, **kwargs
        )

        self._validator.register_hook(
            hookable_method="process_message_before_send", hook=self._result_validator_structured_output_hook
        )
        self._validator.register_hook(
            hookable_method="process_all_messages_before_reply", hook=self._only_previous_function_results_hook
        )
        self._function: ReliableFunctionWrapper = ReliableFunctionWrapper(function, self._validator, self._context_variables_key)
        
        self._runner.register_hook(hookable_method="process_message_before_send", hook=self._ensure_function_call)

        # All hand offs are done as function calls, but just in case there is an issue this will force messages back to the runner.
        register_hand_off(self._runner, AfterWork(self._runner))
         
        self.register_reply([Agent, None], self._reply)

    def _reply(self, messages: Optional[List[Dict]] = None,
                        sender: Optional[ConversableAgent] = None,
                        config: Optional[Any] = None,
                    ) -> Tuple[bool, Union[str, Dict, None]]:
        # Just to be safe for now, I haven't dug through to see what messages it returns last
        task = _get_last_non_empty_message(messages)
        sender_context = sender._context_variables
        sender_has_context = sender_context is not None
        if sender_context is None:
            sender_context = {}
        reliable_function_context:ReliableFunctionContext = self.run_func(task, messages, context_variables=sender_context)
        self._swarm_after_work = AfterWork(agent=reliable_function_context.swarm_result_agent)
        # I think you can just pass the dictionary reference back out
        # This probably breaks things still though, haven't fully thought through returning a new context_variables if the function didn't have it
        # And it is not expected. I think it is probably fine though because this is not treated as a function call so there 
        # shouldn't be anything that happens after to 
        return reliable_function_context.result_str
    
    def _only_previous_function_results_hook(self, messages: list[dict]) -> list[dict]:
        function_results = _get_last_non_empty_message(messages)

        function_result_message = {"role": "user", "name": "_User", "content": function_results}

        return self._invoked_messages + [function_result_message]

    def _result_validator_structured_output_hook(
        self, sender: Agent, message: Union[dict, str], recipient: SwarmAgent, silent: bool
    ) -> Union[dict, str]:
        message_text = message if isinstance(message, str) else message.get("content", message)

        validation_result: ValidationResult = ValidationResult.model_validate_json(message_text)

        sender._context_variables[self._context_variables_key]

        self._pending_validation_result = validation_result
        if validation_result.validation_result:
            register_hand_off(self._validator, AfterWork(AfterWorkOption.TERMINATE))
        else:
            register_hand_off(self._validator, AfterWork(self._runner))
        return str(validation_result)

    def _ensure_function_call(
        self, sender: Agent, message: Union[dict, str], recipient: SwarmAgent, silent: bool
    ) -> Union[dict, str]:
        if isinstance(message, str) and (recipient is self._runner or recipient.name == "chat_manager"):
            return {
                "name": "_User",
                "role": "user",
                "content": "Please ensure function call meets the strict schema requirements.",
            }

        return message
    
    # Roll this into run(), haven't really looked at how that is implemented yet
    def run_func(self, task: str, messages: list[dict] = [], context_variables: dict = {}) -> ReliableFunctionContext:
        # Needed because we rely on prompt for the task, could move it in to messages and not have to do this
        # Maybe could do something smarter by using UpdateSystemMessage
        self._update_prompts(task)

        reliable_function_context = ReliableFunctionContext(task=task)

        context_variables[self._context_variables_key] = reliable_function_context

        chat_result, final_context_variables, last_active_agent = initiate_swarm_chat(
            initial_agent=self._runner,
            agents=[self._runner, self._validator],
            messages=messages + [{
                "role": "user",
                "content": task,
            }],
            max_rounds=self.max_rounds,
            context_variables=context_variables,
        )

        return reliable_function_context

    def _update_prompts(self, user_input):
        self._runner.update_system_message(
            get_runner_prompt(
                user_input, self._agent_system_message
            )
        )

        self._validator.update_system_message(
            get_validator_prompt(self._validator_system_message)
        )


class ReliableFunctionWrapper:
    def __init__(self, tool_function, results_validator, context_variables_key):
        self.tool_function = tool_function
        self.results_validator = results_validator
        self._context_variables_key = context_variables_key

    def __call__(self, *args, hypothesis="", context_variables={}, **kwargs) -> SwarmResult:
        try:
            # Call the tool function
            if iscoroutinefunction(self.tool_function):
                # Await the async tool function
                result = asyncio.run( self.tool_function(context_variables=context_variables, *args, **kwargs))
            else:
                # Call the sync tool function directly
                result = self.tool_function(context_variables=context_variables, *args, **kwargs)
            
            # Not sure how to name these, result_data is Any, result_direct is what will be passed to the next agent, but can be SwarmResult
            result_data, result_direct = (None, None)

            # This is used to keep track of the agent which the SwarmResult intends to pass through.
            outer_agent_override = None

            # result is a single SwarmResult
            if isinstance(result, SwarmResult):
                outer_agent_override = result.agent
                result_data = result.values
                result_direct = result.values
            elif isinstance(result, str):
                result_data = result
                result_direct = result
            elif isinstance(result, Tuple):
                result_data, result_direct = result
                if result_direct and isinstance(result_direct, SwarmResult):
                    outer_agent_override = result_direct.agent
                    result_direct = result_direct.values

            # Just so I don't have to put str() everywhere
            result_str = str(result_direct)
            
            # Can be sure its set before this happens
            context_state:ReliableFunctionContext = context_variables[self._context_variables_key]
            context_state.args = args
            context_state.kwargs = kwargs
            context_state.hypothesis = hypothesis
            context_state.result_data = result_data
            context_state.result_str = result_str
            # A little weird, but it needs to be a string since complex classes like Agents can't be in context variables I think 
            context_state.swarm_result_agent = outer_agent_override if isinstance(outer_agent_override, str) else outer_agent_override.name
            
            # When the function returns a SwarmResult, then expect that this function call terminates the research and the LearningAgent passes to the result.
            return SwarmResult(
                context_variables=context_variables,
                agent=self.results_validator,
                values=result_str,
            )
        except Exception as e:
            # Capture the exception and debug it, not sure if I need to do this actually?
            
            print(traceback.format_exc())
            return f"""There was an issue with your tool call. You must strictly adhere to the schema in the function specification.
            Error:
            {str(e)}"""

    @property
    def __signature__(self):
        """
        Generate a custom function signature for the wrapper.

        Includes context_variables and hypothesis as parameters.
        """
        tool_sig = signature(self.tool_function)
        params = list(tool_sig.parameters.values())

        context_variables_param = Parameter(
            "context_variables",
            kind=Parameter.POSITIONAL_OR_KEYWORD,
            annotation=dict,
        )

        hypothesis_param = Parameter(
            "hypothesis",
            kind=Parameter.POSITIONAL_OR_KEYWORD,
            annotation=Annotated[str, AG2Field(description=HYPOTHESIS_DEF)],
        )

        # Insert the new parameters before args/kwargs
        params = [p for p in params if p.name not in ("context_variables", "hypothesis")] + [hypothesis_param, context_variables_param]

        # Return the updated signature
        return Signature(parameters=params, return_annotation=str)

    @property
    def __doc__(self):
        return self.tool_function.__doc__ or ""

    @property
    def __name__(self):
        # Forward the name of the tool function
        return self.tool_function.__name__


# non-destructive
# TODO Not sure if it is acceptable to mess with a llm_config
def _configure_structured_output(llm_config, structured_output_type):
    llm_config_copy = copy.deepcopy(llm_config)
    for config in llm_config_copy["config_list"]:
        config["response_format"] = structured_output_type
    return llm_config_copy


# TODO Check if there is an actual util for this
def _get_last_non_empty_message(messages):
    for message in reversed(messages):
        if message["content"]:
            return message["content"]
    return None


def get_runner_prompt(task, agent_system_message) -> str:
    prompt = f"""
Task: You are responsible for invoking the function provided to you based on the input from the user.
You can only invoke one function one time per response. Do not make a list of function calls.

{agent_system_message}

Task:
{task}
"""
    return prompt


def get_validator_prompt(validator_system_message) -> str:
    return f"""
Task: You will be provided user_input and result. Perform the following validations on the result based on the user_input.

{validator_system_message}
"""
