
# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
import functools
import inspect
import traceback

from asyncio import iscoroutinefunction
import asyncio
import copy
from inspect import Parameter, Signature, markcoroutinefunction, signature
from typing import Annotated, Any, Callable, Dict, List, Optional, Tuple, Union

from openai import BaseModel

from autogen import ConversableAgent, a_initiate_swarm_chat
from autogen.tools.tool import Tool
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
    
# These models need some improvement, I still don't know how to properly
# Set them up so they are ok in context_variables and structured output
class ValidationResult(BaseModel):
    validation_result: Annotated[bool, AG2Field(description=VALIDATION_RESULT_RESULT_DEF)] 
    justification: Annotated[str, AG2Field(description=VALIDATION_RESULT_JUSTIFICATION_DEF)]
    class Config:
        arbitrary_types_allowed = True
        extra = "forbid"
    def __str__(self) -> str:
        return (
            f"Validation Result: {'Passed' if self.validation_result else 'Failed'}\n"
            f"Justification: {self.justification}"
        )
        
class ReliableFunctionContext(BaseModel):
    task: Annotated[Optional[str], AG2Field(description="The task for this invocation of the agent.")]
    result_data: Annotated[Optional[Any], AG2Field(description="The result of the function invocation as Any type.")] = ""
    result_str: Annotated[Optional[str], AG2Field(description="The result of the function invocation as str type.")] = ""
    hypothesis: Annotated[Optional[str], AG2Field(description=HYPOTHESIS_DEF)] = ""
    args: Annotated[Optional[list[Any]], AG2Field(description="The args for the function invocation")] = []
    kwargs: Annotated[Optional[dict], AG2Field(description="The kwargs for the function invocation")] = {}
    swarm_result_agent: Annotated[Optional[str], AG2Field(description="If the function returned SwarmResult with an agent, store it here.")] = ""
    validation_result: Annotated[bool, AG2Field(description=VALIDATION_RESULT_RESULT_DEF)] = False
    justification: Annotated[str, AG2Field(description=VALIDATION_RESULT_JUSTIFICATION_DEF)] = ""
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
__all__ = ["ReliableFunctionAgent"]


class ReliableFunctionAgent(ConversableAgent):
    def __init__(
        self,
        name: str,
        runner_llm_config: dict,
        validator_llm_config: dict,
        agent_system_message: str,
        validator_system_message: str,
        func_or_tool: Union[Callable, Tool],
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
        if isinstance(func_or_tool, Tool):
            func_or_tool._func = reliable_function_wrapper(func_or_tool.func, self._validator, self._context_variables_key)
            self._func = func_or_tool
        else:
            self._func = reliable_function_wrapper(func_or_tool, self._validator, self._context_variables_key)

        # self._runner.register_for_llm()(self._func_or_tool)
        
        markcoroutinefunction(self._func)
        self._runner = ConversableAgent(
            name=name+"-Runner", llm_config=runner_llm_config,**kwargs
        )

        self._runner.register_for_llm()(func_or_tool)
        self._runner.register_for_execution()(func_or_tool)

        self._validator.register_hook(
            hookable_method="process_message_before_send", hook=self._validator_structured_output_hook
        )
        self._validator.register_hook(
            hookable_method="process_all_messages_before_reply", hook=self._only_previous_function_results_hook
        )

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

    def _validator_structured_output_hook(
        self, sender: Agent, message: Union[dict, str], recipient: SwarmAgent, silent: bool
    ) -> Union[dict, str]:
        message_text = message if isinstance(message, str) else message.get("content", message)

        validation_result: ValidationResult = ValidationResult.model_validate_json(message_text)

        reliable_function_context = _get_reliable_function_context(sender._context_variables,self._context_variables_key)
        reliable_function_context.validation_result = validation_result.validation_result
        reliable_function_context.justification = validation_result.justification

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
        self._invoked_messages = messages
        reliable_function_context = ReliableFunctionContext(task=task)
        _set_reliable_function_context(context_variables, self._context_variables_key, reliable_function_context)

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

        return _get_reliable_function_context(final_context_variables, self._context_variables_key)
    
    async def a_run_func(self, task: str, messages: list[dict] = [], context_variables: dict = {}) -> ReliableFunctionContext:
        # Needed because we rely on prompt for the task, could move it in to messages and not have to do this
        # Maybe could do something smarter by using UpdateSystemMessage
        self._update_prompts(task)
        self._invoked_messages = messages
        reliable_function_context = ReliableFunctionContext(task=task)
        _set_reliable_function_context(context_variables, self._context_variables_key, reliable_function_context)

        chat_result, final_context_variables, last_active_agent = await a_initiate_swarm_chat(
            initial_agent=self._runner,
            agents=[self._runner, self._validator],
            messages=messages + [{
                "role": "user",
                "content": task,
            }],
            max_rounds=self.max_rounds,
            context_variables=context_variables,
        )

        return _get_reliable_function_context(final_context_variables, self._context_variables_key)

    def _update_prompts(self, task):
        self._runner.update_system_message(
            get_runner_prompt(
                task, self._agent_system_message
            )
        )

        self._validator.update_system_message(
            get_validator_prompt(task, self._validator_system_message)
        )

def reliable_function_wrapper(tool_function, validator, context_variables_key):
    @functools.wraps(tool_function)
    async def wrapper(*args, hypothesis="", context_variables: Dict[str, str], **kwargs) -> Any:
        print(
            f"Running tool function {tool_function.__name__} with hypothesis: {hypothesis} and context_variables: {context_variables}"
        )
        # Call the tool function with the given parameters.
        result = tool_function(context_variables=context_variables, *args, **kwargs)
        # If the result is a coroutine, await it.
        if asyncio.iscoroutine(result):
            result = await result

            # Process the result into two parts.
            result_data, result_direct = (None, None)
            outer_agent_override = ""

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

            # Convert the final result to a string.
            result_str = str(result_direct)

            # Update the reliable function context.
            context_state: ReliableFunctionContext = _get_reliable_function_context(
                context_variables, context_variables_key
            )
            context_state.args = args
            context_state.kwargs = kwargs
            context_state.hypothesis = hypothesis
            context_state.result_data = result_data
            context_state.result_str = result_str
            # In case outer_agent_override has a name attribute.
            context_state.swarm_result_agent = getattr(outer_agent_override, "name", None)
            _set_reliable_function_context(context_variables, context_variables_key, context_state)

            # Return a SwarmResult.
            # The fact that it is validator.name is CRITICAL
            # As conversable_agent requires this object to be serializable 
            return SwarmResult(
                context_variables=context_variables,
                agent=validator.name,
                values=result_str,
            )

        return result

    orig_sig = inspect.signature(tool_function)
    params = list(orig_sig.parameters.values())

    # Create new parameters for hypothesis and context_variables.
    hypothesis_param = inspect.Parameter("hypothesis", kind=inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=str)
    context_variables_param = inspect.Parameter(
        "context_variables", kind=inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=Dict[str, str]
    )

    # Remove any pre-existing parameters with the same names,
    # then add our new parameters at the end.
    params = [p for p in params if p.name not in ("hypothesis", "context_variables")]
    params.extend([hypothesis_param, context_variables_param])
    wrapper.__signature__ = inspect.Signature(parameters=params, return_annotation=orig_sig.return_annotation)

    return wrapper

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

def _get_reliable_function_context(context_variables, reliable_function_context_key):
    return ReliableFunctionContext.model_validate_json(context_variables[reliable_function_context_key])

def _set_reliable_function_context(context_variables, reliable_function_context_key, reliable_function_context):
    context_variables[reliable_function_context_key] = reliable_function_context.model_dump_json()

def get_runner_prompt(task, agent_system_message) -> str:
    prompt = f"""
You are responsible for invoking the function provided to you based on the input from the user.
You can only invoke one function one time per response. Do not make a list of function calls.

{agent_system_message}

Task:
{task}
"""
    return prompt


def get_validator_prompt(task, validator_system_message) -> str:
    return f"""
You will be provided user_input and result. Perform the following validations on the result based on the task.

Task:
{task}

{validator_system_message}
"""
