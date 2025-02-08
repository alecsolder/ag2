# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
import traceback

from asyncio import iscoroutinefunction
import asyncio
import copy
from inspect import Parameter, Signature, signature
from typing import Annotated, Any, Callable, List, Optional, Tuple, Union

from autogen.agentchat.agent import Agent
from autogen.agentchat.contrib.deep_research.function_models import (
    HYPOTHESIS_DEF,
    FunctionResponse,
    Memory,
    ValidationResult,
)
from autogen.agentchat.contrib.swarm_agent import (
    AFTER_WORK,
    AfterWorkOption,
    SwarmAgent,
    SwarmResult,
    initiate_swarm_chat,
    register_hand_off
)
from autogen.tools.dependency_injection import Field as AG2Field


class ReliableFunctionSwarm(SwarmAgent):
    def __init__(
        self,
        name: str,
        researcher_llm_config: dict,
        result_validator_llm_config: dict,
        agent_system_message: str,
        validator_system_message: str,
        functions: List[Callable],
        user_notes: List[str] = [],
        max_rounds: int = 10,
        use_own_knowledge: bool = False,
        verbose: bool = True,
        knowledge_sources: Optional[List[SwarmAgent]] = None,
        silent=True,
        **kwargs,
    ) -> None:
        super().__init__(
            name=name,
            llm_config=None,  # None for now, it shouldn't do LLM things
            system_message="",
            silent=silent,
            **kwargs,
        )

        self._verbose = verbose
        self.user_notes = user_notes
        self._agent_system_message = agent_system_message
        self._validator_system_message = validator_system_message
        self.max_rounds = max_rounds

        self.sentence = ""

        self._result_validator = SwarmAgent(
            name="ResultValidator-" + name,
            llm_config=_configure_structured_output(result_validator_llm_config, ValidationResult),
            silent=silent,
        )

        self._result_validator.register_hook(
            hookable_method="process_message_before_send", hook=self._result_validator_structured_output_hook
        )
        self._result_validator.register_hook(
            hookable_method="process_all_messages_before_reply", hook=self._only_previous_function_results_hook
        )
        self._original_function = functions[0]
        self._functions: list[SubSwarmFunctionWrapper] = [
            SubSwarmFunctionWrapper(function, self._function_response_callback, self._result_validator)
            for function in functions
        ]
        self._wrapped_function = self._functions[0]

        # A hack because it wants the name to be in there and I just want to keep it with this name for now
        self._memory = SwarmAgent(name="memory", silent=silent)
        self._researcher = SwarmAgent(
            name="Researcher-" + name, llm_config=researcher_llm_config, functions=self._functions, silent=silent
        )
        register_hand_off(self._researcher, AFTER_WORK(self._researcher))
        self._researcher.register_hook(hookable_method="process_message_before_send", hook=self._ensure_function_call)
        # self._researcher.register_hook(
        #     hookable_method="process_all_messages_before_reply", hook=self._remove_failed_tool_calls
        # )

    def _only_previous_function_results_hook(self, messages: list[dict]) -> list[dict]:
        function_results = _get_last_non_empty_message(messages)

        function_result_message = {"role": "user", "name": "_User", "content": function_results}

        return self._invoked_messages + [function_result_message]

    def _remove_failed_tool_calls(self, messages: list[dict]) -> list[dict]:
        if len(messages) <= 2:
            return messages
        last_message = messages[-1]

        if "FORMATTING ERROR" in last_message["content"]:
            last_message["name"] = "_User"
            last_message["role"] = "user"
            last_message["content"] = (
                "Please answer the previous question ensuring that all function calls are properly formatted."
            )
            return messages[:-1] + [last_message]

        return messages

    def _result_validator_structured_output_hook(
        self, sender: Agent, message: Union[dict, str], recipient: SwarmAgent, silent: bool
    ) -> Union[dict, str]:
        message_text = message if isinstance(message, str) else message.get("content", message)

        validation_result: ValidationResult = ValidationResult.model_validate_json(message_text)
        self._pending_validation_result = validation_result
        if validation_result.validation_result:
            register_hand_off(self._result_validator, AFTER_WORK(AfterWorkOption.TERMINATE))
        else:
            register_hand_off(self._result_validator, AFTER_WORK(self._researcher))
        return str(validation_result)

    def _ensure_function_call(
        self, sender: Agent, message: Union[dict, str], recipient: SwarmAgent, silent: bool
    ) -> Union[dict, str]:
        if isinstance(message, str) and (recipient is self._researcher or recipient.name == "chat_manager"):
            return {
                "name": "_User",
                "role": "user",
                "content": "Please ensure function call meets the strict schema requirements.",
            }

        return message

    # TODO Function results should be in a dictionary keyed by function for multi-function
    def _function_response_callback(self, function_response: FunctionResponse) -> str:
        self._pending_function_response = function_response
        return function_response.to_response_str(self._pending_user_input)

    def _get_messages(self, user_input):
        return [
            {
                "role": "user",
                # "name": "user",
                "content": user_input,
            }
        ]

    # Maybe figure out a way to do currying or dependency injection to provide stuff as default more easily.
    def auto_gen_func(self, user_input: str, messages: list[dict] = [], context_variables: dict = {}) -> Tuple[Any, Memory]:
        self._wrapped_function.num_invocations = 0
        self._pending_function_response: FunctionResponse = None
        self._pending_validation_result = None
        self._pending_user_input = user_input
        self._invoked_messages = messages
        # Must use findings and also must be run before every invocation
        self._update_prompts(user_input)

        chat_result, final_context_variables, last_active_agent = initiate_swarm_chat(
            initial_agent=self._researcher,
            agents=[self._researcher, self._result_validator, self._memory],
            # Trying out passing the memories in as messages
            messages=messages + self._get_messages(user_input),
            max_rounds=self.max_rounds,
            # I think this works as a reference everywhere basically
            context_variables=context_variables,
        )

        memory: Memory = Memory(
            user_input=user_input,
            result_data=self._pending_function_response.result_data,
            result_str=self._pending_function_response.result_str,
            validation_justification=""
            if not self._pending_validation_result
            else self._pending_validation_result.justification,
        )

        return self._pending_function_response.result_data, memory

    def set_agent_system_message(self, new_message):
        self._agent_system_message = new_message

    def _update_prompts(self, user_input):
        # We want the code from the original function
        self._researcher.update_system_message(
            get_researcher_prompt(
                user_input, self._agent_system_message, self._original_function, self._wrapped_function
            )
        )

        self._result_validator.update_system_message(
            get_result_validator_prompt(user_input, self._validator_system_message)
        )


class SubSwarmFunctionWrapper:
    def __init__(self, tool_function, function_response_callback, results_validator):
        self.tool_function = tool_function
        self.function_response_callback = function_response_callback
        self.results_validator = results_validator
        self.num_invocations = 0

    def __call__(self, *args, hypothesis="", context_variables={}, **kwargs) -> SwarmResult:
        self.num_invocations += 1
        try:
            # Call the tool function
            if iscoroutinefunction(self.tool_function):
                # Await the async tool function
                result_data, result_str = asyncio.run( self.tool_function(context_variables=context_variables, *args, **kwargs))
            else:
                # Call the sync tool function directly
                result_data, result_str = self.tool_function(context_variables=context_variables, *args, **kwargs)
            function_response: FunctionResponse = FunctionResponse(
                function=self.tool_function,
                hypothesis=hypothesis,
                args=args,
                kwargs=kwargs,
                result_data=result_data,
                result_str=result_str,
            )
            response_str = self.function_response_callback(function_response)
            if "terminate" in response_str.lower():
                return SwarmResult(
                    context_variables=context_variables,
                    agent=AfterWorkOption.TERMINATE,
                    values=response_str,
                )
            # When the function returns a SwarmResult, then expect that this function call terminates the research and the LearningAgent passes to the result.
            return SwarmResult(
                context_variables=context_variables,
                agent=self.results_validator,
                values=response_str,
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

        # TODO: Re-add context variable passthrough
        # context_variables_param = Parameter(
        #     "context_variables",
        #     kind=Parameter.POSITIONAL_OR_KEYWORD,
        #     annotation=dict,
        # )

        hypothesis_param = Parameter(
            "hypothesis",
            kind=Parameter.POSITIONAL_OR_KEYWORD,
            annotation=Annotated[str, AG2Field(description=HYPOTHESIS_DEF)],
        )

        # Insert the new parameters before args/kwargs
        params = [p for p in params if p.name not in ("context_variables", "hypothesis")] + [hypothesis_param]

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


def get_researcher_prompt(user_input, agent_system_message, function, wrapped_function_sig) -> str:
    prompt = f"""
Task: You are responsible for invoking the function provided to you based on the input from the user.
You can only invoke one function one time per response. Do not make a list of function calls.

Ensure that all string parameters which contain quotes are properly escaped.

{agent_system_message}

User Input:
{user_input}
"""
    return prompt


# TODO maybe need to fix and provide more detail here?
def get_result_validator_prompt(user_input, validator_system_message) -> str:
    return f"""
Task: You will be provided user_input and result. Perform the following validations on the result based on the user_input.

Ensure that all string parameters which contain quotes are properly escaped.

{validator_system_message}
"""
