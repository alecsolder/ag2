# -*- coding: utf-8 -*-
# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT

import asyncio
import copy
import functools
import inspect
import json
import logging
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from pydantic import BaseModel, Field, ValidationError, ConfigDict

from ....agentchat.agent import Agent
from ....agentchat.conversable_agent import ConversableAgent
from ....agentchat.group.context_variables import ContextVariables
from ....agentchat import initiate_group_chat
from ....agentchat.group.patterns import DefaultPattern
from ....agentchat.group import AgentTarget, ReplyResult
from ....doc_utils import export_module
from ....llm_config import LLMConfig
from ....tools.tool import Tool

# --- Setup Logger ---
logger = logging.getLogger(__name__)

# --- Pydantic Models for Context ---
class ValidationResult(BaseModel):
    """Represents the outcome of a single validation step."""

    model_config = ConfigDict(extra="forbid")

    validation_result: bool
    justification: str

    def __str__(self) -> str:
        status = "Passed" if self.validation_result else "Failed"
        return f"Validation Result: {status}\nJustification: {self.justification}"

    def format(self) -> str:
        """Returns the JSON representation for AutoGen compatibility."""
        return self.model_dump_json()


class ExecutionAttempt(BaseModel):
    """Stores the state of a single attempt to execute and validate the function."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    attempt_number: int
    timestamp: float = Field(default_factory=time.time)
    args: List[Any] = Field(default_factory=list)
    kwargs: Dict[str, Any] = Field(default_factory=dict)
    hypothesis: Optional[str] = None
    error: Optional[str] = None
    result_data: Optional[Any] = None
    result_str: Optional[str] = None
    validation: Optional[ValidationResult] = None

    @property
    def did_execute_successfully(self) -> bool:
        """Check if the attempt executed without raising an error."""
        return self.error is None

    @property
    def did_validate_successfully(self) -> bool:
        """Check if the attempt passed validation."""
        return self.validation is not None and self.validation.validation_result


class ReliableToolContext(BaseModel):
    """Main context object holding the overall state and history of attempts."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    task: str
    reliable_tool_name: str
    start_time: float = Field(default_factory=time.time)
    dynamic_validation_input: Optional[str] = None
    attempts: List[ExecutionAttempt] = Field(default_factory=list)

    @property
    def attempt_count(self) -> int:
        """Return the number of attempts made."""
        return len(self.attempts)

    @property
    def latest_attempt(self) -> Optional[ExecutionAttempt]:
        """Return the most recent attempt, if any."""
        return self.attempts[-1] if self.attempts else None

    @property
    def is_complete_and_successful(self) -> bool:
        """Check if the process finished with a validated successful attempt."""
        latest = self.latest_attempt
        return latest is not None and latest.did_execute_successfully and latest.did_validate_successfully

    def get_final_result_data(self) -> Any:
        """Return the result_data from the first successful and validated attempt."""
        for attempt in reversed(self.attempts):
            if attempt.did_execute_successfully and attempt.did_validate_successfully:
                return attempt.result_data
        return None

    def get_failure_summary(self) -> str:
        """Provide a summary of why the overall execution failed."""
        latest = self.latest_attempt
        if latest is None:
            return "No attempts were made."
        if not latest.did_execute_successfully:
            return f"Last attempt ({latest.attempt_number}) failed execution: {latest.error}"
        if not latest.did_validate_successfully:
            justification = latest.validation.justification if latest.validation else "validation result missing"
            return (
                f"Last attempt ({latest.attempt_number}) executed but failed validation (Justification: {justification})"
            )
        # Should not happen if is_complete_and_successful is False
        return "Execution completed but overall status indicates failure (Internal inconsistency)."


# --- Helper Functions ---
def _configure_llm_for_structured_output(
    llm_config: Optional[Union[LLMConfig, dict, bool]], structured_output_type: Type[BaseModel]
) -> Optional[Union[LLMConfig, dict, bool]]:
    """Configure LLM config for structured output using a Pydantic model.

    Args:
        llm_config: The LLM configuration to modify. Can be an LLMConfig object, a dictionary, or False.
        structured_output_type: The Pydantic model class for structured output.

    Returns:
        The modified LLM configuration, or None if input was None or False.

    Raises:
        TypeError: If llm_config is not LLMConfig, dict, or False, or if structured_output_type is not a Pydantic BaseModel.
        ValueError: If the llm_config dict structure is invalid or resolves to False unexpectedly.
    """
    if llm_config is None or llm_config is False:
        return llm_config
    if not issubclass(structured_output_type, BaseModel):
        raise TypeError(f"{structured_output_type} must be a Pydantic BaseModel subclass.")

    # Validate and get a mutable LLMConfig object (or False)
    llm_config_obj = ConversableAgent._validate_llm_config(llm_config)

    if llm_config_obj is False:
        # If the validated config is False, return False
        return False

    # Work on the LLMConfig object directly (it's effectively a copy due to _validate_llm_config)
    try:
        llm_config_obj.response_format = structured_output_type # type: ignore[attr-defined] # Pydantic V2 handles this via __setattr__
    except AttributeError:
        # Fallback for potential issues setting directly if not a standard LLMConfig structure
        # This assumes llm_config_obj has a way to set/get items like a dict
        if hasattr(llm_config_obj, "__setitem__"):
            llm_config_obj["response_format"] = structured_output_type
        else:
             logger.warning("Could not directly set 'response_format' on LLMConfig object.")


    # Remove potentially conflicting tool/function settings
    if hasattr(llm_config_obj, "tools") and getattr(llm_config_obj, "tools", None): # type: ignore[attr-defined]
        logger.debug(
            "Removing 'tools' from validator LLM config for structured output (response_format=%s)",
            structured_output_type.__name__,
        )
        setattr(llm_config_obj, 'tools', []) # Use setattr for safety
    if hasattr(llm_config_obj, "tool_choice") and getattr(llm_config_obj, "tool_choice", None): # type: ignore[attr-defined]
        logger.debug(
            "Removing 'tool_choice' from validator LLM config for structured output (response_format=%s)",
            structured_output_type.__name__,
        )
        setattr(llm_config_obj, 'tool_choice', None)
    if hasattr(llm_config_obj, "functions") and getattr(llm_config_obj, "functions", None): # type: ignore[attr-defined]
         logger.debug(
            "Removing 'functions' from validator LLM config for structured output (response_format=%s)",
            structured_output_type.__name__,
        )
         setattr(llm_config_obj, 'functions', [])

    # Also clean config_list items within the LLMConfig object
    config_list = getattr(llm_config_obj, 'config_list', None)
    if isinstance(config_list, list): # type: ignore[attr-defined]
        new_config_list = []
        for item in config_list: # type: ignore[attr-defined]
            # Modify the config entry dictionary directly if it's a dict
            if isinstance(item, dict):
                item_copy = item.copy() # Modify a copy
                item_copy["response_format"] = structured_output_type
                item_copy.pop("tools", None)
                item_copy.pop("tool_choice", None)
                item_copy.pop("functions", None)
                new_config_list.append(item_copy)
            # If item is an object (like LLMConfigEntry), modify its attributes if possible
            elif hasattr(item, 'response_format'):
                 item_copy = copy.deepcopy(item) # Modify a copy
                 setattr(item_copy, "response_format", structured_output_type)
                 if hasattr(item_copy, 'tools'): setattr(item_copy, "tools", [])
                 if hasattr(item_copy, 'tool_choice'): setattr(item_copy, "tool_choice", None)
                 if hasattr(item_copy, 'functions'): setattr(item_copy, "functions", [])
                 new_config_list.append(item_copy)
            else:
                 new_config_list.append(item) # Append unmodified if not dict or expected object
        setattr(llm_config_obj, 'config_list', new_config_list)


    logger.debug("Prepared LLM config for validator (response_format=%s)", structured_output_type.__name__)
    return llm_config_obj


def _get_last_non_empty_message_content(messages: Optional[List[Dict]]) -> Optional[str]:
    """Get content of the last message with non-empty content."""
    if not messages:
        return None
    for message in reversed(messages):
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
        # Handle potential multimodal content (take first element as representative string)
        if isinstance(content, list) and content:
            try:
                first_item = content[0]
                if isinstance(first_item, dict) and "text" in first_item:
                    return str(first_item["text"]).strip()
                elif isinstance(first_item, dict):
                    # Fallback for other dict structures
                    return json.dumps(first_item)
                else:
                    return str(first_item).strip()
            except Exception:
                # Ignore errors converting complex content to string
                pass
    return None


def _get_reliable_tool_context(context_variables: ContextVariables, context_key: str) -> ReliableToolContext:
    """Retrieve and validate the ReliableToolContext from ContextVariables.

    Args:
        context_variables: The ContextVariables instance.
        context_key: The key for the ReliableToolContext.

    Returns:
        The validated ReliableToolContext instance.

    Raises:
        KeyError: If the context_key is not found.
        ValueError: If the data associated with the key cannot be loaded into ReliableToolContext.
    """
    context_data = context_variables.get(context_key)
    if context_data is None:
        raise KeyError(f"ReliableToolContext key '{context_key}' not found.")
    try:
        if isinstance(context_data, ReliableToolContext):
            return context_data
        if isinstance(context_data, str):
            return ReliableToolContext.model_validate_json(context_data)
        if isinstance(context_data, dict):
            return ReliableToolContext.model_validate(context_data)

        raise TypeError(f"Unexpected type {type(context_data)} for context key '{context_key}'.")
    except (ValidationError, json.JSONDecodeError, TypeError) as e:
        preview = f" Preview: '{str(context_data)[:100]}...'" if isinstance(context_data, (str, dict)) else ""
        logger.error(
            "Failed loading ReliableToolContext '%s'. Error: %s. Type: %s.%s",
            context_key,
            e,
            type(context_data).__name__,
            preview,
            exc_info=True,
        )
        raise ValueError(f"Failed loading ReliableToolContext key '{context_key}': {e}") from e


def _set_reliable_tool_context(
    context_variables: ContextVariables, context_key: str, context: ReliableToolContext
) -> None:
    """Serialize and store the ReliableToolContext in ContextVariables.

    Args:
        context_variables: The ContextVariables instance.
        context_key: The key for the ReliableToolContext.
        context: The ReliableToolContext instance to store.

    Raises:
        TypeError: If context is not a ReliableToolContext instance.
        ValueError: If serialization fails.
    """
    if not isinstance(context, ReliableToolContext):
        raise TypeError(f"Object to set must be a ReliableToolContext, got {type(context)}.")
    try:
        # Store as JSON string for broader compatibility
        context_variables[context_key] = context.model_dump_json(warnings="warn")
    except Exception as e:
        context_dict_str = "N/A"
        try:
            context_dict_str = str(context.model_dump(warnings="warn", exclude_none=True))[:500]
        except Exception:
            pass
        logger.critical(
            "Failed serializing ReliableToolContext key '%s': %s. Context: %s",
            context_key,
            e,
            context_dict_str,
            exc_info=True,
        )
        raise ValueError(f"Critical error serializing ReliableToolContext: {e}") from e


def get_runner_prompt(
    task: str, agent_system_message: str, current_attempt_number: int, internal_tool_name: str
) -> str:
    """Generate the system prompt for the internal runner agent."""
    return f"""
You are an AI assistant responsible for invoking a specific function based on the user's task.
Function to call: '{internal_tool_name}'
This is attempt number {current_attempt_number}. Ensure arguments are appropriate for this attempt number (e.g., for retry logic).

You MUST invoke the function '{internal_tool_name}' exactly one time per response using a tool call format that the system can execute.
Do NOT just output text explaining what you would do, or asking for confirmation. Directly make the tool call.
Analyze the task description and conversation history carefully to determine the correct arguments for the function call.
You MUST provide a 'hypothesis' argument summarizing the expected outcome or result format of the function call based on the inputs.

Base Instructions:
{agent_system_message}

Current Task:
{task}
"""


def get_validator_prompt(
    task: str, base_validator_system_message: str, dynamic_validation_addition: Optional[str] = None
) -> str:
    """Generate the system prompt for the internal validator agent."""
    dynamic_section = (
        f"\n\nAdditional Requirements for This Specific Run:\n{dynamic_validation_addition.strip()}"
        if dynamic_validation_addition and dynamic_validation_addition.strip()
        else ""
    )
    return f"""
You are an AI validation assistant. You will receive the result (or error message) of a function call intended to accomplish a task.
Your goal is to validate if the function call result meets ALL requirements: the base task description, the base validation rules, and any additional requirements specified below.

Task Description:
{task}

Base Validation Rules/Context:
{base_validator_system_message}{dynamic_section}

Evaluate the function call result/error based on the information provided in the user message and all the requirements listed above.
Your output MUST strictly conform to the required JSON format for the ValidationResult model. Respond ONLY with the JSON object.
"""
# --- Reliable Function Wrapper (Conditionally Sync/Async) ---
def reliable_function_wrapper(tool_function: Callable, validator: ConversableAgent, context_variables_key: str):
    """Wraps the target function, returning a sync or async wrapper."""
    is_original_async = inspect.iscoroutinefunction(tool_function)
    tool_sig = inspect.signature(tool_function)

    # --- Internal Helper Functions for the Wrapper ---
    def _prepare_call_args(args: tuple, kwargs: dict, context_vars: ContextVariables) -> Tuple[tuple, dict]:
        """Prepares args/kwargs, injecting context_variables if needed."""
        call_args, call_kwargs = args, kwargs.copy()
        param = tool_sig.parameters.get("context_variables")
        if param and (param.annotation is ContextVariables or param.annotation is inspect.Parameter.empty or param.annotation is Any):
            call_kwargs["context_variables"] = context_vars
        return call_args, call_kwargs

    def _process_result(attempt: ExecutionAttempt, context_vars: ContextVariables, context: ReliableToolContext, result: Any) -> ReplyResult:
        """Shared logic to process successful execution result."""
        res_data, res_str = None, ""
        if isinstance(result, tuple) and len(result) == 2:
            res_data, repr_str = result
            res_str = str(repr_str) if not isinstance(repr_str, str) else repr_str
        elif isinstance(result, ReplyResult):
            logger.warning("Original function '%s' returned ReplyResult directly.", tool_function.__name__)
            res_data, res_str = result.message, str(result.message) # Assuming message holds result
        else:
            res_data = result
            try: res_str = str(res_data)
            except Exception as str_e: logger.warning("Could not convert result type %s to string: %s. Using repr().", type(res_data).__name__, str_e); res_str = repr(res_data)

        attempt.result_data, attempt.result_str = res_data, res_str
        context.attempts.append(attempt)
        try: _set_reliable_tool_context(context_vars, context_variables_key, context)
        except Exception as e: logger.error("Failed saving context after success: %s", e, exc_info=True)
        # Send result string to validator
        return ReplyResult(context_variables=context_vars, target=AgentTarget(validator), message=res_str or "")

    def _handle_error(attempt: ExecutionAttempt, context_vars: ContextVariables, context: ReliableToolContext, e: Exception) -> ReplyResult:
        """Shared logic to handle execution error."""
        err_msg = f"{type(e).__name__}: {e}"
        logger.error("Function execution error (Attempt %d): %s", attempt.attempt_number, err_msg, exc_info=True) # Add traceback here
        attempt.error = err_msg
        # Provide a concise error message for the validator agent, avoiding full traceback in chat
        err_msg_validator = f"Function execution failed on attempt {attempt.attempt_number} with error: {err_msg}."
        context.attempts.append(attempt)
        try: _set_reliable_tool_context(context_vars, context_variables_key, context)
        except Exception as e_ctx: logger.error("Failed saving context after error: %s", e_ctx, exc_info=True)
        return ReplyResult(context_variables=context_vars, target=AgentTarget(validator), message=err_msg_validator)

    # --- Define and return the appropriate wrapper ---
    if not is_original_async:
        @functools.wraps(tool_function)
        def sync_wrapper(*args, hypothesis: str, context_variables: ContextVariables, **kwargs) -> ReplyResult:
            context = _get_reliable_tool_context(context_variables, context_variables_key)
            attempt = ExecutionAttempt(attempt_number=context.attempt_count + 1, args=list(args), kwargs=kwargs, hypothesis=hypothesis)
            try:
                call_args, call_kwargs = _prepare_call_args(args, kwargs, context_variables)
                # MAKE SURE THE TYPES MATCH AND COMPLAINIF THEY"RE WRONG
                result = tool_function(*call_args, **call_kwargs)
                return _process_result(attempt, context_variables, context, result)
            except Exception as e: return _handle_error(attempt, context_variables, context, e)
        wrapper_func = sync_wrapper
    else:
        @functools.wraps(tool_function)
        async def async_wrapper(*args, hypothesis: str, context_variables: ContextVariables, **kwargs) -> ReplyResult:
            context = _get_reliable_tool_context(context_variables, context_variables_key)
            attempt = ExecutionAttempt(attempt_number=context.attempt_count + 1, args=list(args), kwargs=kwargs, hypothesis=hypothesis)
            try:
                call_args, call_kwargs = _prepare_call_args(args, kwargs, context_variables)
                result = await tool_function(*call_args, **call_kwargs)
                return _process_result(attempt, context_variables, context, result)
            except Exception as e: return _handle_error(attempt, context_variables, context, e)
        wrapper_func = async_wrapper

    # Adapt Signature for the chosen wrapper
    params = list(tool_sig.parameters.values())

    # Separate original params based on kind
    pos_or_kw_params = []
    kw_only_params = []
    var_pos_param = None # *args
    var_kw_param = None  # **kwargs

    for p in params:
        # Exclude our wrapper params if they happen to exist in original (unlikely but safe)
        if p.name in ("hypothesis", "context_variables"):
            continue

        if p.kind == inspect.Parameter.POSITIONAL_ONLY or p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
            pos_or_kw_params.append(p)
        elif p.kind == inspect.Parameter.VAR_POSITIONAL:
            var_pos_param = p
        elif p.kind == inspect.Parameter.KEYWORD_ONLY:
            kw_only_params.append(p)
        elif p.kind == inspect.Parameter.VAR_KEYWORD:
            var_kw_param = p

    # Define our new keyword-only parameters
    new_kw_only_params = [
        inspect.Parameter("hypothesis", inspect.Parameter.KEYWORD_ONLY, annotation=str),
        inspect.Parameter("context_variables", inspect.Parameter.KEYWORD_ONLY, annotation=ContextVariables)
    ]

    # Construct the final parameter list in the correct order
    wrapper_params = (
        pos_or_kw_params +
        ([var_pos_param] if var_pos_param else []) + # Add *args if it existed
        kw_only_params + new_kw_only_params + # Add original keyword-only, then our new ones
        ([var_kw_param] if var_kw_param else []) # Add **kwargs if it existed
    )

    wrapper_func.__signature__ = inspect.Signature(parameters=wrapper_params, return_annotation=ReplyResult)
    return wrapper_func

# (The rest of the file remains the same)

# --- Crucial Change needed in _setup_runner_agent ---
# Even with the above wrapper, you MUST still use the "Manual Schema" approach
# in _setup_runner_agent to avoid the schema generation error during initialization.
# Replace the internal_tool.register_tool(runner) call with the manual steps:
# 1. runner.update_tool_signature(llm_visible_schema, ...)
# 2. runner.register_function({internal_tool_name: self._reliable_func_wrapper})

# --- ReliableTool Class ---
@export_module("autogen.tools.experimental")
class ReliableToolError(Exception):
    """Custom exception for errors during ReliableTool execution."""

    def __init__(self, message: str, final_context: Optional[ReliableToolContext] = None):
        super().__init__(message)
        self.final_context = final_context


@export_module("autogen.tools.experimental")
class ReliableTool(Tool):
    """Wraps a function/tool for reliable execution via internal validation/retry.

    This tool uses an internal group chat with two agents (Runner and Validator)
    to attempt executing a target function, validate its output against criteria,
    and potentially retry upon failure, aiming for a validated successful execution.

    Attributes:
        name (str): The public name of this reliable tool.
        description (str): The public description of this tool.
        max_retries (int): Maximum number of retries allowed after the initial attempt.
    """

    INTERNAL_TOOL_NAME_PREFIX = "execute_"

    def __init__(
        self,
        name: str,
        func_or_tool: Union[Callable, Tool],
        runner_llm_config: Union[LLMConfig, dict, bool],
        validator_llm_config: Union[LLMConfig, dict, bool],
        description: Optional[str] = None,
        runner_system_message: str = "",
        validator_system_message: str = "",
        max_retries: int = 3,
        agent_kwargs: Optional[dict] = None,
        enable_dynamic_validation: bool = False,
    ) -> None:
        """Initializes the ReliableTool instance.

        Args:
            name: The public name for this ReliableTool.
            func_or_tool: The function or autogen.Tool instance to wrap.
            runner_llm_config: LLM configuration for the internal Runner agent. Can be LLMConfig, dict, or False.
            validator_llm_config: LLM configuration for the internal Validator agent. Must include a model. Can be LLMConfig or dict.
            description: Optional public description for the ReliableTool. Defaults to a generated description.
            runner_system_message: Base system message for the Runner agent.
            validator_system_message: Base system message for the Validator agent (validation rules).
            max_retries: Maximum number of retries allowed (default: 3). Total attempts = 1 + max_retries.
            agent_kwargs: Optional shared keyword arguments for creating internal agents (e.g., common LLM settings like temperature).
            enable_dynamic_validation: If True, allows passing 'validation_prompt_addition' during run/a_run.
        """
        self._original_func, original_name, original_description = self._extract_func_details(func_or_tool)
        self._is_original_func_async = inspect.iscoroutinefunction(self._original_func)

        # Initialize the base Tool class with the *original* function details
        # The reliability mechanism is handled internally by this class's run/a_run
        super().__init__(
            name=name,
            description=description or f"Reliably executes '{original_name}': {original_description}",
            func_or_tool=self._original_func,  # Base class needs the callable
        )

        # Validate and store configurations using the helper from ConversableAgent
        self._runner_llm_config = ConversableAgent._validate_llm_config(runner_llm_config)
        self._validator_llm_config = ConversableAgent._validate_llm_config(validator_llm_config)

        if self._validator_llm_config is False: # Explicit check for False
            raise ValueError("Validator agent requires a valid LLM configuration (LLMConfig or dict), but received False.")

        # Check if validator LLM config has a model specified (necessary for structured output)
        validator_model_specified = False
        if isinstance(self._validator_llm_config, LLMConfig):
            # Check model attribute directly on LLMConfig and within config_list entries
            validator_model_specified = getattr(self._validator_llm_config, 'model', None) or any(
                 getattr(cfg, 'model', None) for cfg in getattr(self._validator_llm_config, 'config_list', [])
            )

        if not validator_model_specified:
             raise ValueError("Validator LLM configuration must specify a 'model' for structured output.")


        self._runner_system_message = runner_system_message
        self._validator_system_message = validator_system_message
        self.max_retries = max_retries
        self._context_variables_key = f"{self.name}_ReliableToolContext"
        self._agent_kwargs = copy.deepcopy(agent_kwargs) if agent_kwargs is not None else {}
        self._original_func_name = original_name
        self._enable_dynamic_validation = enable_dynamic_validation

        # Setup internal components
        self._validator_name = f"{self.name}_Validator"
        self._runner_name = f"{self.name}_Runner"
        self._validator = self._setup_validator_agent()
        self._reliable_func_wrapper = reliable_function_wrapper(
            self._original_func, self._validator, self._context_variables_key
        )
        self._runner = self._setup_runner_agent()
        self._register_internal_hooks()

    def _extract_func_details(self, func_or_tool: Union[Callable, Tool]) -> Tuple[Callable, str, str]:
        """Extract the core function, its name, and description from input."""
        if isinstance(func_or_tool, Tool):
            if not callable(func_or_tool.func):
                raise TypeError(f"Tool '{func_or_tool.name}' func attribute is not callable.")
            return func_or_tool.func, func_or_tool.name, func_or_tool.description
        if callable(func_or_tool):
            name = getattr(func_or_tool, "__name__", "callable_function")
            doc = inspect.getdoc(func_or_tool)  # Use inspect.getdoc for cleaner docstrings
            desc = doc.strip() if doc else f"Callable function '{name}'."
            return func_or_tool, name, desc
        raise TypeError("Input 'func_or_tool' must be a callable or an autogen.Tool instance.")

    def _setup_validator_agent(self) -> ConversableAgent:
        """Create and configure the internal validator agent."""
        try:
            # Configure for structured output (ValidationResult)
            structured_llm_config = _configure_llm_for_structured_output(
                self._validator_llm_config, ValidationResult
            )
            if structured_llm_config is False: # Check explicitly for False after configuration attempt
                 raise ValueError("Validator LLM config resolved to False after attempting structured output configuration.")

        except (TypeError, ValueError) as e:
            raise ValueError(f"Failed to configure validator LLM for structured output: {e}") from e

        # Merge shared kwargs, allowing specific llm_config to override
        merged_kwargs = {**self._agent_kwargs, "llm_config": structured_llm_config}

        return ConversableAgent(
            name=self._validator_name,
            system_message="[Validator Prompt Updated Per Run]",  # Placeholder, updated in _execute_internal_group_chat
            human_input_mode="NEVER",
            **merged_kwargs,
        )

    def _setup_runner_agent(self) -> ConversableAgent:
        """Create and configure the internal runner agent."""
        # Deepcopy the config to avoid modifying the original passed to init
        runner_llm_config_copy = copy.deepcopy(self._runner_llm_config)

        # Merge shared kwargs, allowing specific llm_config to override
        merged_kwargs = {**self._agent_kwargs, "llm_config": runner_llm_config_copy}

        runner = ConversableAgent(
            name=self._runner_name,
            system_message="[Runner Prompt Updated Per Run]",  # Placeholder, updated in _execute_internal_group_chat
            human_input_mode="NEVER",
            **merged_kwargs,
        )

        # Setup and register the internal tool that wraps the original function
        internal_tool_name = f"{self.INTERNAL_TOOL_NAME_PREFIX}{self._original_func_name}"
        internal_tool_desc = (
            f"Executes the core logic for task: {self._original_func_name}. " f"Must provide 'hypothesis'."
        )
        # Create the internal tool using the reliable wrapper
        internal_tool = Tool(
            name=internal_tool_name, description=internal_tool_desc, func_or_tool=self._reliable_func_wrapper
        )

        # Register the tool for both LLM proposal and execution with the runner agent
        # This handles updating the runner's llm_config correctly.
        try:
            internal_tool.register_tool(runner)
        except TypeError as e:
            logger.error(f"Failed to register internal tool '{internal_tool_name}' for runner '{runner.name}': {e}")
            logger.error(f"Internal tool function signature: {getattr(internal_tool.func, '__signature__', 'N/A')}")
            raise TypeError(f"Error registering internal tool for runner: {e}. Check wrapper signature generation.") from e


        # Remove response_format if present *after* tool registration,
        # as runner needs tool calls, not structured responses.
        if runner.llm_config and runner.llm_config is not False:
             # Use getattr with default None for safety
             current_response_format = getattr(runner.llm_config, 'response_format', None)
             if current_response_format:
                 logger.debug("Removing 'response_format' from runner LLM config to allow tool calls.")
                 try:
                     setattr(runner.llm_config, 'response_format', None) # Use setattr
                 except AttributeError:
                     logger.warning("Could not remove 'response_format' attribute from runner LLM config object.")

             # Also check within config_list items
             config_list = getattr(runner.llm_config, 'config_list', None)
             if isinstance(config_list, list):
                 new_config_list = []
                 for item in config_list:
                     if isinstance(item, dict) and "response_format" in item:
                         item_copy = item.copy()
                         item_copy.pop("response_format", None)
                         new_config_list.append(item_copy)
                     elif hasattr(item, "response_format"):
                         item_copy = copy.deepcopy(item)
                         setattr(item_copy, "response_format", None)
                         new_config_list.append(item_copy)
                     else:
                         new_config_list.append(item) # Keep unmodified item
                 try:
                     setattr(runner.llm_config, 'config_list', new_config_list)
                 except AttributeError:
                      logger.warning("Could not update 'config_list' attribute on runner LLM config object.")


        return runner

    def _register_internal_hooks(self) -> None:
        """Register hooks for internal group chat interaction."""
        # Hook called before the validator sends its message
        self._validator.register_hook(
            hookable_method="process_message_before_send", hook=self._validator_structured_output_hook
        )
        # Hook called before the validator generates a reply (to filter history)
        self._validator.register_hook(
            hookable_method="process_all_messages_before_reply", hook=self._validator_history_filter_hook
        )
        # Hook called before the runner sends its message (to encourage tool calls)
        self._runner.register_hook(hookable_method="process_message_before_send", hook=self._ensure_function_call_hook)

    # --- Hook Implementations ---
    def _validator_structured_output_hook(
        self, sender: Agent, message: Union[dict, str], recipient: Agent, silent: bool
    ) -> Union[dict, str]:
        """Process validator LLM output, update context, return formatted JSON string.

        This hook intercepts the message the validator *intends* to send. It expects
        the LLM's raw output (ideally JSON matching ValidationResult). It attempts
        to parse this, updates the ReliableToolContext with the validation outcome,
        and then replaces the outgoing message content with the *formatted* JSON
        string representation of the ValidationResult. If parsing fails, it logs
        an error and returns an error message string.
        """
        # Only process messages sent *by* the validator
        if sender.name != self._validator_name:
            return message

        # Extract text content robustly
        message_text = None
        if isinstance(message, str):
            message_text = message.strip()
        elif isinstance(message, dict):
            content = message.get("content")
            try:
                if isinstance(content, str):
                    message_text = content.strip()
                elif content is not None:
                    # Attempt to get string representation, careful with complex types
                    message_text = str(content).strip()
            except Exception:
                logger.warning(
                    "Validator hook: Could not reliably stringify content of type %s", type(content), exc_info=True
                )

        if not message_text:
            logger.warning("Validator hook: Received empty or invalid content from validator LLM.")
            # Return a clear error message for the chat flow
            return "[Validator produced empty output]"

        # Attempt parsing, validation logging
        validation_result_obj: Optional[ValidationResult] = None
        parsed_successfully = False
        try:
            # Assume LLM output is JSON string matching ValidationResult
            validation_result_obj = ValidationResult.model_validate_json(message_text)
            parsed_successfully = True
            status = "PASSED" if validation_result_obj.validation_result else "FAILED"
            log_level = logging.INFO if status == "PASSED" else logging.WARNING
            logger.log(log_level, f"Validator Hook: {status}. Justification: {validation_result_obj.justification}")
        except (ValidationError, json.JSONDecodeError) as e:
            # Log validation/parsing errors specifically
            logger.warning(
                f"Validator Hook: Failed to parse/validate LLM output ({type(e).__name__}): {e}. "
                f"Raw text: ```\n{message_text}\n```"
            )
            validation_result_obj = ValidationResult(
                validation_result=False, justification=f"Validation output parsing failed ({type(e).__name__})"
            )
        except Exception as e:
            # Log unexpected errors more severely
            logger.error(
                f"Validator Hook: Unexpected error processing LLM output ({type(e).__name__}): {e}. "
                f"Raw text: ```\n{message_text}\n```",
                exc_info=True,
            )
            validation_result_obj = ValidationResult(
                validation_result=False, justification=f"Unexpected error in validation hook ({type(e).__name__})"
            )

        # Attempt context update regardless of parsing success (to record the attempt)
        if validation_result_obj:
            self._try_update_context_validation(sender, validation_result_obj)

        # Return formatted JSON message string for the chat, or error message
        if parsed_successfully and validation_result_obj:
            # Use the model's format() method which returns JSON string
            return validation_result_obj.format()
        else:
            # Return a user-friendly error message within the chat flow
            err_just = validation_result_obj.justification if validation_result_obj else "Parsing/Validation Failed"
            # Combine error justification with raw output for debugging within chat history
            return f"Error processing validation result: {err_just}. Raw Output:\n```json\n{message_text}\n```"

    def _try_update_context_validation(self, sender: Agent, validation_result: ValidationResult) -> None:
        """Helper to attempt updating the validation state in the ReliableToolContext."""
        try:
            # Access context variables stored on the agent instance
            # (assuming DefaultPattern or similar mechanism sets it)
            context_vars = getattr(sender, "context_variables", None)
            if not isinstance(context_vars, ContextVariables):
                logger.error("Validator hook: Cannot access valid ContextVariables on sender '%s'.", sender.name)
                return

            tool_context = _get_reliable_tool_context(context_vars, self._context_variables_key)
            latest_attempt = tool_context.latest_attempt
            if latest_attempt and latest_attempt.validation is None:
                latest_attempt.validation = validation_result
                _set_reliable_tool_context(context_vars, self._context_variables_key, tool_context)
                logger.info(
                    "Validator hook: Updated attempt %d validation status: %s",
                    latest_attempt.attempt_number,
                    "Passed" if validation_result.validation_result else "Failed",
                )
            elif latest_attempt:
                logger.warning(
                    "Validator hook: Attempt %d validation already set, not overwriting.",
                    latest_attempt.attempt_number,
                )
            else:
                logger.warning("Validator hook: No attempts found in context to update.")
        except (KeyError, ValueError, TypeError) as e_get_ctx:
            logger.error(
                "Validator hook: Failed to get/update ReliableToolContext: %s", e_get_ctx, exc_info=True
            )  # Log context access errors
        except Exception as e_ctx:
            logger.error("Validator hook: Unexpected error during context update: %s", e_ctx, exc_info=True)

    def _validator_history_filter_hook(self, messages: List[Dict], **kwargs: Any) -> List[Dict]:
        """Filter message history for the validator agent.

        Provides only the system message and the last message content (result/error)
        to the validator LLM for focused validation.
        """
        if not messages:
            return []

        # Preserve the original system message if it exists
        system_message = messages[0] if messages and messages[0].get("role") == "system" else None

        # Get the content of the last message (which should be the output from the runner/tool)
        last_content = _get_last_non_empty_message_content(messages) or "[No function result content found]"

        # Construct the filtered history for the validator
        filtered_messages = []
        if system_message:
            filtered_messages.append(system_message)
        filtered_messages.append(
            {"role": "user", "content": f"Function Result/Error to Validate:\n```\n{last_content}\n```"}
        )

        return filtered_messages

    def _ensure_function_call_hook(
        self, sender: Agent, message: Union[dict, str], recipient: Agent, silent: bool
    ) -> Union[dict, str]:
        """Hook to encourage the runner LLM to make a tool call if it hasn't.

        Checks if the message from the runner agent contains a tool call. If not,
        and if the content doesn't seem like an error related to tool calling,
        it appends a reminder to the message content.
        """
        # Only process messages sent *by* the runner
        if sender.name != self._runner_name:
            return message

        is_tool_call = isinstance(message, dict) and message.get("tool_calls")
        content_str = ""
        if isinstance(message, dict):
            content_str = message.get("content", "") or ""
        elif isinstance(message, str):
            content_str = message

        tool_name_expected = f"{self.INTERNAL_TOOL_NAME_PREFIX}{self._original_func_name}"

        # Check conditions for adding reminder: no tool call, and content isn't clearly about tool call failure
        needs_reminder = not is_tool_call and not (
            tool_name_expected in content_str
            or ("error" in content_str.lower() and ("tool" in content_str.lower() or "function" in content_str.lower()))
        )

        if needs_reminder:
            logger.warning("Runner '%s' did not generate a tool call. Appending reminder.", self._runner_name)
            reminder = f"\n\n[[System Reminder: You MUST invoke the function '{tool_name_expected}' using a tool call. Provide the required 'hypothesis' argument.]]"
            if isinstance(message, dict):
                # Safely append to existing content or set new content
                message["content"] = (message.get("content") or "") + reminder
            else:
                # Ensure message is treated as string before appending
                message = str(message) + reminder
        return message

    # --- Core Execution Logic ---
    def _execute_internal_group_chat(
        self, task: str, initial_context: ContextVariables, dynamic_validation_str: Optional[str] = None
    ) -> Tuple[Optional[Union[Dict, ReplyResult]], ContextVariables, Optional[Agent]]:
        """Set up and run the internal synchronous group chat between Runner and Validator.

        Args:
            task: The task description.
            initial_context: The starting ContextVariables, containing the initialized ReliableToolContext.
            dynamic_validation_str: Optional additional validation criteria for this specific run.

        Returns:
            A tuple containing the last reply message, the final ContextVariables, and the last agent.

        Raises:
            ReliableToolError: If initial context loading or agent setup fails.
        """
        try:
            current_tool_context = _get_reliable_tool_context(initial_context, self._context_variables_key)
        except (KeyError, ValueError) as e:
            logger.critical("Cannot start group chat: Failed loading initial ReliableToolContext: %s", e)
            raise ReliableToolError(f"Failed loading initial context: {e}") from e

        next_attempt_number = current_tool_context.attempt_count + 1
        internal_tool_name = f"{self.INTERNAL_TOOL_NAME_PREFIX}{self._original_func_name}"
        max_attempts = self.max_retries + 1

        # Update Agent Prompts dynamically for this run
        try:
            runner_prompt = get_runner_prompt(
                task, self._runner_system_message, next_attempt_number, internal_tool_name
            )
            self._runner.update_system_message(runner_prompt)

            validator_prompt = get_validator_prompt(
                task, self._validator_system_message, dynamic_validation_str
            )
            self._validator.update_system_message(validator_prompt)
        except Exception as e:
            logger.critical("Failed setting up agent prompts for ReliableTool chat: %s", e, exc_info=True)
            raise ReliableToolError(f"Failed setting up agent prompts: {e}") from e

        # Prepare and Run Group Chat using DefaultPattern
        initial_messages = [{"role": "user", "content": f"Start Reliable Task Execution: {task}"}]
        # Estimate max rounds: each attempt is ~2 turns (runner -> validator) + buffer
        max_group_chat_rounds = max_attempts * 2 + 4
        agent_pattern = DefaultPattern(
            agents=[self._runner, self._validator], initial_agent=self._runner, context_variables=initial_context
        )

        logger.info(
            "--- Starting ReliableTool '%s' Internal Chat (Attempt %d / %d Max) ---",
            self.name,
            next_attempt_number,
            max_attempts,
        )
        try:
            # Use autogen's initiate_group_chat
            last_reply, final_context, last_agent = initiate_group_chat(
                pattern=agent_pattern,
                messages=initial_messages,
                max_rounds=max_group_chat_rounds,
            )
            logger.info("--- ReliableTool '%s' Internal Chat Finished ---", self.name)
            return last_reply, final_context, last_agent
        except Exception as e:
            # Catch potential errors within initiate_group_chat itself
            logger.critical(
                "CRITICAL ERROR during internal group chat for ReliableTool '%s': %s", self.name, e, exc_info=True
            )
            # Return initial context on critical failure to allow inspection of state before crash
            return None, initial_context, None

    # --- Main Execution Processing ---
    def _process_run(
        self,
        task: str,
        context_variables: Optional[ContextVariables] = None,
        validation_prompt_addition: Optional[str] = None,
    ) -> Any:
        """Internal synchronous logic shared by run() and a_run().

        Handles context initialization, running the internal group chat loop,
        and processing the final result or raising an error.

        Args:
            task: The task description.
            context_variables: Optional existing ContextVariables. If None, a new one is created.
            validation_prompt_addition: Optional dynamic validation criteria.

        Returns:
            The final validated result data from the wrapped function.

        Raises:
            ReliableToolError: If the process fails after all retries or due to setup errors.
            TypeError: If context_variables is not of the expected type.
        """
        # 1. Initialize Context
        current_context_variables = context_variables if context_variables is not None else ContextVariables()
        if not isinstance(current_context_variables, ContextVariables):
            raise TypeError(
                f"Expected context_variables as autogen.agentchat.group.ContextVariables or None, "
                f"got {type(context_variables)}"
            )

        dynamic_criteria = validation_prompt_addition if self._enable_dynamic_validation else None
        if validation_prompt_addition and not self._enable_dynamic_validation:
            warnings.warn(
                f"[{self.name}]: 'validation_prompt_addition' provided but dynamic validation is disabled. Ignoring.",
                UserWarning,
            )

        # Initialize the tool-specific context within the broader context variables
        initial_tool_context = ReliableToolContext(
            task=task, reliable_tool_name=self.name, dynamic_validation_input=dynamic_criteria
        )
        try:
            _set_reliable_tool_context(current_context_variables, self._context_variables_key, initial_tool_context)
        except ValueError as e_set:
            # If setting initial context fails, it's a critical setup error
            raise ReliableToolError(
                f"Failed to initialize ReliableToolContext: {e_set}", final_context=initial_tool_context
            ) from e_set

        # 2. Execute the Internal Group Chat Loop
        final_context_variables = current_context_variables  # Start with initial
        final_tool_context = initial_tool_context

        while True: # Loop controlled by context state
            try:
                _, chat_context_variables, _ = self._execute_internal_group_chat(
                    task=task, initial_context=final_context_variables, dynamic_validation_str=dynamic_criteria
                )

                # Ensure final_context_variables is valid even if chat returns None on error
                if chat_context_variables is None:
                    logger.warning(
                        "[%s]: Internal group chat returned None for context variables, likely due to a critical error. "
                        "Attempting to use previous context state.",
                        self.name
                    )
                    # Use the context from *before* the failed chat attempt
                else:
                     final_context_variables = chat_context_variables

                # Reload the tool context after the chat
                final_tool_context = _get_reliable_tool_context(final_context_variables, self._context_variables_key)

            except ReliableToolError as e_chat:
                # Errors during chat setup are critical failures
                logger.error("[%s] Failed during internal chat setup/execution: %s", self.name, e_chat)
                raise e_chat  # Re-raise to signal failure
            except (KeyError, ValueError) as e_get_ctx:
                 # If context retrieval fails *after* a chat, something is seriously wrong
                 logger.critical("[%s] Failed to retrieve ReliableToolContext after chat: %s", self.name, e_get_ctx)
                 raise ReliableToolError(f"Critical error: Could not retrieve final context post-chat: {e_get_ctx}", final_context=final_tool_context) from e_get_ctx

            # 3. Check State and Decide Action (Break or Retry)
            latest_attempt = final_tool_context.latest_attempt

            # Check if validation state is missing (e.g., validator hook failed critically)
            if latest_attempt and latest_attempt.validation is None:
                logger.warning(
                    "[%s]: Validation state missing for attempt %d after chat. Assuming failure for this attempt.",
                    self.name,
                    latest_attempt.attempt_number,
                )
                latest_attempt.validation = ValidationResult(
                    validation_result=False, justification="Validation state missing post-chat; assumed failure."
                )
                try:
                    # Try to save the assumed failure state
                    _set_reliable_tool_context(
                        final_context_variables, self._context_variables_key, final_tool_context
                    )
                except Exception as e_ctx_save:
                    logger.error("[%s]: Failed saving assumed failure state: %s", self.name, e_ctx_save)

            # Check for overall success
            if final_tool_context.is_complete_and_successful:
                logger.info(
                    "ReliableTool '%s' succeeded after %d attempt(s).", self.name, final_tool_context.attempt_count
                )
                return final_tool_context.get_final_result_data()

            # Check if max retries reached
            if final_tool_context.attempt_count > self.max_retries:
                 failure_summary = final_tool_context.get_failure_summary()
                 error_message = (
                     f"ReliableTool '{self.name}' failed after {final_tool_context.attempt_count} attempts "
                     f"(max_retries={self.max_retries}). Reason: {failure_summary}"
                 )
                 logger.error(error_message)
                 raise ReliableToolError(error_message, final_context=final_tool_context)

            # If not successful and retries remain, loop continues for the next attempt
            logger.info(
                "[%s]: Attempt %d failed or was invalidated. Retrying (%d/%d retries left)...",
                self.name,
                final_tool_context.attempt_count,
                self.max_retries - final_tool_context.attempt_count + 1, # Correct calculation
                self.max_retries
            )
            # The loop will automatically start the next chat with the updated context


    # --- Public Run Methods ---
    def run(
        self,
        task: str,
        context_variables: Optional[ContextVariables] = None,
        validation_prompt_addition: Optional[str] = None,
    ) -> Any:
        """Synchronously execute the wrapped function reliably.

        This method orchestrates the internal Runner-Validator chat to achieve
        a validated result for the given task.

        Args:
            task: The specific, detailed task description for the function.
            context_variables: Optional existing ContextVariables to use or extend.
            validation_prompt_addition: Optional additional validation criteria specific to this run
                                        (only used if enable_dynamic_validation=True).

        Returns:
            The result returned by the wrapped function upon successful and validated execution.

        Raises:
            TypeError: If this method is called on a tool wrapping an async function.
            ReliableToolError: If the execution fails after all allowed attempts.
        """
        if self._is_original_func_async:
            raise TypeError(
                f"Cannot use sync 'run()' for ReliableTool '{self.name}': "
                f"wrapped function '{self._original_func_name}' is async. Use 'a_run()'."
            )
        # Directly call the internal processing logic
        return self._process_run(
            task=task, context_variables=context_variables, validation_prompt_addition=validation_prompt_addition
        )

    async def a_run(
        self,
        task: str,
        context_variables: Optional[ContextVariables] = None,
        validation_prompt_addition: Optional[str] = None,
    ) -> Any:
        """Asynchronously execute the wrapped function reliably.

        This method orchestrates the internal Runner-Validator chat asynchronously
        to achieve a validated result for the given task.

        Note: While the internal group chat currently runs synchronously via
        `initiate_group_chat`, this method provides an async interface. True async
        execution of the internal chat might require `a_initiate_group_chat` if available
        or running the sync chat in a separate thread using `asyncio.to_thread`.

        Args:
            task: The specific, detailed task description for the function.
            context_variables: Optional existing ContextVariables to use or extend.
            validation_prompt_addition: Optional additional validation criteria specific to this run
                                        (only used if enable_dynamic_validation=True).

        Returns:
            The result returned by the wrapped function upon successful and validated execution.

        Raises:
            ReliableToolError: If the execution fails after all allowed attempts.
        """
        # For now, run the synchronous internal logic potentially in a thread
        # if the original function is async, otherwise directly call it.
        # This provides an async API but might block if _process_run is long.
        # A more robust solution would involve making _execute_internal_group_chat async
        # or reliably using asyncio.to_thread.

        # If the original function is sync, _process_run is fully sync.
        # If the original function is async, the wrapper call inside _process_run will be async,
        # but initiate_group_chat itself is sync.
        # Running the whole _process_run in a thread ensures non-blocking async API.

        loop = asyncio.get_running_loop()
        try:
             # Use functools.partial to pass arguments to the sync function
             func_call = functools.partial(
                 self._process_run,
                 task=task,
                 context_variables=context_variables,
                 validation_prompt_addition=validation_prompt_addition
            )
             # Run the potentially blocking sync function in a separate thread
             result = await loop.run_in_executor(None, func_call)
             return result
        except Exception as e:
             # Catch exceptions raised from _process_run (e.g., ReliableToolError)
             # and re-raise them in the async context.
             logger.error("[%s] a_run failed: %s", self.name, e, exc_info=True)
             raise e


    # --- Tool Schema and Call Override ---
    @property
    def tool_schema(self) -> dict[str, Any]:
        """Generate the tool schema for the public run/a_run methods.

        This schema defines how an LLM should call *this* ReliableTool,
        specifying the 'task' and optional 'validation_prompt_addition'.
        It does *not* describe the schema of the original wrapped function.
        """
        props = {"task": {"type": "string", "description": "Specific, detailed task for the reliable tool execution."}}
        req = ["task"]
        if self._enable_dynamic_validation:
            props["validation_prompt_addition"] = {
                "type": "string",
                "description": "Optional additional validation criteria or context specific to this execution run.",
            }
        # Schema for how an LLM should call this tool's run()/a_run()
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {"type": "object", "properties": props, "required": req},
            },
        }

    # Override __call__ to prevent direct invocation of the Tool instance itself.
    # Users should call run() or a_run().
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Override direct calling to guide users to use run() or a_run()."""
        sync_msg = " 'run(task=...)' or" if not self._is_original_func_async else ""
        async_msg = " 'a_run(task=...)'"
        raise NotImplementedError(
            f"Direct call ('{self.name}()') is not supported for ReliableTool. " f"Use the{sync_msg}{async_msg} method to execute the tool reliably."
        )

    # Ensure func property returns the original function for compatibility
    # although direct execution via func() bypasses the reliability mechanism.
    @property
    def func(self) -> Callable[..., Any]:
         """Return the original wrapped function."""
         # Note: Calling this directly bypasses the ReliableTool mechanism.
         # Use run() or a_run() for reliable execution.
         warnings.warn(f"Accessing ReliableTool.func returns the original, unreliable function '{self._original_func_name}'. Use run() or a_run() for reliable execution.", UserWarning)
         return self._original_func