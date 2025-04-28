# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import copy
import functools
import inspect
import traceback
import json
import time
import logging # Import logging module
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Type

# Use pydantic v2 BaseModel
from pydantic import BaseModel, Field as PydanticField, ValidationError

# Assuming these are available in your autogen installation path
# Adjust imports based on your actual project structure
try:
    from autogen import ConversableAgent, Agent
    # Import from new group chat modules
    # Assuming initiate_group_chat is synchronous based on provided signature
    from autogen.agentchat import initiate_group_chat
    from autogen.agentchat.group.patterns import DefaultPattern
    from autogen.agentchat.group import (
        ContextVariables,
        ReplyResult,
        AgentTarget,
        TerminateTarget,
    )
    from autogen.tools.tool import Tool
except ImportError as e:
    print(f"ImportError: {e}")
    # Use print here as logging might not be configured yet if autogen fails import
    print("Warning: Autogen components not found. Please install 'pyautogen'.")
    # Define dummy classes/functions if needed for static analysis, but runtime will fail.
    class Agent: pass
    class ConversableAgent(Agent): pass
    class Tool: pass
    class ContextVariables:
        def __init__(self, data: Optional[Dict] = None): self.data = data or {}
        def get(self, key, default=None): return self.data.get(key, default)
        def __getitem__(self, key): return self.data[key]
        def __setitem__(self, key, value): self.data[key] = value
        def __contains__(self, key): return key in self.data
    class ReplyResult:
        def __init__(self, context_variables=None, target=None, message=None): # Dummy init
            self.context_variables = context_variables
            self.target = target
            self.message = message
    class DefaultPattern:
         def __init__(self, *args, **kwargs): pass # Dummy init
    class AgentTarget:
         def __init__(self, name): self.name = name # Dummy init
    # Dummy sync initiate_group_chat
    def initiate_group_chat(*args, **kwargs) -> tuple[Optional[Any], ContextVariables, Optional[Agent]]:
        print("Warning: Using dummy initiate_group_chat. Autogen not installed.")
        initial_context = kwargs.get("pattern").context_variables if hasattr(kwargs.get("pattern"), "context_variables") else ContextVariables()
        return ({"role": "assistant", "content": "Dummy reply"}, initial_context, None)

# --- Setup Logger ---
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# --- Configuration Models ---
class AgentConfig(BaseModel):
    """Configuration for an internal agent within ReliableTool."""
    llm_config: Optional[dict] = PydanticField(default=None, description="LLM configuration for the agent.")
    system_message: str = PydanticField(default="", description="Base system message for the agent.")

# --- Pydantic Models for Context ---
class ValidationResult(BaseModel):
    """Represents the outcome of a single validation step."""
    validation_result: bool = PydanticField(description="Whether the function result satisfies validation criteria.")
    justification: str = PydanticField(description="Justification for the validation result.")

    class Config:
        extra = "forbid" # Enforce schema strictly

    def __str__(self) -> str:
        # Standard string representation (human-readable)
        return (
            f"Validation Result: {'Passed' if self.validation_result else 'Failed'}\n"
            f"Justification: {self.justification}"
        )

    def format(self) -> str:
        """
        Dumps the result as json so it can be interpreted back into an object later
        Uses the standard __str__ representation.
        """
        return self.model_dump_json()
    
class ExecutionAttempt(BaseModel):
    """Stores the state of a single attempt to execute and validate the function."""
    attempt_number: int = PydanticField(description="Sequential number of this attempt (starting from 1).")
    timestamp: float = PydanticField(default_factory=time.time, description="Timestamp when the attempt started.")
    args: List[Any] = PydanticField(default_factory=list, description="Positional arguments passed (best-effort serialization).")
    kwargs: Dict[str, Any] = PydanticField(default_factory=dict, description="Keyword arguments passed (best-effort serialization).")
    hypothesis: Optional[str] = PydanticField(None, description="Runner's hypothesis before this attempt.")
    error: Optional[str] = PydanticField(None, description="Error message (string) if function execution failed.")
    result_data: Optional[Any] = PydanticField(None, description="Raw result object (may not survive serialization cycles).")
    result_str: Optional[str] = PydanticField(None, description="String representation of function result.")
    validation: Optional[ValidationResult] = PydanticField(None, description="Result of the validation step for this attempt.")
    class Config: arbitrary_types_allowed = True
    def did_execute_successfully(self) -> bool: return self.error is None
    def did_validate_successfully(self) -> bool: return self.validation is not None and self.validation.validation_result

class ReliableToolContext(BaseModel):
    """Main context object holding the overall state and history of attempts."""
    task: str = PydanticField(description="The overall task description.")
    reliable_tool_name: str = PydanticField(description="Name of the ReliableTool instance.")
    start_time: float = PydanticField(default_factory=time.time, description="Timestamp when the reliable execution started.")
    dynamic_validation_input: Optional[str] = PydanticField(None, description="Dynamic validation criteria provided in the run() call, if any.")
    attempts: List[ExecutionAttempt] = PydanticField(default_factory=list, description="Chronological list of execution attempts.")
    class Config: arbitrary_types_allowed = True
    def get_attempt_count(self) -> int: return len(self.attempts)
    def get_latest_attempt(self) -> Optional[ExecutionAttempt]: return self.attempts[-1] if self.attempts else None
    def is_complete_and_successful(self) -> bool:
        latest = self.get_latest_attempt()
        return latest is not None and latest.did_execute_successfully() and latest.did_validate_successfully()
    def get_final_result_data(self) -> Any:
        for attempt in reversed(self.attempts):
            if attempt.did_execute_successfully() and attempt.did_validate_successfully(): return attempt.result_data
        return None
    def get_failure_summary(self) -> str:
        latest = self.get_latest_attempt()
        if latest is None: return "No attempts were made."
        if not latest.did_execute_successfully(): return f"Last attempt ({latest.attempt_number}) failed execution: {latest.error}"
        elif not latest.did_validate_successfully():
            just = latest.validation.justification if latest.validation else "Validation result missing or failed before completion"
            return f"Last attempt ({latest.attempt_number}) executed but failed validation (Justification: {just})"
        else: return "Execution seems successful but overall status check indicates failure (Internal inconsistency)."

# --- Helper Functions ---
def _configure_structured_output(llm_config: Optional[dict], structured_output_type: Type[BaseModel]) -> Optional[dict]:
    """
    Creates deep copy of llm_config, assigns the Pydantic model class directly
    to the 'response_format' key for AutoGen to handle, and removes
    conflicting 'tools'/'tool_choice' keys.
    """
    if llm_config is None:
        return None
    if not issubclass(structured_output_type, BaseModel):
         # Ensure we are actually getting a Pydantic model class
         raise TypeError("structured_output_type must be a Pydantic BaseModel subclass.")

    # Create a deep copy to avoid modifying the original config dict
    llm_config_copy = copy.deepcopy(llm_config)

    if "config_list" not in llm_config_copy or not isinstance(llm_config_copy["config_list"], list):
         raise ValueError("LLM config must contain a 'config_list' list.")

    config_list_updated = []
    config_modified = False
    for config_item in llm_config_copy["config_list"]:
        if not isinstance(config_item, dict):
            raise ValueError("Each item in 'config_list' must be a dictionary.")

        current_config = config_item.copy() # Work on a copy of the item

        # --- Assign Pydantic Model directly ---
        # Let AutoGen handle the conversion to the correct API format later.
        if current_config.get("response_format") is not structured_output_type:
             current_config["response_format"] = structured_output_type # Assign the class itself
             logger.debug(f"Assigned Pydantic model '{structured_output_type.__name__}' directly to response_format for validator.")
             config_modified = True

        # --- Remove Conflicting Keys ---
        if "tools" in current_config:
            del current_config["tools"]
            logger.debug("Removed 'tools' key from validator agent LLM config item.")
            config_modified = True

        if "tool_choice" in current_config:
            del current_config["tool_choice"]
            logger.debug("Removed 'tool_choice' key from validator agent LLM config item.")
            config_modified = True

        config_list_updated.append(current_config)

    # Only update if changes were made
    if config_modified:
        llm_config_copy["config_list"] = config_list_updated

    # Also handle potential top-level keys for safety
    if "tools" in llm_config_copy:
        del llm_config_copy["tools"]
        logger.debug("Removed top-level 'tools' key from validator agent LLM config.")
    if "tool_choice" in llm_config_copy:
        del llm_config_copy["tool_choice"]
        logger.debug("Removed top-level 'tool_choice' key from validator agent LLM config.")
    # Check/set top-level response_format as well
    if llm_config_copy.get("response_format") is not structured_output_type:
         llm_config_copy["response_format"] = structured_output_type
         logger.debug(f"Assigned Pydantic model '{structured_output_type.__name__}' directly to top-level response_format.")


    logger.debug("Final LLM config prepared for validator agent setup (response_format is Pydantic class): %s", llm_config_copy)
    # NOTE: The logger will show the type, e.g., <class '__main__.ValidationResult'>
    # for the response_format value now. This is expected.
    return llm_config_copy

def _get_last_non_empty_message_content(messages: Optional[List[Dict]]) -> Optional[str]:
    """Gets content of the last message with non-empty content."""
    if not messages: return None
    for message in reversed(messages):
        content = message.get("content")
        if isinstance(content, str) and content.strip(): return content
        if isinstance(content, list) and content:
            try: return json.dumps(content[0]) if isinstance(content[0], dict) else str(content[0])
            except Exception: pass
    return None

def _get_reliable_tool_context(context_variables: ContextVariables, context_key: str) -> ReliableToolContext:
    """Retrieves and validates the ReliableToolContext from ContextVariables."""
    if context_key not in context_variables.data: raise KeyError(f"ReliableToolContext key '{context_key}' not found in ContextVariables data.")
    context_data = context_variables.data[context_key]
    try:
        if isinstance(context_data, str): model = ReliableToolContext.model_validate_json(context_data)
        elif isinstance(context_data, dict): model = ReliableToolContext.model_validate(context_data)
        elif isinstance(context_data, ReliableToolContext): model = context_data
        else: raise TypeError(f"Expected context data for '{context_key}' to be str, dict, or ReliableToolContext, got {type(context_data)}")
        return model
    except (ValidationError, json.JSONDecodeError, TypeError) as e:
        error_detail = f"Data type: {type(context_data)}, Preview: '{str(context_data)[:100]}...'" if isinstance(context_data, (str, dict)) else f"Data type: {type(context_data)}"
        logger.error("Failed loading ReliableToolContext from key '%s'. Error: %s. %s", context_key, e, error_detail)
        raise ValueError(f"Failed loading ReliableToolContext from key '{context_key}'. Error: {e}. {error_detail}") from e

def _set_reliable_tool_context(context_variables: ContextVariables, context_key: str, context: ReliableToolContext):
    """Serializes and stores the ReliableToolContext in ContextVariables."""
    if not isinstance(context, ReliableToolContext): raise TypeError("Object to set must be a ReliableToolContext instance.")
    try: context_variables.data[context_key] = context.model_dump_json(warnings='warn')
    except Exception as e:
        context_dict_str = "Could not represent context as dict for logging."
        try: context_dict_str = str(context.model_dump(warnings='warn'))
        except Exception: pass
        logger.error("CRITICAL ERROR: Failed to serialize ReliableToolContext key '%s': %s. Context: %s", context_key, e, context_dict_str[:500], exc_info=True)
        raise ValueError(f"Critical error serializing ReliableToolContext: {e}") from e

def get_runner_prompt(task: str, agent_system_message: str, current_attempt_number: int, internal_tool_name: str) -> str:
    """Generates the system prompt for the internal runner agent."""
    return f"""
You are an AI assistant responsible for invoking a specific function based on the user's task.
Function to call: '{internal_tool_name}'
This is attempt number {current_attempt_number}. Ensure arguments are appropriate for this attempt number (e.g., for retry logic).

You MUST invoke the function '{internal_tool_name}' exactly one time per response using a tool call format that the system can execute.
Do NOT just output text explaining what you would do, or asking for confirmation. Directly make the tool call.
Analyze the task description and conversation history carefully to determine the correct arguments for the function call.
You MUST provide a 'hypothesis' argument summarizing the expected outcome or result format of the function call based on the inputs.

{agent_system_message}

Current Task:
{task}
"""

def get_validator_prompt(task: str, base_validator_system_message: str, dynamic_validation_addition: Optional[str] = None) -> str:
    """Generates the system prompt for the internal validator agent."""
    dynamic_section = ""
    if dynamic_validation_addition and dynamic_validation_addition.strip():
        dynamic_section = f"\nAdditional Requirements for This Specific Run:\n{dynamic_validation_addition.strip()}"
    return f"""
You are an AI validation assistant. You will receive the result (or error message) of a function call intended to accomplish a task.
Your goal is to validate if the function call result meets ALL requirements: the base task description, the base validation rules, and any additional requirements specified below.

Task Description:
{task}

Base Validation Rules/Context:
{base_validator_system_message}{dynamic_section}

Evaluate the function call result/error based on the information provided in the user message and all the requirements listed above.
Your output should conform to the required structured format for ValidationResult.
"""

# --- Reliable Function Wrapper (Remains Async) ---
# --- Reliable Function Wrapper (Conditionally Sync/Async) ---
def reliable_function_wrapper(tool_function: Callable, validator: ConversableAgent, context_variables_key: str):
    """
    Wraps the target function to integrate with the reliable execution flow.
    Returns a synchronous wrapper if tool_function is sync,
    and an asynchronous wrapper if tool_function is async.
    """
    is_original_async = inspect.iscoroutinefunction(tool_function)

    # --- Define the core logic (can be used by both sync/async wrappers) ---
    def _process_execution(current_attempt: ExecutionAttempt, context_variables: ContextVariables, context: ReliableToolContext, result: Any):
        """Processes the result after successful execution."""
        result_data, result_str = None, ""
        if isinstance(result, tuple) and len(result) == 2:
            result_data, result_direct_repr = result
            result_str = str(result_direct_repr) if not isinstance(result_direct_repr, str) else result_direct_repr
        elif isinstance(result, ReplyResult):
            logger.warning("Original function '%s' returned ReplyResult directly.", tool_function.__name__)
            result_data, result_str = result.message, str(result.message)
        else:
            result_data = result
            try: result_str = str(result_data)
            except Exception as str_e:
                logger.warning("Could not convert result type %s to string: %s. Using repr().", type(result_data).__name__, str_e)
                result_str = repr(result_data)
        current_attempt.result_data, current_attempt.result_str = result_data, result_str

        # Add successful attempt and update context
        context.attempts.append(current_attempt)
        try: _set_reliable_tool_context(context_variables, context_variables_key, context)
        except Exception as set_ctx_e: logger.error("Failed to update context variables after successful function execution: %s", set_ctx_e, exc_info=True)

        # Return ReplyResult directing to validator with result string
        return ReplyResult(context_variables=context_variables, target=AgentTarget(validator), message=current_attempt.result_str or "")

    def _handle_execution_error(current_attempt: ExecutionAttempt, context_variables: ContextVariables, context: ReliableToolContext, e: Exception):
        """Handles errors during execution."""
        tb_str = traceback.format_exc()
        error_msg = f"{type(e).__name__}: {e}"
        logger.error("Error during function execution (Attempt %d): %s\n%s", current_attempt.attempt_number, error_msg, tb_str)
        current_attempt.error = error_msg
        error_message_for_validator = f"Function execution failed on attempt {current_attempt.attempt_number}.\nError Type: {type(e).__name__}\nError: {e}\nTraceback:\n{tb_str}"

        # Add failed attempt and update context
        context.attempts.append(current_attempt)
        try: _set_reliable_tool_context(context_variables, context_variables_key, context)
        except Exception as set_ctx_e: logger.error("Failed to update context variables after function execution error: %s", set_ctx_e, exc_info=True)

        # Return ReplyResult directing back to validator with error
        return ReplyResult(context_variables=context_variables, target=AgentTarget(validator), message=error_message_for_validator)

    def _prepare_call_args(tool_sig: inspect.Signature, args: tuple, kwargs: dict, context_variables: ContextVariables) -> Tuple[tuple, dict]:
        """Prepares args/kwargs for the tool function call, injecting context_variables if needed."""
        call_args, call_kwargs = args, kwargs.copy()
        if "context_variables" in tool_sig.parameters:
            param_type = tool_sig.parameters["context_variables"].annotation
            if param_type is ContextVariables or param_type is inspect.Parameter.empty or param_type is Any:
                 call_kwargs["context_variables"] = context_variables
        return call_args, call_kwargs

    # --- Create either a sync or async wrapper ---
    tool_sig = inspect.signature(tool_function)

    if not is_original_async:
        # --- Synchronous Wrapper ---
        @functools.wraps(tool_function)
        def sync_wrapper(*args, hypothesis: str, context_variables: ContextVariables, **kwargs) -> ReplyResult:
            context: ReliableToolContext = _get_reliable_tool_context(context_variables, context_variables_key)
            attempt_number = context.get_attempt_count() + 1
            current_attempt = ExecutionAttempt(attempt_number=attempt_number, args=list(args), kwargs=kwargs.copy(), hypothesis=hypothesis)
            try:
                call_args, call_kwargs = _prepare_call_args(tool_sig, args, kwargs, context_variables)
                # Directly call the synchronous function
                result = tool_function(*call_args, **call_kwargs)
                return _process_execution(current_attempt, context_variables, context, result)
            except Exception as e:
                return _handle_execution_error(current_attempt, context_variables, context, e)
        wrapper_func = sync_wrapper
    else:
        # --- Asynchronous Wrapper ---
        @functools.wraps(tool_function)
        async def async_wrapper(*args, hypothesis: str, context_variables: ContextVariables, **kwargs) -> ReplyResult:
            context: ReliableToolContext = _get_reliable_tool_context(context_variables, context_variables_key)
            attempt_number = context.get_attempt_count() + 1
            current_attempt = ExecutionAttempt(attempt_number=attempt_number, args=list(args), kwargs=kwargs.copy(), hypothesis=hypothesis)
            try:
                call_args, call_kwargs = _prepare_call_args(tool_sig, args, kwargs, context_variables)
                # Await the asynchronous function
                result = await tool_function(*call_args, **call_kwargs)
                return _process_execution(current_attempt, context_variables, context, result)
            except Exception as e:
                return _handle_execution_error(current_attempt, context_variables, context, e)
        wrapper_func = async_wrapper

    # --- Adapt Signature for the chosen Wrapper ---
    params = list(tool_sig.parameters.values())
    hypothesis_param = inspect.Parameter("hypothesis", kind=inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=str)
    context_variables_param = inspect.Parameter("context_variables", kind=inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=ContextVariables)
    original_params_without_conflict = [p for p in params if p.name not in ("hypothesis", "context_variables")]
    wrapper_func.__signature__ = inspect.Signature(
        parameters=[hypothesis_param, context_variables_param] + original_params_without_conflict ,
        return_annotation=ReplyResult
    )

    return wrapper_func # Return the sync or async wrapper function

# --- ReliableTool Class (Refactored) ---
class ReliableToolError(Exception):
    """Custom exception for errors during ReliableTool execution."""
    def __init__(self, message: str, final_context: Optional[ReliableToolContext] = None):
        super().__init__(message)
        self.final_context = final_context

class ReliableTool(Tool):
    """
    A Tool wrapper that ensures reliable execution of a function or another tool
    through an internal LLM-based validation and retry loop using a Group Chat.

    Execution Interfaces:
    - `a_run()`: Asynchronous method, usable for both sync and async wrapped functions.
                 Provides an awaitable interface required by async environments.
    - `run()`:   Synchronous method. Blocks until completion.
                 *ONLY* usable if the wrapped function (`func_or_tool`) is synchronous.

    Direct calling (`tool()`) is disabled.

    Assumes `autogen.agentchat.initiate_group_chat` is synchronous.
    """
    INTERNAL_TOOL_NAME_PREFIX = "execute_"

    def __init__(
        self,
        name: str,
        func_or_tool: Union[Callable, Tool],
        runner_config: AgentConfig,
        validator_config: AgentConfig,
        description: Optional[str] = None,
        max_retries: int = 3,
        agent_kwargs: Optional[dict] = None,
        enable_dynamic_validation: bool = False,
    ) -> None:
        """Initializes the ReliableTool instance."""
        self._original_func, original_name, original_description = self._extract_func_details(func_or_tool)
        # Store whether the *original* function provided by the user is async
        self._is_original_func_async = inspect.iscoroutinefunction(self._original_func)

        # Initialize base Tool class
        super().__init__(
            name=name,
            description=description or f"Reliably executes '{original_name}': {original_description}",
            func_or_tool=self._original_func # Store original func for base class metadata
        )

        # Store configurations and parameters
        if not validator_config.llm_config: raise ValueError("Validator agent requires an LLM configuration (`validator_config.llm_config`).")
        # --- Crucial Step: Decouple LLM Configs ---
        # Store deep copies of the configurations to prevent shared object issues.
        self._runner_config = copy.deepcopy(runner_config)
        self._validator_config = copy.deepcopy(validator_config)
        # --------------------------------------------

        # Validate validator config after deep copying
        if not self._validator_config.llm_config:
            raise ValueError("Validator agent requires an LLM configuration (`validator_config.llm_config`).")

        # Store other parameters
        self.max_retries = max_retries
        self._context_variables_key = f"{self.name}-ReliableToolContext"
        # Deep copy agent_kwargs as well, as they might contain mutable defaults (like lists)
        self._agent_kwargs = copy.deepcopy(agent_kwargs) if agent_kwargs else {}
        self._agent_kwargs = agent_kwargs or {}
        self._original_func_name = original_name
        self._enable_dynamic_validation = enable_dynamic_validation

        # Setup internal components
        self._validator_name = f"{self.name}_Validator"
        self._runner_name = f"{self.name}_Runner"
        self._validator = self._setup_validator_agent()
        # The wrapper function passed to the runner's tool IS async,
        # as it needs to handle awaiting the original function if necessary.
        self._reliable_func_wrapper = reliable_function_wrapper(
            self._original_func, self._validator, self._context_variables_key
        )
        self._runner = self._setup_runner_agent() # Runner agent uses the async wrapper
        self._register_internal_hooks()

    def _extract_func_details(self, func_or_tool: Union[Callable, Tool]) -> Tuple[Callable, str, str]:
        """Extracts the core function, its name, and description from input."""
        if isinstance(func_or_tool, Tool):
            if not callable(func_or_tool.func): raise TypeError(f"The 'func' attribute of the Tool '{func_or_tool.name}' is not callable.")
            return func_or_tool.func, func_or_tool.name, func_or_tool.description
        elif callable(func_or_tool):
            name = getattr(func_or_tool, "__name__", "callable_function")
            desc = getattr(func_or_tool, "__doc__", "") or f"Callable function '{name}'."
            return func_or_tool, name, desc
        else: raise TypeError("Input 'func_or_tool' must be a callable function/method or an autogen.Tool instance.")

    def _setup_validator_agent(self) -> ConversableAgent:
        """Creates and configures the internal validator agent."""
        try: structured_llm_config = _configure_structured_output(self._validator_config.llm_config, ValidationResult)
        except Exception as e: raise ValueError(f"Failed to configure validator LLM config for structured output: {e}")
        return ConversableAgent(name=self._validator_name, llm_config=structured_llm_config,
                                system_message="[Validator Prompt Updated Per Run]", human_input_mode="NEVER", **self._agent_kwargs)

    def _setup_runner_agent(self) -> ConversableAgent:
        """Creates and configures the internal runner agent."""
        runner = ConversableAgent(name=self._runner_name, llm_config=copy.deepcopy(self._runner_config.llm_config),
                                  system_message="[Runner Prompt Updated Per Run]", human_input_mode="NEVER", **self._agent_kwargs)
        internal_tool_name = f"{self.INTERNAL_TOOL_NAME_PREFIX}{self._original_func_name}"
        # The runner's tool uses the async wrapper function
        internal_tool = Tool(name=internal_tool_name,
                             description=f"Executes the core logic for the task: {self._original_func.__doc__ or self._original_func_name}. Must provide 'hypothesis'.",
                             func_or_tool=self._reliable_func_wrapper) # Pass the async wrapper here
        # Register tool for execution. Autogen's agent execution logic should handle calling the async wrapper.
        internal_tool.register_tool(runner)

        # Ensure runner's LLM config includes the tool schema
        if runner.llm_config:
            if "tools" not in runner.llm_config or not isinstance(runner.llm_config.get("tools"), list): runner.llm_config["tools"] = []
            schema = internal_tool.tool_schema
            if not any(t.get("function", {}).get("name") == internal_tool_name for t in runner.llm_config["tools"] if isinstance(t, dict)):
                runner.llm_config["tools"].append(schema)
        else: logger.warning("Runner agent '%s' has no llm_config. Function calling relies on execution registration.", self._runner_name)
        return runner

    def _register_internal_hooks(self) -> None:
        """Registers hooks necessary for the internal group chat interaction."""
        self._validator.register_hook("process_message_before_send", self._validator_structured_output_hook)
        self._validator.register_hook("process_all_messages_before_reply", self._validator_history_filter_hook)
        self._runner.register_hook("process_message_before_send", self._ensure_function_call_hook)

    # --- Hook Implementations (Unchanged from previous version) ---
    def _validator_structured_output_hook(
        self,
        sender: Agent, # The validator agent instance
        message: Union[dict, str],
        recipient: Agent,
        silent: bool
    ) -> Union[dict, str]:
        """
        Processes validator's output before sending. Extracts content, validates
        against ValidationResult, UPDATES the context's latest attempt using
        sender.context_variables, and returns a consistent string representation.
        """
        if sender.name != self._validator_name:
            return message # Only process messages from the validator

        # 1. Extract text content
        message_text: Optional[str] = None
        # --- (Extraction logic - same as previous version) ---
        if isinstance(message, str):
            message_text = message.strip()
        elif isinstance(message, dict) and isinstance(message.get("content"), str):
            message_text = message["content"].strip()
        elif isinstance(message, dict) and message.get("content") is not None:
             try: message_text = str(message["content"]).strip()
             except Exception: pass
        # ---

        if not message_text:
            logger.warning("Validator hook: Received empty/invalid content. Cannot update context.")
            return message if isinstance(message, dict) else "[Validator produced empty output]"

        # 2. Attempt parsing and validation
        validation_result_obj: Optional[ValidationResult] = None
        parsed_successfully = False
        try:
            validation_result_obj = ValidationResult.model_validate_json(message_text)
            parsed_successfully = True
            log_level = logging.INFO if validation_result_obj.validation_result else logging.WARNING
            logger.log(
                log_level,
                "Validator Hook: %s. Justification: %s",
                "PASSED" if validation_result_obj.validation_result else "FAILED",
                validation_result_obj.justification
            )

        except (ValidationError, json.JSONDecodeError, Exception) as e:
            # Log parsing/validation failure
            log_level = logging.ERROR if not isinstance(e, (ValidationError, json.JSONDecodeError)) else logging.WARNING
            logger.log(
                 log_level,
                 "Validator Hook: Error processing validator output (%s): %s\nRaw text: ```\n%s\n```",
                 type(e).__name__, e, message_text,
                 exc_info=log_level == logging.ERROR
            )
            # Create a default 'failed' validation object for context update attempt
            validation_result_obj = ValidationResult(
                validation_result=False,
                justification=f"Validation failed due to parsing/validation error in hook: {type(e).__name__}: {e}"
            )
            # We still want to try and update the context attempt as failed

        # 3. Attempt to update the context (if parsing succeeded or failed gracefully)
        if validation_result_obj: # Ensure we have a ValidationResult object
            try:
                # --- Access ContextVariables directly from sender ---
                current_context_variables = getattr(sender, "context_variables", None)

                if not isinstance(current_context_variables, ContextVariables):
                    # Log this critical issue if the expected attribute isn't there/right type
                    logger.error(
                        "Validator hook: Could not access 'context_variables' attribute or it's not a ContextVariables instance on sender agent '%s'. Cannot update context state from hook.",
                        sender.name
                    )
                else:
                    # --- ContextVariables found - Proceed with update ---
                    tool_context = _get_reliable_tool_context(current_context_variables, self._context_variables_key)
                    latest_attempt = tool_context.get_latest_attempt()

                    if latest_attempt:
                        if latest_attempt.validation is None: # Only update if not already set
                            latest_attempt.validation = validation_result_obj
                            logger.info("Validator hook: Updated latest attempt (%d) validation status.", latest_attempt.attempt_number)
                            # Save the updated context back
                            _set_reliable_tool_context(current_context_variables, self._context_variables_key, tool_context)
                        else:
                            # This case might happen if the hook runs multiple times for some reason, or state is weird
                            logger.warning("Validator hook: Latest attempt (%d) validation already set. Skipping update.", latest_attempt.attempt_number)
                    else:
                        logger.warning("Validator hook: No attempts found in context to update.")

            except KeyError as ke:
                logger.error("Validator hook: ReliableToolContext key '%s' not found in ContextVariables. Error: %s", self._context_variables_key, ke)
            except (AttributeError, TypeError, ValueError) as e:
                # Catch errors during context retrieval/update
                logger.error("Validator hook: Failed to retrieve or update ReliableToolContext. Error: %s", e, exc_info=True)
            except Exception as e:
                # Catch any other unexpected errors during context update
                logger.error("Validator hook: Unexpected error during context update. Error: %s", e, exc_info=True)

        # 4. Return the consistent string representation for the chat message
        if parsed_successfully and validation_result_obj:
             # If parsing succeeded, use the object's format/str method
             return validation_result_obj.format() # Use format() which returns JSON string
        else:
             # If parsing failed, return the informative error string
             # Use the justification from the default failure object if available
             err_justification = validation_result_obj.justification if validation_result_obj else "Parsing/Validation Failed"
             return f"Error validating result: {err_justification}. Raw Output:\n{message_text}" # Keep it concise

    def _validator_history_filter_hook(self, messages: List[Dict], **kwargs) -> List[Dict]:
        """Filters message history for validator: only system prompt + last user message (result/error)."""
        if not messages: return []
        system_message = messages[0] if messages and messages[0].get("role") == "system" else None
        last_content = _get_last_non_empty_message_content(messages) or "No function result content found in history."
        filtered = [msg for msg in ([system_message] if system_message else [])]
        filtered.append({"role": "user", "content": f"Function Result/Error to Validate:\n```\n{str(last_content)}\n```"})
        return filtered

    def _ensure_function_call_hook(self, sender: Agent, message: Union[dict, str], recipient: Agent, silent: bool) -> Union[dict, str]:
        """Hook to encourage the runner LLM to make a tool call if it outputs plain text."""
        if sender.name == self._runner_name:
             is_tool_call = isinstance(message, dict) and message.get("tool_calls")
             content = message.get("content", "") if isinstance(message, dict) else message
             internal_tool_name = f"{self.INTERNAL_TOOL_NAME_PREFIX}{self._original_func_name}"
             # Add reminder if no tool call and doesn't seem like an error/clarification related to the function
             if not is_tool_call and not (internal_tool_name in content or ("error" in content.lower() and "function" in content.lower())):
                logger.warning("Runner '%s' did not generate tool call. Appending reminder to call '%s'.", self._runner_name, internal_tool_name)
                reminder = f"\n\n[[System Reminder: You MUST invoke the function '{internal_tool_name}' using a tool call. Do not just reply with text.]]"
                if isinstance(message, dict): message["content"] = (message.get("content") or "") + reminder
                else: message += reminder
        return message

    # --- Core Execution Logic (Sync Helper) ---
    def _execute_internal_group_chat( self, task: str, initial_context: ContextVariables, dynamic_validation_str: Optional[str] = None ) -> Tuple[Optional[Union[Dict, ReplyResult]], ContextVariables, Optional[Agent]]:
        """Sets up prompts, runs the internal group chat (synchronously), and returns final state."""
        current_tool_context = _get_reliable_tool_context(initial_context, self._context_variables_key)
        next_attempt_number = current_tool_context.get_attempt_count() + 1
        internal_tool_name = f"{self.INTERNAL_TOOL_NAME_PREFIX}{self._original_func_name}"

        # Update Agent Prompts
        runner_prompt = get_runner_prompt(task, self._runner_config.system_message, next_attempt_number, internal_tool_name)
        self._runner.update_system_message(runner_prompt)
        validator_prompt = get_validator_prompt(task, self._validator_config.system_message, dynamic_validation_str)
        self._validator.update_system_message(validator_prompt)

        # Prepare and Run Group Chat (assuming initiate_group_chat is sync)
        initial_messages = [{"role": "user", "content": f"Start Reliable Task: {task}"}]
        max_group_chat_rounds = (self.max_retries + 1) * 2 + 4 # Heuristic max rounds
        agent_pattern = DefaultPattern(agents=[self._runner, self._validator], initial_agent=self._runner, context_variables=initial_context)

        logger.info("--- Starting ReliableTool '%s' Internal Group Chat (Sync) (Attempt %d / %d Max) ---", self.name, next_attempt_number, self.max_retries + 1)
        try:
            # Call the synchronous initiate_group_chat function
            last_reply, final_context, last_agent = initiate_group_chat(
                pattern=agent_pattern,
                messages=initial_messages,
                max_rounds=max_group_chat_rounds,
            )
            logger.info("--- ReliableTool '%s' Internal Group Chat Finished ---", self.name)
            return last_reply, final_context, last_agent
        except Exception as e:
             logger.critical("CRITICAL ERROR during sync internal group chat for ReliableTool '%s': %s", self.name, e, exc_info=True)
             # Return initial context on critical failure to preserve state if possible
             return None, initial_context, None

    # --- Main Execution Processing (Sync Helper) ---
    def _process_run(
        self,
        task: str,
        context_variables: Optional[ContextVariables] = None,
        validation_prompt_addition: Optional[str] = None
    ) -> Any:
        """
        Internal synchronous logic shared by run() and a_run().
        Relies on the validator hook to update the context state.
        """
        # 1. Initialize Context / Handle Dynamic Validation Input
        # --- (Same as before) ---
        current_context_variables = context_variables if context_variables is not None else ContextVariables()
        if not isinstance(current_context_variables, ContextVariables): raise TypeError(f"Expected context_variables to be ContextVariables or None, got {type(context_variables)}")
        dynamic_criteria_to_use = None
        if self._enable_dynamic_validation:
            if validation_prompt_addition: dynamic_criteria_to_use = validation_prompt_addition
        elif validation_prompt_addition: logger.warning("[%s]: 'validation_prompt_addition' provided but dynamic validation disabled. Ignoring.", self.name)
        initial_tool_context = ReliableToolContext(task=task, reliable_tool_name=self.name, dynamic_validation_input=dynamic_criteria_to_use)
        _set_reliable_tool_context(current_context_variables, self._context_variables_key, initial_tool_context)
        # ---

        # 2. Execute the Internal Group Chat (Sync)
        #    The validator hook within this chat attempts to update the context.
        last_reply, final_context_variables, last_agent = self._execute_internal_group_chat(
            task=task, initial_context=current_context_variables, dynamic_validation_str=dynamic_criteria_to_use
        )

        # 3. Process Final Result (Retrieve Context Updated by Hook)
        try:
            # Retrieve the potentially modified context
            final_tool_context = _get_reliable_tool_context(final_context_variables, self._context_variables_key)
        except (KeyError, ValueError) as e:
             # If context retrieval fails, something went very wrong
             raise ReliableToolError(f"ReliableTool '{self.name}' failed: Could not retrieve final execution context. Group chat error or context corruption likely: {e}", final_context=None) from e

        # --- Check if Hook Succeeded ---
        # Get the last attempt from the final context
        last_attempt = final_tool_context.get_latest_attempt()

        # If the validation field is *still* None after the chat, it means
        # the hook failed to access/update the context, or the chat ended
        # before the hook could run properly. Assume failure in this case.
        if last_attempt and last_attempt.validation is None:
             logger.warning(
                 "[%s]: Final validation state for last attempt (%d) is missing after chat completion. "
                 "This might indicate the validator hook failed to update context or the chat ended prematurely. Assuming failure.",
                 self.name, last_attempt.attempt_number
             )
             # Explicitly set the validation state to failed
             last_attempt.validation = ValidationResult(
                 validation_result=False,
                 justification="Validation state missing after chat completion; assumed failure."
             )
             # Attempt to save this assumed failure state back to the context variables
             try:
                 _set_reliable_tool_context(final_context_variables, self._context_variables_key, final_tool_context)
             except Exception as e:
                 # Log if saving the assumed failure state fails, but proceed
                 logger.error("[%s]: Failed to save assumed validation failure state after chat: %s", self.name, e)

        # --- REMOVED: Redundant logic parsing last_reply content ---
        # The 'validation_updated' flag and associated parsing block are gone.

        # 4. Check Overall Success Based on Final Context State and Return/Raise
        if final_tool_context.is_complete_and_successful():
             logger.info("ReliableTool '%s' completed successfully and validated after %d attempt(s).", self.name, final_tool_context.get_attempt_count())
             return final_tool_context.get_final_result_data()
        else:
            # Use the failure summary which reads directly from the (potentially updated) context
            failure_summary = final_tool_context.get_failure_summary()
            total_attempts = final_tool_context.get_attempt_count()
            error_message = f"ReliableTool '{self.name}' failed after {total_attempts} attempt(s) (max_retries={self.max_retries}). Reason: {failure_summary}"
            logger.error(error_message)
            # Raise error with the final context object for inspection
            raise ReliableToolError(error_message, final_context=final_tool_context)
    # --- Public Run Methods ---

    def run(
        self,
        task: str,
        context_variables: Optional[ContextVariables] = None,
        validation_prompt_addition: Optional[str] = None
    ) -> Any:
        """
        Synchronously executes the wrapped function reliably using an internal group chat.

        *Important*: This method can ONLY be used if the original `func_or_tool`
        provided during initialization was synchronous (i.e., not an `async def` function).
        Using this method with an async `func_or_tool` will raise a `TypeError`.

        Args:
            task: The specific task description for the function to accomplish.
            context_variables: Optional ContextVariables object to pass initial state. If None,
                               a fresh context is created for this run.
            validation_prompt_addition: Optional string with additional validation criteria for this run.
                                        Only used if `enable_dynamic_validation` is True.

        Returns:
            The validated `result_data` (potentially complex object) from the successful function execution.

        Raises:
            TypeError: If this method is called when the original `func_or_tool` was asynchronous.
            ReliableToolError: If execution fails validation after max_retries, or if a critical error occurs.
        """
        if self._is_original_func_async:
            # Raise error if the user tries sync run with an async function
            raise TypeError(
                f"Cannot use synchronous 'run()' for ReliableTool '{self.name}' because the wrapped "
                f"function '{self._original_func_name}' is asynchronous. Use the asynchronous 'a_run()' method instead."
            )

        # If the original function is sync, we can proceed with the synchronous execution flow.
        # _process_run handles the synchronous group chat and processing.
        return self._process_run(
            task=task,
            context_variables=context_variables,
            validation_prompt_addition=validation_prompt_addition
        )

    async def a_run(
        self,
        task: str,
        context_variables: Optional[ContextVariables] = None,
        validation_prompt_addition: Optional[str] = None
    ) -> Any:
        """
        Asynchronously executes the wrapped function reliably using an internal group chat.

        This method provides an awaitable interface suitable for use in `asyncio` applications.
        It can be used regardless of whether the original `func_or_tool` was synchronous or asynchronous.

        Note: While this method is `async`, the underlying group chat execution
        (`initiate_group_chat`) is assumed to be synchronous based on its signature.
        This `a_run` method primarily serves to allow integration into async code flows.
        If the underlying synchronous operations block significantly (e.g., long LLM calls),
        consider running this coroutine using `asyncio.to_thread` in the calling code
        to avoid blocking the main event loop.

        Args:
            task: The specific task description for the function to accomplish.
            context_variables: Optional ContextVariables object to pass initial state.
            validation_prompt_addition: Optional string with additional validation criteria.

        Returns:
            The validated `result_data` from the successful execution.

        Raises:
            ReliableToolError: If execution fails validation after max_retries or critical error.
        """
        # Although _process_run is synchronous internally (due to sync initiate_group_chat),
        # wrapping it in an async method allows it to be awaited by async callers.
        try:
            # Directly call the synchronous processing function.
            # The `await` in the calling code will effectively wait for this sync function to complete.
            result = self._process_run(
                task=task,
                context_variables=context_variables,
                validation_prompt_addition=validation_prompt_addition
            )
            return result
        except Exception as e:
             # Propagate exceptions, including ReliableToolError, from the sync process.
             raise e

    # --- Tool Schema and Call Override ---
    @property
    def tool_schema(self) -> dict[str, Any]:
        """Generates schema for the 'run'/'a_run' methods, conditionally adding dynamic validation parameter."""
        props = {"task": {"type": "string", "description": "Specific, detailed task for the function."}}
        req = ["task"]
        if self._enable_dynamic_validation:
            props["validation_prompt_addition"] = {"type": "string", "description": "(Optional) Specific, additional instructions or criteria for the validator agent for this particular run."}
        # The schema describes how an LLM should call *this tool* (i.e., invoke run/a_run)
        return {"type": "function", "function": {"name": self.name, "description": self.description, "parameters": {"type": "object", "properties": props, "required": req}}}

    def __call__(self, *args, **kwargs):
        """Override direct calling of the tool instance to guide users."""
        # Provide guidance based on whether sync `run` is allowed
        sync_msg_part = ""
        if not self._is_original_func_async:
             sync_msg_part = " 'run(task=...)' or" # Only suggest 'run' if original func is sync

        async_msg_part = " 'a_run(task=...)'"
        raise NotImplementedError(f"Direct call ('{self.name}()') unsupported. Use{sync_msg_part}{async_msg_part} method.")