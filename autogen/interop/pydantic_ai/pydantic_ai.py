# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0


import sys
import warnings
from functools import wraps
from inspect import signature
from typing import Any, Callable, Optional

from ..registry import register_interoperable_class
from .pydantic_ai_tool import PydanticAITool as AG2PydanticAITool

__all__ = ["PydanticAIInteroperability"]


@register_interoperable_class("pydanticai")
class PydanticAIInteroperability:
    """A class implementing the `Interoperable` protocol for converting Pydantic AI tools
    into a general `Tool` format.

    This class takes a `PydanticAITool` and converts it into a standard `Tool` object,
    ensuring compatibility between Pydantic AI tools and other systems that expect
    the `Tool` format. It also provides a mechanism for injecting context parameters
    into the tool's function.
    """

    @staticmethod
    def inject_params(
        ctx: Any,
        tool: Any,
    ) -> Callable[..., Any]:
        """Wraps the tool's function to inject context parameters and handle retries.

        This method ensures that context parameters are properly passed to the tool
        when invoked and that retries are managed according to the tool's settings.

        Args:
        ----
            ctx (Optional[RunContext[Any]]): The run context, which may include dependencies and retry information.
            tool (PydanticAITool): The Pydantic AI tool whose function is to be wrapped.

        Returns:
        -------
            Callable[..., Any]: A wrapped function that includes context injection and retry handling.

        Raises:
        ------
            ValueError: If the tool fails after the maximum number of retries.

        """
        from pydantic_ai import RunContext
        from pydantic_ai.tools import Tool as PydanticAITool

        ctx_typed: Optional[RunContext[Any]] = ctx  # type: ignore
        tool_typed: PydanticAITool[Any] = tool  # type: ignore

        max_retries = tool_typed.max_retries if tool_typed.max_retries is not None else 1
        f = tool_typed.function

        @wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if tool_typed.current_retry >= max_retries:
                raise ValueError(f"{tool_typed.name} failed after {max_retries} retries")

            try:
                if ctx_typed is not None:
                    kwargs.pop("ctx", None)
                    ctx_typed.retry = tool_typed.current_retry
                    result = f(**kwargs, ctx=ctx_typed)  # type: ignore[call-arg]
                else:
                    result = f(**kwargs)  # type: ignore[call-arg]
                tool_typed.current_retry = 0
            except Exception as e:
                tool_typed.current_retry += 1
                raise e

            return result

        sig = signature(f)
        if ctx_typed is not None:
            new_params = [param for name, param in sig.parameters.items() if name != "ctx"]
        else:
            new_params = list(sig.parameters.values())

        wrapper.__signature__ = sig.replace(parameters=new_params)  # type: ignore[attr-defined]

        return wrapper

    @classmethod
    def convert_tool(cls, tool: Any, deps: Any = None, **kwargs: Any) -> AG2PydanticAITool:
        """Converts a given Pydantic AI tool into a general `Tool` format.

        This method verifies that the provided tool is a valid `PydanticAITool`,
        handles context dependencies if necessary, and returns a standardized `Tool` object.

        Args:
        ----
            tool (Any): The tool to convert, expected to be an instance of `PydanticAITool`.
            deps (Any, optional): The dependencies to inject into the context, required if
                                   the tool takes a context. Defaults to None.
            **kwargs (Any): Additional arguments that are not used in this method.

        Returns:
        -------
            AG2PydanticAITool: A standardized `Tool` object converted from the Pydantic AI tool.

        Raises:
        ------
            ValueError: If the provided tool is not an instance of `PydanticAITool`, or if
                        dependencies are missing for tools that require a context.
            UserWarning: If the `deps` argument is provided for a tool that does not take a context.

        """
        from pydantic_ai import RunContext
        from pydantic_ai.tools import Tool as PydanticAITool

        if not isinstance(tool, PydanticAITool):
            raise ValueError(f"Expected an instance of `pydantic_ai.tools.Tool`, got {type(tool)}")

        # needed for type checking
        pydantic_ai_tool: PydanticAITool[Any] = tool  # type: ignore[no-any-unimported]

        if tool.takes_ctx and deps is None:
            raise ValueError("If the tool takes a context, the `deps` argument must be provided")
        if not tool.takes_ctx and deps is not None:
            warnings.warn(
                "The `deps` argument is provided but will be ignored because the tool does not take a context.",
                UserWarning,
            )

        if tool.takes_ctx:
            ctx = RunContext(
                deps=deps,
                retry=0,
                # All messages send to or returned by a model.
                # This is mostly used on pydantic_ai Agent level.
                messages=[],  # TODO: check in the future if this is needed on Tool level
                tool_name=pydantic_ai_tool.name,
            )
        else:
            ctx = None

        func = PydanticAIInteroperability.inject_params(
            ctx=ctx,
            tool=pydantic_ai_tool,
        )

        return AG2PydanticAITool(
            name=pydantic_ai_tool.name,
            description=pydantic_ai_tool.description,
            func=func,
            parameters_json_schema=pydantic_ai_tool._parameters_json_schema,
        )

    @classmethod
    def get_unsupported_reason(cls) -> Optional[str]:
        if sys.version_info < (3, 9):
            return "This submodule is only supported for Python versions 3.9 and above"

        try:
            import pydantic_ai.tools
        except ImportError:
            return "Please install `interop-pydantic-ai` extra to use this module:\n\n\tpip install ag2[interop-pydantic-ai]"

        return None
