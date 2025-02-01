# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
import copy
from typing import Any, List, Optional

from autogen.agentchat.contrib.learning.function_models import Memory
from autogen.agentchat.contrib.swarm_agent import SwarmAgent


class KnowledgeSharingSwarmAgent(SwarmAgent):
    def __init__(
        self,
        name: str,
        llm_config: Optional[dict] = None,
        system_message: str = "",
        knowledge_sources: Optional[List[SwarmAgent]] = None,
        use_own_knowledge: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(name=name, llm_config=llm_config, system_message=system_message, *args, **kwargs)

        self._base_system_message = system_message

        # Initialize knowledge sources
        self._knowledge_sources: List[KnowledgeSharingSwarmAgent] = [self] if use_own_knowledge else []
        self._knowledge_sources += knowledge_sources or []

        self.memories: list[Memory] = []

    def register_knowledge_sources(self, knowledge_sources: List[SwarmAgent]) -> None:
        """
        Registers knowledge sources for the agent.

        Args:
            knowledge_sources (List[SwarmAgent]): List of agents acting as knowledge sources.
        """
        self._knowledge_sources = knowledge_sources

    def add_knowledge_source(self, knowledge_source: SwarmAgent) -> None:
        """
        Adds a single knowledge source to the agent.

        Args:
            knowledge_source (SwarmAgent): An agent to add as a knowledge source.
        """
        self._knowledge_sources.append(knowledge_source)

    def remember(self, memory: Memory) -> None:
        self.memories.append(memory)

    def get_function_memories_as_messages(self) -> list[dict[str, Any]]:
        messages = [memory.to_memory_message() for memory in self.memories]

        for source in self._knowledge_sources:
            messages += [memory.to_memory_message() for memory in source.memories]

        return messages

    def get_function_memories_as_messages_recursive(self) -> list[dict[str, Any]]:
        messages = [memory.to_memory_message() for memory in self.memories]

        for source in self._knowledge_sources:
            messages += source.get_function_memories_as_messages_recursive()

        return messages


# I'm making both of these super safe and doing deepcopy everywhere


def get_all_agent_memories(agent_list) -> list:
    res = []
    for agent in agent_list:
        res.append(copy.deepcopy(agent.memories))
    return res


def restore_agent_memories(agent_list, memory_list):
    for i in range(len(agent_list)):
        agent_list[i].memories = copy.deepcopy(memory_list[i])
