import os

import nest_asyncio
from pydantic import BaseModel

from autogen import AssistantAgent, UserProxyAgent
from autogen.tools.experimental import Crawl4AITool
config_list = [{"model": "gpt-4o-mini", "api_key": os.environ["OPENAI_API_KEY"]}]

llm_config = {
    "config_list": config_list,
}

class Notebook(BaseModel):
    title: str
    url: str

class Notebooks(BaseModel):
    blogs: list[Notebook]

# Set llm_config and extraction_model to Crawl4AITool
crawlai_tool = Crawl4AITool(llm_config=llm_config, extraction_model=Notebooks)
from autogen.agents.experimental import ReliableFunctionAgent

crawler = ReliableFunctionAgent(name="Crawler", runner_llm_config=llm_config,
                                validator_llm_config=llm_config, func_or_tool=crawlai_tool, 
                                agent_system_message="""
You are a web surfer agent responsible for gathering information from the web to answer the provided question.
Call crawl4ai with all the provided paramteres to execute the scrape.""",
                                validator_system_message="""
Validations:
1. Validate that the URL is structurally valid
2. Validate that there is in fact a title
""")

res = crawler.run_func(task="Extract 2 notebooks urls from https://docs.ag2.ai/docs/use-cases/notebooks/Notebooks")
print(res)


# # %%
# import os

# import nest_asyncio

# from autogen.agentchat import UserProxyAgent
# from autogen.agents.experimental import WebSurferAgent

# nest_asyncio.apply()

# # %% [markdown]
# # ### Crawl4AI WebSurferAgent
# # 
# # > **Note:** [`Crawl4AI`](https://github.com/unclecode/crawl4ai) is built on top of [LiteLLM](https://github.com/BerriAI/litellm) and supports the same models as LiteLLM.
# # >
# # > We had great experience with `OpenAI`, `Anthropic`, `Gemini` and `Ollama`. However, as of this writing, `DeepSeek` is encountering some issues.
# # 

# # %%
# config_list = [{"model": "gpt-4o-mini", "api_key": os.environ["OPENAI_API_KEY"]}]

# llm_config = {
#     "config_list": config_list,
# }

# # %% [markdown]
# # There are two ways to start a chat session which is using only one agent with LLM configuration.
# # 
# # #### **Recommended:** Using the `run` Method
# # 
# # The new `run` method simplifies the process by eliminating the need for manual `UserProxyAgent` creation.
# # 
# # - ✅ **Easier setup** – No need to manually register tools
# # 

# # %%
# # `web_tool` parameter must be set to `crawl4ai` in order for the `Crawl4AITool` to be used.
# websurfer = WebSurferAgent(name="WebSurfer", llm_config=llm_config, web_tool="crawl4ai")

# websurfer.run(
#     message="Get info from https://docs.ag2.ai/docs/Home",
#     tools=websurfer.tools,
#     max_turns=2,
#     user_input=False,
# )
