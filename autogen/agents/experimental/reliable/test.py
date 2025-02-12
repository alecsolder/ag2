# # Demo of Reliable Function Swarm Agent with Crawl4AI
# ## Installation
# 
# To get started with the `crawl4ai` integration in AG2, follow these steps:
# 
# 1. Install AG2 with the `crawl4ai` extra:
#    ```bash
#    pip install ag2[crawl4ai]
#    ```
#    > **Note:** If you have been using `autogen` or `pyautogen`, all you need to do is upgrade it using:  
#    > ```bash
#    > pip install -U autogen[crawl4ai]
#    > ```
#    > or  
#    > ```bash
#    > pip install -U pyautogen[crawl4ai]
#    > ```
#    > as `pyautogen`, `autogen`, and `ag2` are aliases for the same PyPI package.  
# 2. Set up Playwright:
#    
#    ```bash
#    # Installs Playwright and browsers for all OS
#    playwright install
#    # Additional command, mandatory for Linux only
#    playwright install-deps
#    ```
# 
# 3. For running the code in Jupyter, use `nest_asyncio` to allow nested event loops.
#     ```bash
#     pip install nest_asyncio
#     ```
# 
# 
# You're all set! Now you can start using browsing features in AG2.
# 
# 
# ## Imports

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

res = crawler.run_func(task="Extract all notebooks urls from https://docs.ag2.ai/docs/use-cases/notebooks/Notebooks")
print(res)


