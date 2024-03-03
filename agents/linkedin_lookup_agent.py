from tabnanny import verbose
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama

from langchain.agents import initialize_agent, Tool, AgentType


def lookup(name: str) -> str:
    llm = Ollama(model="gemma")
    template = """given the full name {name_of_person} I want to you to get me a link to their linkedin profile page.
                    Your answer should contain only a URL """
    tools_for_agent = [
        Tool(
            name="Crawl Goolge for linkedin profile page",
            func="?",
            description="useful for when ou need to get the linkedin page url",
        )
    ]

    agent = initialize_agent(
        tools=tools_for_agent,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )
    prompt_template = PromptTemplate(
        template=template, input_variables=["name_of_person"]
    )
    linkedin_profile_url = agent.run(prompt_template.format_prompt(name_of_person=name))
    return linkedin_profile_url
