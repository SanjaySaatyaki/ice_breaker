from itertools import chain
import os
from pyexpat import model
from click import prompt
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import Ollama
from langchain.chains import LLMChain

from third_parties.linkedin import scrap_linkedin_profile

if __name__ == "__main__":
    load_dotenv()
    print("LangChain")
    
    summary_template = """
    given the Linkedin information {information} about a person I want to create:
    1. A short summary
    2. two interesting facts about them
    """
    summary_prompt_template = PromptTemplate(input_variables=["informations"],template=summary_template)
    # llm = ChatOpenAI(temperature=0,model_name="gpt-3.5-turbo")
    llm = Ollama(model="gemma")
    chain = LLMChain(llm=llm,prompt=summary_prompt_template)
    information = scrap_linkedin_profile("hello")
    res = chain.invoke(input={"information":information})
    print(res)