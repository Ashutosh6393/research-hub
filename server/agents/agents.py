from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from tools.scrape_url import scrape_url
from tools.web_search import web_search

load_dotenv()

llm = ChatAnthropic(model="claude-sonnet-5-5")


def search_agent():
    return create_agent(model=llm, tools=[web_search])


def reader_agent():
    return create_agent(model=llm, tools=[scrape_url])


writer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert researcher and writer. You are given a topic and research gathered. You need to write a detailed research report on the topic.",
        ),
        (
            "human",
            """Write a detailed research report on the topic below
    
        topic: {topic}

        Research gathered:
        {research}

        structure the report as:
        - Introduction
        - Key Findings
        - Conclusion

        Be detailed, factual and professional.
    
    """,
        ),
    ]
)


writer_chain = writer_prompt | llm | StrOutputParser()


# critic_chain

critic_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are sharp and constructive research critic that critiques the given text. Be honest and specific."),
    ("human", """Review the following research report and provide feedback on its quality and accuracy:
    

    report:

    {text}
    
    Be detailed, factual and professional.

    respond in this exact format:

    Score: X/10

    strengths:
    - ...
    - ... 

    one line verdict: 
    ... 
    """),
])


critic_chain = critic_prompt | llm | StrOutputParser()