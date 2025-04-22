from dotenv import load_dotenv

from pydantic import BaseModel
from langchain_openai import ChatOpenAI

# from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor

from tools import search_tool, wiki_tool, save_tool


class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]


def main():
    load_dotenv()

    # Define the temperature parameter for the model.
    temperature = 0.02
    # Define the model type.
    model = "gpt-4.1-mini"
    llm = ChatOpenAI(model=model, temperature=temperature)
    parser = PydanticOutputParser(pydantic_object=ResearchResponse)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
            You are a research assistant that will help generate a research paper.
            Answer the query and use the necessary tools
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
            ),
            ("placeholder", "{chat_history}"),
            ("human", "{query}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    ).partial(format_instructions=parser.get_format_instructions())

    tools = [search_tool, wiki_tool, save_tool]
    agent = create_tool_calling_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
    )

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    query = input("What can I help you reserach? ")
    raw_response = agent_executor.invoke({"query": query})
    try:
        structured_response = parser.parse(raw_response.get("output"))
        print("Structured Response:", structured_response)
    except Exception as e:
        print("Error parsing response:", e, "Raw Response: ", raw_response)


if __name__ == "__main__":
    main()
