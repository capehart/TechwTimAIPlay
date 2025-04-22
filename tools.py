from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime


search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="search",
    func=search.run,
    description="Search the web for information",
)


api_wrapper = WikipediaAPIWrapper(
    top_k_results=1,
    doc_content_chars_max=100,
    lang="en",
)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)


def save_to_txt(data: str, filename: str = "research_output.txt") -> str:
    """
    Save the data to a text file with the current date and time in the filename.
    """
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n--- End of Output ---"

    with open(filename, "a", encoding="utf-8") as file:
        file.write(formatted_text)
    return f"Data saved to {filename}"


save_tool = Tool(
    name="save_to_txt",
    func=save_to_txt,
    description="Save the research output to a text file with a timestamp in the contents.",
)
