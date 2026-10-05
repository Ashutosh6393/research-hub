from langchain_core.messages import ToolMessage

from agents.agents import (
    critic_chain,
    reader_agent,
    search_agent,
    writer_chain,
)


def search_tool_results(messages) -> str:
    parts = []
    for message in messages:
        if isinstance(message, ToolMessage):
            content = message.content
            parts.append(content if isinstance(content, str) else str(content))
    return "\n\n".join(parts)


def run_research_pipeline(topic: str) -> dict:
    state = {}

    ## step 1: search agent is working

    print("\n" + "-" * 50 + "\n")
    print("step 1: search agent is working")

    search = search_agent()
    search_result = search.invoke(
        {
            "messages": [
                (
                    "user",
                    f"Find recent, reliable and detailed information about: {topic}",
                )
            ]
        }
    )

    state["search_result"] = search_result["messages"][-1].content
    state["sources"] = search_tool_results(search_result["messages"])

    print("search result: \n", state["search_result"])
    print("search sources: \n", state["sources"])
    print("\n" + "-" * 50 + "\n")

    ## step 2: scraping agent is working
    print("step 2: scraping agent is working")

    reader = reader_agent()

    reader_result = reader.invoke(
        {
            "messages": [
                (
                    "user",
                    f"Pick the most relevant URL from the search tool results below and scrape it for deeper content about: '{topic}'. Use one of these URLs. Do not ask for links.\n\nSearch tool results:\n{state['sources']}",
                )
            ]
        }
    )

    # [-1] because the last message is the response from the reader agent
    state["reader_result"] = reader_result["messages"][-1].content

    print("reader result: \n", state["reader_result"])

    print("\n" + "-" * 50 + "\n")

    print("step 3: report writing agent is working")

    research_combined = f"SEARCH RESULTS: \n {state['search_result'][:800]} \n\n SCRAPED RESULTS: \n {state['reader_result']}"

    writer_result = writer_chain.invoke(
        {
            "topic": topic,
            "research": research_combined,
        }
    )

    state["research_report"] = writer_result
    print("final research report: \n", state["research_report"])

    print("\n" + "-" * 50 + "\n")

    ## step 4: critic agent is working

    print("step 4: critic agent is working")

    critic_result = critic_chain.invoke(
        {
            "text": state["research_report"],
        }
    )

    state["critic_result"] = critic_result

    print("final critic result: \n", state["critic_result"])

    print("\n" + "-" * 50 + "\n")

    return state
