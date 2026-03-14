from graph import graph
from langfuse import get_client
from langfuse.langchain import CallbackHandler

langfuse = get_client()

async def main():
    langfuse_handler = CallbackHandler()
    state = await graph.ainvoke(
        {"question": "How to answer in aon interviews?"},
        config={"callbacks": [langfuse_handler]},
    )
    print(state["generated_answer"])

    scores = state.get("scores", {})
    print("Scores:", scores)

    trace_id = langfuse_handler.last_trace_id
    for metric_name, value in scores.items():
        numeric_value = value if isinstance(value, float) else (1.0 if value == "pass" else 0.0)
        langfuse.create_score(
            trace_id=trace_id,
            name=metric_name,
            value=numeric_value,
            data_type="NUMERIC",
            comment=str(value),
        )

    langfuse.flush()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())