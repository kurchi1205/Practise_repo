from graph import graph

async def main():
    state = await graph.ainvoke({"question": "What is today's weather is SIngapore?"})
    print(state["generated_answer"])

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())