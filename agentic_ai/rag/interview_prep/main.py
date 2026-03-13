from graph import graph

async def main():
    state = await graph.ainvoke({"question": "Tell about AON culture"})
    print(state["generated_answer"])

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())