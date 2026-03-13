from graph import graph

async def main():
    state = await graph.ainvoke({"question": "How to answer in aon interviews?"})
    print(state["generated_answer"])

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())