from graph import graph


if __name__ == "__main__":
    state = graph.invoke({"question": "What should I prepare for interview?"})
    print(state["generated_answer"])