from graph import graph
from state import MenuState

if __name__ == "__main__":
    menu = [
        "pizza", "sandwich", "salad", "soup", "steak", "tacos", "veggie", "wrap"
    ]
   
    print(graph.invoke({"menu": menu}))
