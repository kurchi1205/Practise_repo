from typing import TypedDict, Literal, List

class MenuState(TypedDict):
    menu: List
    risk_factor: Literal["low", "medium", "high"] = ""
    reason: str = ""
    nutrition: dict = {}
    risk_score: int = 0
    revised_menu: List = []

    
