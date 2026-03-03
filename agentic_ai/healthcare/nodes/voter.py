import sys
sys.path.append("../")
from load_model import load_nutri_model
from state import MenuState


def voter(state: MenuState):
    print(state)
    llm = load_nutri_model()
    SYSTEM_PROMPT = """Based on the extracted nutrition information and the risk factor, rate the menu on a risk scale of 1 to 10 with proper reasoning: 
    give the result in the following format {
        "risk_score": risk_score,
        "reason": reason
    }
    
    """
    nutri_info = {
        "nutrition": state['nutrition'],
        "risk_factor": state['risk_factor']
    }
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.append({"role": "user", "content": f" nutri_info: {nutri_info}"})
    response = llm.invoke(messages)

    state['risk_score'] = response["risk_score"]
    state['reason'] = response["reason"]

    return "The menu is classified as " + state['risk_score'] + " with reason " + state['reason']