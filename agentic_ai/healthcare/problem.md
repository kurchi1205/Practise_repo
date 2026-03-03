Build an autonomous multi-agent system that evaluates whether a proposed school cafeteria menu is nutritionally risky for children, using:

LLM reasoning

Tool calling

Planner agent

Validator agent

Retry loop


1. retrieves the nutrition in gram (parallely for all the items in the menu)
2. evaluator evaluates the results -> classifies it into low medium and high
3. voting agent -> based on the evaluator comes up with a risk score(1 - 10) and a reasoning
4. if the risk score is high then it forwards to a revised menu generator agent else gives output response.
5. revised menu generator takes in the original menu and thr risk reasoning -> then for that it comes up with a new plan, checks the mutrition of the plan and if it is still risky then, retries unless it gets a new plan.



