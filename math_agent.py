"""Basic LangChain math tool for the Oracle agent lab.

Install the dependency in your lab environment:
    python -m pip install langchain

Run the addition tool locally (no Oracle credentials needed):
    python math_agent.py

After configuring your lab's tool-capable chat model, use:
    from math_agent import ask_math
    print(ask_math(model, "What is 12.5 plus 7.5?"))

The Oracle model and authentication setup stay in your lab notebook.
Reference: https://docs.langchain.com/oss/python/langchain/quickstart
"""

from langchain.agents import create_agent
from langchain.tools import tool


# STEP 1: Define the math tool.
@tool
def add(a: float, b: float) -> float:
    """Add two numbers together. Use for addition operations.

    Args:
        a: The first number.
        b: The second number.
    """
    return a + b


# STEP 2: Supply the model already configured in your Oracle lab.
# STEP 3: Create the agent and give it the addition tool.
def build_math_agent(model):
    """Build a math agent using your lab's configured chat model."""
    return create_agent(
        model=model,
        tools=[add],
        system_prompt=(
            "You are a helpful math assistant. "
            "Always use the add tool for addition. "
            "Give a concise answer."
        ),
    )


# STEP 4: Send a question to the agent.
def ask_math(model, question: str):
    """Ask the agent a question and return its final response content."""
    agent = build_math_agent(model)
    result = agent.invoke({
        "messages": [{"role": "user", "content": question}]
    })
    return result["messages"][-1].content


if __name__ == "__main__":
    # This tests the tool directly; it does not call a language model.
    answer = add.invoke({"a": 12.5, "b": 7.5})
    print(f"Addition tool: 12.5 + 7.5 = {answer}")
    print("To run the agent, call ask_math(model, question) from your lab.")
