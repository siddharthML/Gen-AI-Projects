"""
This module contains functions for getting the weather and time based on user input.
It also sets up two agents, `assistant` and `user_proxy`, which interact with each other
to execute tool calls and provide responses to queries.
"""

import autogen
from typing import Annotated
import datetime

def get_weather(location: Annotated[str, "The location"]) -> str:
    """
    Get the weather information based on the provided location.

    Parameters:
        location (str): The location to get the weather for.

    Returns:
        str: Weather information for the given location.
    """
    if location == "Florida":
        return "It's hot in Florida!"
    elif location == "Maine":
        return "It's cold in Maine!"
    else:
        return f"I don't know this place {location}"

def get_time(timezone: Annotated[str, "The timezone we are in"]) -> str:
    """
    Get the current time in the specified timezone.

    Parameters:
        timezone (str): The IANA timezone name (e.g., America/Los_Angeles).

    Returns:
        str: The current time in the specified timezone.
    """
    now = datetime.datetime.now()

    # Get the time with timezone information
    current_time = now.strftime("%Y-%m-%d %H:%M:%S %Z")

    return f"Current time in your timezone ({timezone}): {current_time}"

# Define the assistant agent that suggests tool calls.
assistant = autogen.ConversableAgent(
    name="Assistant",
    system_message=(
        "You are a helpful AI assistant. "
        "Return 'TERMINATE' when the task is done."
    ),
    llm_config={
        "config_list": [
            {
                "model": "gpt-4",
                "api_key": "sk-proj-1111",
            }
        ]
    }
)

# Define the user proxy agent used for interacting with the assistant agent
# and executing tool calls.
user_proxy = autogen.ConversableAgent(
    name="User",
    is_termination_msg=lambda msg: (
        msg.get("content") is not None and "TERMINATE" in msg["content"]
    ),
    human_input_mode="NEVER"
)

# Register the tool signature with the assistant agent.
assistant.register_for_llm(
    name="get_weather",
    description="Get the current weather for a specific location"
)(get_weather)

assistant.register_for_llm(
    name="get_time",
    description="The IANA time zone name, e.g. America/Los_Angeles"
)(get_time)

# Register the tool function with the user proxy agent.
user_proxy.register_for_execution(name="get_weather")(get_weather)
user_proxy.register_for_execution(name="get_time")(get_time)

# Initiate the conversation between the user proxy and the assistant.
user_proxy.initiate_chat(
    assistant,
    message="What time is it in Florida?"
)
