"""
This script sets up a ConversableAgent named 'Phil' using a comedian persona,
and initiates a conversation with the agent via a UserProxyAgent.
It also uses a locally hosted llm. Hosting is done through lm studio.
"""

import autogen


def main():
    """
    Main function to set up and initiate a conversation with the Phil agent.
    """
    # Configuration for the Phi2 model
    phi2 = {
        "config_list": [
            {
                "model": "TheBloke/phi-2-GGUF",
                "base_url": "http://localhost:1234/v1",
                "api_key": "lm-studio",
            },
        ],
        "cache_seed": None,
        "max_tokens": 1024
    }

    # Initialize the Phil agent with the given configuration and system message
    phil = autogen.ConversableAgent(
        "Phil (Phi-2)",
        llm_config=phi2,
        system_message="Your name is Phil and you are a comedian.",
    )

    # Create the agent that represents the user in the conversation.
    user_proxy = autogen.UserProxyAgent(
        "user_proxy",
        code_execution_config=False,
        default_auto_reply="...",
        human_input_mode="NEVER"
    )

    # Initiate a conversation with Phil by sending a joke request
    user_proxy.initiate_chat(phil, message="Tell me a joke!")


if __name__ == "__main__":
    main()