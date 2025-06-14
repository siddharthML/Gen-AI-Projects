"""
This script initializes an OpenAI DALL-E model configuration
and generates an image based on a user request using AutoGen library.
"""

import autogen


def main():
    """
    Main function to configure agents and initiate a chat to create an image.

    It reads a JSON configuration file to configure the OpenAI model and sets up
    the assistant agent and user proxy agent to interact with each other.
    """
    # Load configuration list from JSON file, filtering for the DALL-E 3 model.
    config_list = autogen.config_list_from_json(
        env_or_file="OAI_CONFIG_LIST.json",
        filter_dict={"model": ["dall-e-3"]}
    )

    # Create the assistant agent using the loaded configuration for LLM.
    assistant = autogen.AssistantAgent("assistant", llm_config={"config_list": config_list})

    # Create the user proxy agent, configuring code execution settings.
    user_proxy = autogen.UserProxyAgent(
        "user_proxy",
        code_execution_config={"work_dir": "coding", "use_docker": False}
    )

    # Initiate a chat between the user proxy and assistant agent to create an image.
    user_proxy.initiate_chat(
        assistant,
        message="Create an image of a robot holding a sign saying 'AutoGen' with the background of wallstreet."
    )


if __name__ == "__main__":
    main()