import autogen

def main():
    """
    Entry point of the program. Initializes the assistant and user proxy agents, 
    then starts a chat interaction with a request to plot META and TESLA stock price change.
    """
    # Load configuration list from a JSON file
    config_list = autogen.config_list_from_json(
        env_or_file="OAI_CONFIG_LIST.json"
    )

    # Initialize the Assistant agent with the configuration list
    assistant = autogen.AssistantAgent(
        name="Assistant",
        llm_config={"config_list": config_list}
    )

    # Initialize the User Proxy agent with human input mode and code execution configuration
    user_proxy = autogen.UserProxyAgent(
        name="user",
        human_input_mode="ALWAYS",
        code_execution_config={
            "work_dir": "coding",
            "use_docker": False
        }
    )

    # Start a chat interaction with the assistant, requesting a plot for META and TESLA stock price change
    user_proxy.initiate_chat(assistant, message="Plot a chart of META and TESLA stock price change.")

if __name__ == "__main__":
    # Execute the main function
    main()