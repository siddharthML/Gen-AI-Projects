import autogen

def fetch_config_list():
    """Fetch the configuration list for the assistant agents.

    Returns:
        list: A list of configurations filtered by model name.
    """
    return autogen.config_list_from_json(
        env_or_file="OAI_CONFIG_LIST.json",
        filter_dict={"model": ["gpt-3.5-turbo"]}
    )

def create_assistant_agent(name, system_message, llm_config, max_consecutive_auto_reply=None):
    """Create an instance of AssistantAgent.

    Args:
        name (str): Name of the assistant agent.
        system_message (str): System message for the agent.
        llm_config (dict): LLM configuration dictionary.
        max_consecutive_auto_reply (int, optional): Max consecutive auto reply limit.

    Returns:
        AssistantAgent: Configured assistant agent instance.
    """
    return autogen.AssistantAgent(
        name=name,
        system_message=system_message,
        llm_config=llm_config,
        max_consecutive_auto_reply=max_consecutive_auto_reply
    )

def create_user_proxy_agent():
    """Create an instance of UserProxyAgent for initiating chats.

    Returns:
        UserProxyAgent: Configured user proxy agent instance.
    """
    return autogen.UserProxyAgent(
        name="user_proxy",
        is_termination_msg=lambda msg: msg.get("content") is not None and "TERMINATE" in msg["content"],
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        code_execution_config=False
    )

def initiate_chats(user_proxy):
    """Initiate chats with multiple assistants using the user proxy agent.

    Args:
        user_proxy (UserProxyAgent): The user proxy agent instance.
    """
    user_proxy.initiate_chats([
        {
            "recipient": assistant_quote1,
            "message": "give a quote from a famous author",
            "clear_history": True,
            "silent": False,
            "summary_method": "reflection_with_llm"
        },
        {
            "recipient": assistant_quote2,
            "message": "give another quote from a famous author",
            "clear_history": True,
            "silent": False,
            "summary_method": "reflection_with_llm"
        },
        {
            "recipient": assistant_create_new,
            "message": "based on the previous quotes, come up with your own!",
            "clear_history": True,
            "silent": False,
            "summary_method": "reflection_with_llm"
        }
    ])

if __name__ == "__main__":
    # Fetching the configuration list for the assistant agents
    config_list = fetch_config_list()

    # LLM configuration dictionary
    llm_config = {
        "config_list": config_list,
        "timeout": 120,
    }

    # Creating assistant agents with specific configurations
    assistant_quote1 = create_assistant_agent(
        name="assistant1",
        system_message="You are an assistant agent who gives quotes. Return 'TERMINATE' when the task is done.",
        llm_config=llm_config
    )

    assistant_quote2 = create_assistant_agent(
        name="assistant2",
        system_message="You are another assistant agent who gives quotes. Return 'TERMINATE' when the task is done.",
        llm_config=llm_config,
        max_consecutive_auto_reply=1
    )

    assistant_create_new = create_assistant_agent(
        name="assistant3",
        system_message="You will create a new quote based on others. Return 'TERMINATE' when the task is done.",
        llm_config=llm_config,
        max_consecutive_auto_reply=1
    )

    # Creating user proxy agent
    user_proxy = create_user_proxy_agent()

    # Initiating chats with the configured assistant agents
    initiate_chats(user_proxy)