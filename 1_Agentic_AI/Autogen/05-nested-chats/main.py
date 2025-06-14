"""
This script sets up a nested chat system involving three agents: a writer, a user proxy, and a critic.
The writer generates engaging content, the critic ensures compliance with guidelines, and the user
proxy handles the interaction between them, simulating a user's feedback process.
"""

import autogen

# Load configuration list from JSON file
config_list = autogen.config_list_from_json(env_or_file="OAI_CONFIG_LIST.json")
llm_config = {"config_list": config_list}

# Define the task for the writer agent
task = """Write a concise but engaging blogpost about Meta."""

# Set up the Writer assistant agent
writer = autogen.AssistantAgent(
    name="Writer",
    llm_config=llm_config,
    system_message="""
    You are a professional writer, known for your insightful and engaging articles.
    You transform complex concepts into compelling narratives.
    You should improve the quality of the content based on the feedback from the user.
    """
)

# Set up the UserProxy agent to simulate user feedback
user_proxy = autogen.UserProxyAgent(
    name="User",
    human_input_mode="NEVER",
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
    code_execution_config={
        "last_n_messages": 1,
        "work_dir": "my_code",
        "use_docker": False,
    }
)

# Set up the Critic assistant agent to ensure compliance
critic = autogen.AssistantAgent(
    name="Critic",
    llm_config=llm_config,
    system_message="""
    You are a critic, known for your thoroughness and commitment to standards.
    Your task is to scrutinize content for any harmful elements or regulatory violations, ensuring
    all materials align with required guidelines.
    """
)

def reflection_message(recipient, messages, sender, config):
    """
    Generates a reflection message summarizing the latest chat content.
    
    Args:
        recipient (AssistantAgent): The agent receiving the reflection message.
        messages (list): The chat messages to be reflected upon.
        sender (User): The agent sending the reflection message.
        config (dict): Configuration used for reflection.
    
    Returns:
        str: The content of the reflection message.
    """
    print("Reflecting...")
    return f"Reflect and provide critique on the following writing. \n\n {recipient.chat_messages_for_summary(sender)[-1]['content']}"

user_proxy.register_nested_chats(
    [
        {
            "recipient": critic,
            "message": reflection_message,
            "summary_method": "last_msg",
            "max_turns": 1
        }
    ],
    trigger=writer
)

# Initiate the chat between the user proxy and the writer
user_proxy.initiate_chat(recipient=writer, message=task, max_turns=2, summary_method="last_msg")











