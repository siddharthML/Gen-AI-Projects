"""
This script demonstrates setting up logging for interactions between 
an assistant agent and a user proxy agent in a chat application. 
It also provides a mechanism to query and process the logged data 
using SQLite and pandas, ultimately printing the processed data 
in a user-friendly format.
"""

import json
import pandas as pd
import sqlite3
import autogen

# Configure the assistant agent using the llm_configuration
llm_config = {
    "config_list": autogen.config_list_from_json(env_or_file="OAI_CONFIG_LIST.json"),
    "temperature": 0
}

# Start a logging session and print the session ID
logging_session_id = autogen.runtime_logging.start(config={"dbname": "logs.db"})
print("Started Logging session ID: " + str(logging_session_id))

# Create an assistant agent and user proxy agent for automated chat interactions
assistant = autogen.AssistantAgent(name="assistant", llm_config=llm_config)
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    code_execution_config=False,
    human_input_mode="NEVER",
    is_termination_msg=lambda msg: "TERMINATE" in msg["content"],
)

# Initiate a chat with the assistant
user_proxy.initiate_chat(
    assistant, message="What is the height of the Sears Tower? Only respond with the answer and terminate"
)

# Stop the logging after the chat session ends
autogen.runtime_logging.stop()

def get_log(dbname="logs.db", table="chat_completions"):
    """
    Fetch log data from the specified SQLite database and table.

    Args:
        dbname (str): The name of the database file.
        table (str): The name of the table to query.

    Returns:
        List[Dict]: A list of dictionaries containing log data.
    """
    con = sqlite3.connect(dbname)
    query = f"SELECT request, response, cost, start_time, end_time from {table}"
    cursor = con.execute(query)
    rows = cursor.fetchall()
    column_names = [description[0] for description in cursor.description]
    data = [dict(zip(column_names, row)) for row in rows]
    con.close()
    return data

def str_to_dict(s):
    """
    Convert a string representation of a dictionary into an actual dictionary.

    Args:
        s (str): String representation of a dictionary.

    Returns:
        dict: Parsed dictionary object.
    """
    return json.loads(s)

# Fetch the log data using the `get_log` function
log_data = get_log()
log_data_df = pd.DataFrame(log_data)

# Process the log data and extract relevant information
log_data_df["total_tokens"] = log_data_df.apply(
    lambda row: str_to_dict(row["response"])["usage"]["total_tokens"], axis=1
)

log_data_df["request"] = log_data_df.apply(
    lambda row: str_to_dict(row["request"])["messages"][0]["content"], axis=1
)

log_data_df["response"] = log_data_df.apply(
    lambda row: str_to_dict(row["response"])["choices"][0]["message"]["content"], axis=1
)

# Print the processed log data to the terminal
print(log_data_df)