"""
This script loads posts from Reddit using RedditPostsLoader, processes them with autogen (a large language model),
and extracts specific content for a newsletter formatted in Markdown. Comments and docstrings added for clarity.
"""

from langchain_community.document_loaders.reddit import RedditPostsLoader
import autogen

def main():
    """
    Main function to load Reddit posts and extract specific content for a newsletter.
    Follows PEP-8 standards and includes comments for better understanding.
    """
    # Initialize RedditPostsLoader with client credentials and configuration.
    loader = RedditPostsLoader(
        client_id="-client-id",  # Your Reddit API client ID
        client_secret="-secret-key",  # Your Reddit API secret key
        user_agent="extractor by u/tyler_programming",  # User agent for the Reddit API
        categories=["new"],  # Category of posts to fetch ("controversial", "hot", "new", "rising", "top")
        mode="subreddit",  # Mode for Reddit posts search (could be subreddit)
        search_queries=["openai"],  # Subreddits to fetch posts from
        number_posts=3,  # Number of posts to fetch (default is 10)
    )

    # Load documents from Reddit posts
    documents = loader.load()

    # Load configuration list for the autogen agent
    config_list = autogen.config_list_from_json(env_or_file="OAI_CONFIG_LIST.json")
    llm_config = {"config_list": config_list, "seed": 45}

    # Initialize AssistantAgent for writing output
    writer = autogen.AssistantAgent(
        name="Writer",
        llm_config=llm_config,
        system_message=(
            """
            You won't change the information given, just parse the page_content from the reddit post.
            No code will be written.
            """
        ),
    )

    # Initialize UserProxyAgent to manage interaction
    user_proxy = autogen.UserProxyAgent(
        name="User",
        human_input_mode="NEVER",  # Set to never allow human input
        max_consecutive_auto_reply=3,  # Maximum number of consecutive auto-replies
        code_execution_config=False  # Code execution is disabled
    )

    # Initiate chat with the AssistantAgent to generate formatted Markdown newsletter
    user_proxy.initiate_chat(
        recipient=writer,
        message=(
            """
            I need you to extract the page_content and url from each of {documents}, with each document extracted
            separately from each other. Make sure this is formatted with Markdown. Get it ready for an email, but don't 
            add or change what is in the documents. Make sure to use the FULL page_content from the document.

            Create a newsletter from this information with:

            [Newsletter Title Here] - make sure to create a catchy title

            The format for markdown should be:

            Title of the document
            The Page Content
            The Author
            The url
            """
        ),
        max_turns=2,  # Maximum number of turns between user and assistant
        summary_method="last_msg"  # Method to summarize the conversation
    )

if __name__ == "__main__":
    main()