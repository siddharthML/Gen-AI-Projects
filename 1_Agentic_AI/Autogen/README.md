# Autogen Code

Taken from the following channel:

[Autogen Full Beginner Course](https://www.youtube.com/watch?v=JmjxwTEJSE8)

---

# 📘 Autogen Multi-Agent Project Report

This document provides a structured summary of the foundational Autogen agent patterns covered in the video tutorial. Each section corresponds to a directory in the project and describes its key ideas, architecture, and notable implementation points.

---

## 🔹 01-twoway-chat

**📂 Directory:** `01-twoway-chat`

### 📝 Summary

This section demonstrates the basics of a two-way multi-agent chat using Autogen. The setup involves a user agent and an AI assistant agent. The assistant agent is configured to interact with OpenAI's GPT-3.5 Turbo model via API key stored in a config file. The user proxy agent can execute code (optionally using Docker), and their interaction is controlled through the `human_input_mode` parameter, allowing for automatic or manual feedback after each step.

### 🧠 Key Concepts

* Single-agent vs. multi-agent frameworks.
* Configuration via a separate JSON file for API keys and model.
* Modes for user input: `"never"`, `"always"`, and `"terminate"` allow for various levels of interaction and feedback.
* Example task: plotting a chart for Meta and Tesla stock price changes, with automatic error correction and package installation by the agents.
* Agents save code to a working directory for review.

### ✅ Sample Tasks

* Change agents' feedback behavior by altering `human_input_mode`.
* Allow the user to provide corrections after each step or only at the end.

---

## 🔹 02-groupchat

**📂 Directory:** `02-groupchat`

### 📝 Summary

This section introduces group chat capabilities, where multiple agents collaborate to solve a problem. Agents can be assigned roles (e.g., planner, engineer, critic, scientist), and a `GroupChatManager` oversees their coordination. The setup reads configuration from an `.env` file, supporting API key management.

### 🧠 Key Concepts

* Multiple agent roles for better division of tasks.
* Use of system messages to instruct agents on their purpose.
* Caching of results via a cache directory and seed, enabling reproducibility.
* Group chat manager prevents infinite loops with a max round limit.
* Example project: collaboratively searching for LLM research papers and creating a markdown table.

> ⚠️ Outcomes may vary; not all runs produce successful results due to model behavior or timeouts.

---

## 🔹 03-snake

**📂 Directory:** `03-snake`

### 📝 Summary

A project challenge to generate a classic Snake game using Autogen agents. This exercise illustrates agent orchestration for software creation and iteration.

### 🧠 Key Concepts

* Directory structure: separate files for configuration and main logic.
* Recommended to add a `# t filename.py` comment inside code blocks to help agents save generated code correctly.
* Multiple agents: e.g., a coder and a product manager, or a critic to review the output.
* Iterative improvement: changing the cache seed and re-running to achieve a better or correct game implementation.
* Human input can be minimized for automated code creation, or enabled for manual correction.

> 💡 Tip: If the generated game is incorrect, try adjusting the prompt, agents' roles, or the model seed.

---

## 🔹 04-sequence\_chat

**📂 Directory:** `04-sequence_chat`

### 📝 Summary

This section covers sequential agent chats, where multiple agents process tasks in a fixed order, passing context between each other. The transcript's example uses three assistant agents to create and iterate on literary quotes.

### 🧠 Key Concepts

* Sequential execution: agents are activated one after the other.
* Use of context (reflection): previous chat output is summarized and passed to the next agent for continuity.
* Filtering configurations: choose which model to use for each agent.
* Custom termination logic via lambda functions.
* Practical usage: automating a multi-stage reasoning pipeline (e.g., information gathering, synthesis, and critique).

### 💬 Example

1. Agent 1 generates a quote.
2. Agent 2 creates another quote, aware of Agent 1's output.
3. Agent 3 synthesizes a new quote based on the previous two.

---

## 🔹 05-nested-chats

**📂 Directory:** `05-nested-chats`

### 📝 Summary

This section explains nested chats, where the response of one agent triggers a sub-conversation with another agent before the original conversation continues. The classic use-case shown is a writing task: after a "writer" agent produces an output, a "critic" agent is triggered to review it, before the result is finalized.

### 🧠 Key Concepts

* Triggers: configure which agent response initiates a nested chat.
* Reflection functions: inject specific instructions and context into the nested chat.
* Use-case: enables inner monologues, reviews, or stepwise refinement without overloading the main group chat.
* Practical difference from group chats: Nested chats are not part of the main chat loop, but occur between specific agents at specified points.

---

## 🔹 06-logging

**📂 Directory:** `06-logging`

### 📝 Summary

This section presents Autogen's built-in logging for analyzing agent performance and API usage.

### 🧠 Key Concepts

* Logging setup: start and stop logging around a chat session, storing results in a SQLite database.
* Data collected: tokens used, API call cost, runtime duration, etc.
* Usage: analyze model efficiency, compare costs, and optimize workflow.
* Data retrieval: use SQLite browser or pandas in Python to inspect logs.
* Useful for benchmarking between OpenAI models and local LLMs.


