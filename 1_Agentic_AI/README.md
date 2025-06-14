In this repo I have the code for two agentic AI frameworks:

1) Crew AI
2) Autogen

----
# Overview:

## 🤖 CrewAI vs Microsoft AutoGen

This table outlines the key differences between **CrewAI** and **Microsoft AutoGen**—two leading Python frameworks for building multi-agent systems powered by LLMs. Both frameworks aim to enable autonomous agent collaboration, but they differ in architecture, abstraction, and flexibility.

---

## 🧩 Comparison Table

| **Parameter**          | **CrewAI**                                                     | **Microsoft AutoGen**                                                          |
| ---------------------- | -------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| **Origin & Backing**   | Independently developed, open-source                           | Developed by Microsoft Research                                                |
| **Agent Abstraction**  | Agents have **roles**, **goals**, and **backstories**          | Agents are function-driven **actors** engaged in dialogue-style workflows      |
| **Task Design**        | Uses explicit `Task` objects with descriptions and outputs     | Conversation-based model where agents exchange messages to drive tasks         |
| **Coordination Model** | Agents coordinated through a `Crew` executing a **task list**  | Message-passing **conversation graphs** between agents                         |
| **Delegation**         | Agents can delegate to others (if allowed)                     | Agents can invoke each other as **functions** or through messaging             |
| **Tool Integration**   | Built-in tools (e.g., scraping, search); supports custom tools | Any Python function can be registered as a **tool** or **function call**       |
| **Memory/State**       | Optional memory to persist context across tasks                | Maintains **context** via message history                                      |
| **Prompting/LLM Use**  | Structured prompts based on roles and goals                    | Function-call prompts with role-specific customization                         |
| **Extensibility**      | Extensible by defining new agents, tools, and tasks            | Highly flexible; agents can be **pure Python functions or modules**            |
| **LLM Support**        | Supports OpenAI, HuggingFace, Cohere, Mistral, and others      | OpenAI, Azure, HuggingFace, local LLMs, or custom endpoints                    |
| **Sync/Async Support** | Synchronous execution model (as of June 2024)                  | Supports both **synchronous** and **asynchronous** agent execution             |
| **Observability/Logs** | Logs via `verbose` mode and task outputs                       | Detailed **message logs**, reasoning traces, and conversation history          |
| **Community & Docs**   | Small but growing; straightforward documentation               | Larger community, rich documentation, and active GitHub development            |
| **Typical Use Cases**  | Educational demos, workflow automation, content generation     | Advanced assistant orchestration, research workflows, complex multi-agent chat |
| **License**            | Apache 2.0                                                     | MIT License                                                                    |

---

## ✨ Summary

* **CrewAI** provides a high-level interface for defining agents and coordinating them in structured task flows. It excels in educational and business workflow automation use cases, with an emphasis on clarity and simplicity.
* **AutoGen** offers a more flexible, low-level interface suited for advanced users building deeply interactive and reactive multi-agent systems. It shines in use cases requiring asynchronous conversations and complex dependency graphs.

---

> 💡 **Use CrewAI** if you're looking for structured workflows, clear abstractions, and faster prototyping.
> 💡 **Use AutoGen** if you need full control over agent communication, async workflows, and conversation modeling.

---

### I might expand this repo further with chatdev, metagpt, and taskweaver.



