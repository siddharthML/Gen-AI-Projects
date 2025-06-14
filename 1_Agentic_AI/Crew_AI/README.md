This project is created from the materials of the course:

[Multi AI Agent Systems with Crew AI](https://www.deeplearning.ai/short-courses/multi-ai-agent-systems-with-crewai/)
---

# 🧠 Multi-Agent Workflows with CrewAI

This repository documents my exploration of building intelligent, autonomous multi-agent workflows using the [CrewAI](https://crewai.com) framework. The objective of this project is to simulate complex real-world tasks by orchestrating multiple AI agents that collaborate, delegate, and reason through a structured set of roles and responsibilities.

Each component of the project builds progressively—from defining basic agents to constructing full-scale agent teams capable of performing domain-specific tasks. The structure follows a modular breakdown, capturing various dimensions of multi-agent systems including task planning, tool usage, collaboration, and real-world applications.

---

## 🗂️ Project Structure

### 📁 `1_Create_Agents/`

**Notebook:** `research_write_article.ipynb`
Initial setup focusing on how to define agents with specific roles, goals, and backstories.

* Built a team of content generation agents: **Planner**, **Writer**, and **Editor**.
* Explored how role-context improves LLM performance in task execution.

---

### 📁 `2_Agent_Components/`

**Notebook:** `customer_support.ipynb`
Introduced core multi-agent design principles such as **delegation**, **cooperation**, and **role specialization**.

* Created a simulated customer support system.
* Implemented QA review as a second layer of agent validation.
* Introduced early concepts of memory and output constraints.

---

### 📁 `3_Tool_Use/`

**Notebook:** `tools_customer_outreach.ipynb`
Integrated external tools into agents’ workflows.

* Added search, scraping, and structured query capabilities.
* Demonstrated how agents interact with APIs and structured content dynamically.

---

### 📁 `4_Tasks/`

**Notebook:** `tasks_event_planning.ipynb`
Focused on creating well-defined, dependent tasks for agents to collaboratively execute.

* Developed an event planning crew including **Strategist**, **Logistics Manager**, and **PR Lead**.
* Emphasized coordination and handoff between agents across tasks.

---

### 📁 `5_Multi_Agent_Collaboration/`

**Notebook:** `collaboration_financial_analysis.ipynb`
Simulated analytical collaboration between agents handling financial data.

* Split responsibilities between agents for extracting KPIs, generating reports, and suggesting improvements.
* Combined reasoning with structured analysis using synthetic datasets and tool use.

---

### 📁 `6_Multi_Agent_System_Usecase/`

**Notebook:** `job_application_crew.ipynb`
Brought together all previously explored components into a full-scale use case.

* Built a system that analyzes job descriptions, tailors resumes, and drafts cover letters.
* Agents demonstrate contextual memory, iterative refinement, and real-time validation.

---

## 🔍 Key Elements Explored

* **Agent Architecture**: Defining roles, responsibilities, and inter-agent communication.
* **Task Structuring**: Designing modular, sequential tasks for effective delegation.
* **Tool Integration**: Connecting agents to real-world resources via search and scraping tools.
* **Memory and Feedback**: Implementing memory to preserve context across workflows.
* **Guardrails**: Applying constraints to align agent output with defined standards.

---

## ⚙️ Environment Setup

Dependencies used:

```bash
pip install crewai==0.28.8 crewai_tools==0.1.6 langchain_community==0.0.29
```

Ensure relevant API keys (e.g., OpenAI, Serper) are set in your environment. Utility scripts in each folder assist with key retrieval and setup.

---

## 📎 Additional Files

* `fake_resume.md`: Used as input in the job application scenario.
* `utils.py`: Shared utility functions for managing API keys and external integrations.

---

## 🧭 Purpose

This project was developed to deepen my understanding of agent-based architectures in the context of generative AI. It serves as a practical demonstration of how autonomous systems can be applied to various domains by coordinating roles, leveraging tools, and managing tasks across multiple agents.



