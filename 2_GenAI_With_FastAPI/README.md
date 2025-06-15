This project will cover the code and implementation details on how to serve GenAI appliactions with FastAPI.

It is based on the following book:

![book cover](https://learning.oreilly.com/library/cover/9781098160296/250w/)

I've changed the code where required for my own reference.

Below is a chapter wise summary of the book. Detailed chapter breakdown will be in each folder.
---

# 📘 Generative AI with FastAPI

---

## Chapter 1: Introduction

This chapter introduces **Generative AI (GenAI)** and its growing role in application development. GenAI is a subset of machine learning that creates new content by mimicking training data patterns.

* **Example:** A model trained on butterfly images can generate novel images with variation using random noise during sampling.

### Key Generative Models:

* **Diffusion Models:** Add/remove noise incrementally.
* **Transformers:** Use self-attention to model sequences (e.g., ChatGPT).

GenAI models can be:

* **Unimodal:** Text, audio, image, video, point clouds, 3D meshes.
* **Multimodal:** E.g., GPT-4o (OpenAI).

---

### Capabilities of GenAI:

* **Creativity Facilitation:** Tools like DALL-E 3 visualize complex descriptions.
* **Contextual Solutions:** Rich prompts yield precise results.
* **Personalization:** Natural language chatbots (e.g., education, travel).
* **Interface to Systems:** Assist users with complex systems (guardrails needed).
* **Admin Automation:** E.g., document processing.
* **Content Scaling:** Generate content quickly (e.g., blog posts).

---

### GenAI Services Architecture:

* Wrap GenAI models with an **API** behind an **HTTP server** (e.g., FastAPI).
* Responsibilities:

  * Enrich prompts
  * Validate output
  * Interface with APIs/DBs
  * Serve users

**FastAPI** is selected for:

* Speed, async support
* Data validation, type safety
* Auto docs (OpenAPI)
* AI-model serving support

---

### Challenges to Adoption:

* **Inaccuracy & Hallucination:** Limits in sensitive domains (e.g., healthcare).
* **Privacy & Security Risks**
* **Mitigations:** Fine-tuning, prompt optimization, software practices.

---

### Capstone Project:

A GenAI service built with FastAPI that:

* Integrates LLMs, audio, vision models
* Uses RAG
* Interacts with internal/external APIs & DBs
* Supports authentication, authorization, and moderation

---

## Chapter 2: Getting Started with FastAPI

**FastAPI** is an **ASGI web framework** ideal for APIs and model-serving.

### Development Setup:

* Install: `Python 3.11`, `FastAPI`, `Uvicorn`, `OpenAI`

### FastAPI Basics:

* Create `main.py` with minimal code and health endpoint (`/`).
* Docs auto-generated at `/docs`.

---

### Key Features:

* **Flask-like routing**
* **Async + Sync handling**
* **Custom Middleware & CORS**
* **Customizable Layers**
* **Pydantic-based validation**
* **Auto-generated Docs**
* **Dependency Injection**
* **Startup/Shutdown Events**
* **Security Tools**
* **WebSocket & GraphQL support**
* **Modern IDE integration**

---

### Project Structures:

1. **Flat:** Simple files at root – for microservices
2. **Nested:** Grouped modules – for larger projects
3. **Modular:** Domain-based encapsulation – for scalability

---

### Layered Architecture:

* **Router → Controller → Service → Repository → Model/Schema**
* Cross-cutting: Middleware, Dependencies, Pipes, Mappers, Guards

---

### FastAPI vs Other Frameworks:

* Compared with **Django** (monolithic) and **Flask** (minimalistic)

### Limitations:

* **Inefficient GPU management**
* **Dependency Conflicts**
* **Global Interpreter Lock (GIL)**

---

### Recommended Tools:

* **Package Managers:** `Poetry`, `Conda`
* **Linters & Formatters:** `Flake8`, `Black`, `Ruff`
* **Security & Typing:** `Bandit`, `Mypy`, `Pylance`

---

## Chapter 3: AI Integration and Model Serving

### Transformer Architecture:

* Replaces RNNs for better long-range dependency handling
* Self-attention → efficient parallelism on GPUs

---

### LLM Inference Steps:

* **Tokenization → Embedding → Positional Encoding → Autoregressive Prediction**

---

### TinyLlama + FastAPI Example:

* Uses Hugging Face pipeline
* Parameters: `max_new_tokens`, `temperature`, `top_k`, `top_p`

---

### Model Serving Strategies:

1. **Load on every request** – simple but slow
2. **Preload in lifespan events**
3. **Externalize via BentoML or vLLM**

---

### Monitoring:

* Custom middleware logs requests/responses

---

### Other Models:

* **Audio:** Bark (4-model chain)
* **Vision:** Stable Diffusion, LoRA tuning
* **Video:** OpenAI’s Sora, Latte (open-source)
* **3D:** Mesh-based models like Shap-E

---

## Chapter 4: Implementing Type-Safe AI Services

### Type Safety:

* **Annotations** enforce correctness and aid IDEs/static tools
* Python is dynamically typed → `mypy` for static checks

---

### Tools:

* **`Annotated`**: Attach metadata
* **Dataclasses:** Good for simple structures
* **Pydantic Models:** Best for APIs

---

### Pydantic Features:

* Compound models, validators, field constraints
* Computed fields (`@computed_field`)
* Serialization (`model_dump()`)

---

### Settings Management:

* Use `BaseSettings` (via `pydantic-settings`) to load `.env` files

---

## Chapter 5: Achieving Concurrency in AI Workloads

### Concepts:

* **Concurrency:** Multiple tasks overlapping
* **Parallelism:** Tasks on multiple cores

---

### FastAPI Support:

* Async/await for I/O tasks
* Multiprocessing for CPU-bound tasks

---

### Projects:

* **Web Scraper:** Async scraping with `aiohttp`, `BeautifulSoup`
* **RAG System:**

  1. Extract (e.g., `pypdf`)
  2. Transform (clean, embed)
  3. Store (vector DB)
  4. Retrieve (semantic search)
  5. Generate (LLM)

---

### Optimization:

* Use **external AI servers** (e.g., vLLM)
* Techniques: **Batching, Paged Attention**

---

## Chapter 6: Real-Time Communication with Generative Models

### Communication Models:

* **HTTP (stateless)**
* **Short Polling / Long Polling**
* **SSE:** One-way, persistent
* **WebSocket:** Bidirectional, full-duplex

---

### Implementing:

* **SSE:** For LLM response streaming
* **WebSocket:** `WebSocketConnectionManager` for chat-like use cases

---

### Design Advice:

* Use **single streaming entry point**
* **Throttling:** Control throughput (`asyncio.sleep`)

---

## Chapter 7: Integrating Databases into AI Services

### When to Use:

* Need persistence, retrieval, concurrency

---

### Types:

* **SQL:** Postgres
* **NoSQL:** Redis, MongoDB, Qdrant (vector), Neo4j, etc.

---

### Project: Store LLM Conversations in Postgres

* Tools: `sqlalchemy`, `alembic`
* ORM models → SQL tables
* Session management via FastAPI dependencies

---

### CRUD APIs:

* Pydantic models separate from ORM
* Repository & service pattern used

---

### Alembic for Migrations:

* Manage schema changes collaboratively

---

### Real-Time Stream Storage:

* Use background tasks to avoid blocking

---

## Chapter 8: Authentication and Authorization

### Authentication:

* **Basic:** Simple, insecure
* **JWT:** Secure, stateless, used with `python-jose`
* **OAuth:** Delegated access via external identity providers (e.g., Google)

---

### Authorization Models:

* **RBAC:** Based on user roles
* **ReBAC:** Relationship-based
* **ABAC:** Attribute-based
* **Hybrid:** Combine for complex systems

---

### Best Practices:

* Separate auth logic into external services (e.g., **Permify**, **Oso**)

---

## Chapter 9: Securing AI Services

### Guardrails:

* **Input Guardrails:** Topic classification, prompt injection, moderation
* **Output Guardrails:** Fact-checking, toxicity filters, syntax validation

---

### Moderation Example:

* G-Eval scoring with LLM evaluators
* Optimizations: selective, async, sampled

---

### Rate Limiting:

* **Token Bucket**, **Leaky Bucket**, **Fixed/Sliding Window**
* `slowapi` or `fastapi-limiter`
* Redis as backend for distributed rate limits

---

### Stream Throttling:

* `asyncio.sleep` between chunks
* Traffic shaping tools like `tc` (Linux)

---

## Chapter 10: Optimizing AI Services

### Performance:

* **Batch APIs**: Asynchronous, cost-efficient
* **Caching:**

  * Keyword (exact match)
  * Semantic (embedding similarity)
  * Prompt (token reuse)

---

### Quality:

* **Structured Outputs**
* **Prompt Engineering:**

  * Role, Context, Task
  * Techniques:

    * In-context
    * CoT, ToT
    * Plan & Solve
    * Ensembling
    * Self-critique
    * Agentic tools

---

### Fine-Tuning:

* Customizing models to domain
* Format: JSONL
* Evaluate output quality vs. cost

---

## Chapter 11: Testing AI Services

### Types of Testing:

* **Unit → Integration → E2E**
* **Static Checks:** `mypy`, `ruff`
* **Shift-Left & TDD**
* **Test Dimensions:** Scope, Coverage, Comprehensiveness

---

### Test Strategies:

* **Testing Pyramid**, **Trophy**, **Honeycomb**
* **GWT Pattern:** Given-When-Then

---

### Challenges:

* **Non-deterministic outputs**
* **Expensive model runs**
* **Model regressions**

---

### Behavioral Testing:

* **MFTs:** Basic sanity checks
* **ITs:** Invariance under small changes
* **DETs:** Output changes with input changes
* **Auto-evaluation:** Use LLMs to score outputs

---

### Project: Test a RAG Pipeline

* `pytest`, `pytest-asyncio`, `pytest-mock`
* Fixtures, Parameterization, Async Tests
* E2E: Vertical (feature) and Horizontal (scenario)

---

## Chapter 12: Deployment of AI Services

### Deployment Options:

* **VMs:** Isolated, heavy
* **Serverless:** Auto-scaling, cold-start risk
* **PaaS:** Easy, less control
* **Containers:** Best balance

---

### Docker Overview:

* **Dockerfile → Image → Container**
* Layers → Caching → Immutability
* Use `docker init` for best-practice setup

---

### Storage:

* **Volumes**, **Bind Mounts**, **tmpfs**
* Use non-root user for security

---

### Networking:

* **Bridge (default)**, **Host**, **IPVlan**, **None**
* User-defined networks recommended

---

### Docker Compose:

* Define multi-container systems via `compose.yaml`
* Add GPU access if needed
* Use `watch` for hot reloads

---

### Optimization:

* **Slim base images (Alpine)**
* **Externalize model data**
* **Multi-stage builds**
* **.dockerignore** for faster builds

---

**You're now fully equipped to build robust, scalable, real-time GenAI services with FastAPI.** 🚀

---
