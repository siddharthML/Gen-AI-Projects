
# Chapter 2: Getting Started with FastAPI

## Chapter Goals

This chapter aims to teach you:

* What FastAPI is.
* How to create and set up your own FastAPI project.
* FastAPI's features and advantages.
* Different ways to structure FastAPI projects.
* The onion/layered software design pattern.
* A comparison of FastAPI to other web frameworks.
* FastAPI's limitations.
* How to set up a managed Python environment and tooling for your project.

> By the end of this chapter, you should be comfortable using FastAPI, setting up projects, and justifying your technology stack decisions for building Generative AI (GenAI) services.

---

## 1. Introduction to FastAPI

* **Definition**: An ASGI framework for building high-performance APIs using Python.
* **Performance**: Comparable to Gin (Golang) and Express (Node.js), with full Python ecosystem support.
* **Core Features**: Swagger/OpenAPI docs, data validation, serialization via Pydantic.
* **Built on Starlette**: Lightweight and async-capable.
* **Popularity**: Fastest-growing Python web framework by downloads; second most starred on GitHub.
* **Comparison**:

  * **Flask**: Lightweight, but lacks schema validation.
  * **Django**: Full-stack, but less async-friendly.
  * **FastAPI**: Excellent for lean, async-first APIs with GenAI integration.

---

## 2. Setting Up Your Development Environment

### Installing Python, FastAPI, and Required Packages

* **Virtual Environments**:

  * `conda` (Windows)
  * `venv` (macOS/Linux)
* **Installation Command**:

  ```bash
  pip install "fastapi[standard]" uvicorn openai
  ```

### Creating a Simple FastAPI Web Server

```python
# main.py
from fastapi import FastAPI
from openai import OpenAI

app = FastAPI()
openai_client = OpenAI(api_key="your_api_key")

@app.get("/")
def root_controller():
    return {"status": "healthy"}

@app.get("/chat")
def chat_controller(prompt: str = "Inspire me"):
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    statement = response.choices.message.content
    return {"statement": statement}
```

### Running the Server

```bash
fastapi dev
```

* Server: [http://127.0.0.1:8000](http://127.0.0.1:8000)
* Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### FastAPI Auto Features

* **Interactive Docs** via Swagger UI.
* **Auto Serialization**: Converts Python dicts to JSON and vice versa.

---

## 3. FastAPI Features and Advantages

### Routing & Async Support

* Supports both `def` and `async def`.
* Handles I/O-bound operations efficiently.

### Background Tasks

* Built-in support for running tasks asynchronously in the background.

### Middleware & CORS

* Middleware for custom logging, auth, etc.
* Supports Cross-Origin Resource Sharing (CORS).

### Extensibility & Customization

* Override default behaviors.
* Create custom serializers or middlewares.

### Data Validation

```python
from pydantic import BaseModel, Field, field_validator
import re
from fastapi import FastAPI, status, HTTPException

class UserCreate(BaseModel):
    name: str
    password: str = Field(min_length=8, max_length=64)

    @field_validator("password")
    @classmethod
    def validate_password_complexity(cls, value: str) -> str:
        if not re.search(r"[A-Z]", value):
            raise ValueError("Password must contain at least one uppercase letter")
        if not re.search(r"[a-z]", value):
            raise ValueError("Password must contain at least one lowercase letter")
        if not re.search(r"\d", value):
            raise ValueError("Password must contain at least one digit")
        if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", value):
            raise ValueError("Password must contain at least one special character")
        return value

app = FastAPI()

@app.post("/users")
async def create_user_controller(user: UserCreate):
    return {"name": user.name, "message": "Account successfully created"}
```

### Auto Docs Redirect

```python
from fastapi.responses import RedirectResponse
from starlette import status

@app.get("/", include_in_schema=False)
def docs_redirect_controller():
    return RedirectResponse(url="/docs", status_code=status.HTTP_303_SEE_OTHER)
```

### Dependency Injection

#### Pagination Example

```python
from fastapi import FastAPI, Depends

app = FastAPI()

def paginate(skip: int = 0, limit: int = 10):
    return {"skip": skip, "limit": limit}

@app.get("/messages")
def list_messages_controller(pagination: dict = Depends(paginate)):
    return {"messages": f"Skipping {pagination['skip']} messages, taking {pagination['limit']}"}
```

#### DB Session Example

```python
class DBSession:
    def query(self, *args, **kwargs):
        return "data from DB"
    def close(self):
        pass

def get_db():
    db = DBSession()
    try:
        yield db
    finally:
        db.close()

@app.get("/users/{email}/messages")
def get_current_user_messages(email: str, db: DBSession = Depends(get_db)):
    user = db.query("user by email", email)
    messages = db.query("messages for user", user)
    return {"email": email, "messages": messages}
```

### Additional Capabilities

* Lifespan Events for resource setup/teardown.
* WebSocket, GraphQL, custom responses.
* IDE-friendly with modern Python syntax.

---

## 4. FastAPI Project Structures

### Flat Structure

```
flat-project/app/
├── services.py
├── database.py
├── models.py
├── routers.py
└── main.py
```

* **Simple**, good for prototypes.
* **Not scalable**.

### Nested Structure

```
nested-project/app/
└── services/
    ├── users.py
    └── profiles.py
└── models/
    ├── users.py
    └── profiles.py
```

* **Organized**, better for growing projects.
* **Risk of shotgun updates**.

### Modular Structure

```
modular-project/app/
└── modules/
    ├── auth/
    │   ├── routers.py
    │   └── models.py
    └── users/
        ├── router.py
        └── models.py
```

* **Highly scalable**
* **Best for GenAI and complex apps**

---

## 5. Onion/Layered Application Design Pattern

### Layer Breakdown

1. **API Routers**
2. **Controllers**
3. **Services / Providers**
4. **Repositories**
5. **Schemas / Models**

### Cross-Cutting Concerns

* Middleware
* Dependencies
* Pipes / Mappers
* Exception Filters
* Guards

> Designed for maintainability, testability, and scalability. Often implemented in a modular project layout.

---

## 6. Comparing FastAPI to Other Python Web Frameworks

| Framework   | Type            | Strengths                                    | Weaknesses                     |
| ----------- | --------------- | -------------------------------------------- | ------------------------------ |
| Django      | Opinionated     | Full-stack, batteries included               | Heavy, less async-friendly     |
| Flask       | Non-opinionated | Lightweight, flexible                        | No DI, slow with async tasks   |
| Quart       | Non-opinionated | ASGI, Flask-like                             | Smaller ecosystem              |
| **FastAPI** | Non-opinionated | Async-first, excellent docs, DI, GenAI-ready | Limited for heavy AI workloads |

> FastAPI is ideal for modern, async, and AI-enhanced services.

---

## 7. FastAPI Limitations

* **Memory Bottlenecks** with AI models during horizontal scaling.
* **Thread Pool Limits** (default: 40).
* **Global Interpreter Lock (GIL)** constraints.
* **No Built-in Micro-Batching** for AI inference.
* **CPU/GPU Load Splitting** is inefficient.
* **Not ML-Optimized** for billion-parameter models.

> Use **BentoML** or similar tools for production AI serving.

---

## 8. Setting Up a Managed Python Environment and Tooling

### Dependency Management

* `requirements.txt`, `uv`, `conda`
* `Poetry` (recommended for complex projects)

### Code Quality Tools

* **Linters**: `Autoflake`, `Flake8`, `Ruff`
* **Formatters**: `isort`, `Black`, `Ruff`
* **Loggers**: `Loguru`
* **Scanners**: `Bandit`, `Safety`
* **Type Checkers**: `Mypy`, `Pylance`

### Version Control

* Use Git with `.gitignore`
* Pre-commit hooks for quality control

> These tools are critical for managing GenAI codebases with evolving schemas and external dependencies.

---

## Chapter 2 Summary

* Introduced FastAPI, its core benefits, and async-first architecture.
* Built a working web server with GenAI integration.
* Explored project structures: flat, nested, modular.
* Learned the onion/layered design pattern.
* Compared FastAPI to Flask, Django, and others.
* Identified FastAPI's limitations and alternatives for model-heavy workloads.
* Set up a robust, managed development environment for reliable codebases.

---


