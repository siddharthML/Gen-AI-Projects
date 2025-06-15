"""
Using a pagination dependency reduces duplication by centralizing pagination logic.
This approach allows for consistent behavior across different endpoints, cleans up 
individual routing functions, and simplifies maintenance. It ensures that changes 
to pagination parameters affect all endpoints that use this dependency, making 
the codebase more maintainable and scalable.
"""

from fastapi import FastAPI, Depends

app = FastAPI()

def paginate(skip: int = 0, limit: int = 10):
    return {"skip": skip, "limit": limit}

@app.get("/messages")
def list_message_controller(pagination: dict = Depends(paginate)):
    return ... #filter and paginate results using pagination params

@app.get("/conversations")
def list_conversation_contoller(pagination: dict = Depends(paginate)):
    return ... # filter and paginate results using pagination params   