# Validating user passwords in FastAPI using a Pydantic schema
# Start from terminal using `uvicorn 2_2:app --reload`

from fastapi import FastAPI
from pydantic import BaseModel, Field, EmailStr, validator


class UserCreate(BaseModel):
    """
    Pydantic model to validate the user creation input.

    Attributes:
        username (str): The username of the user.
        password (str): The password of the user.
    """
    username: str
    password: str

    @validator('password')
    def validate_password(cls, value):
        """
        Validator method to ensure the password meets the required criteria.

        Args:
            value (str): The password input by the user.

        Raises:
            ValueError: If the password does not meet certain conditions.

        Returns:
            str: The validated password if all conditions are met.
        """
        if len(value) < 8:
            raise ValueError('password must be at least 8 characters long')
        if not any(char.isdigit() for char in value):
            raise ValueError('Password must contain at least one digit')
        if not any(char.isupper() for char in value):
            raise ValueError('Password must contain at least one uppercase letter')
        return value


# Initialize the FastAPI application
app = FastAPI()

@app.post("/users")
async def create_user_controller(user: UserCreate):
    """
    Controller to handle user creation endpoint.

    Args:
        user (UserCreate): The validated user input.

    Returns:
        dict: A response indicating successful user creation.
    """
    return {"name": user.username, "message": "Account successfully created"}