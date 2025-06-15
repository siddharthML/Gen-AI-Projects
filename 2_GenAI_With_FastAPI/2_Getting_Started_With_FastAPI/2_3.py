# Setting up redirect to the auto-generated docs page

"""
Set up a handler for the base root handler, but don't include
it in OpenAPI specifications and the documentation page.

Return a redirect response to the /docs page with a
redirection status code for the browsers to perform the
redirect.
"""
from fastapi import FastAPI, status
from fastapi.responses import RedirectResponse

app = FastAPI()

@app.get("/", include_in_schema=False)
def docs_redirect_controller():
    """
    Handle requests to the root URL and redirect to /docs.

    Returns:
        RedirectResponse: Response object to redirect the browser to /docs with 
                          a status code of 303 SEEOTHER.
    """
    return RedirectResponse(url='/docs', status_code=status.HTTP_303_SEEOTHER)