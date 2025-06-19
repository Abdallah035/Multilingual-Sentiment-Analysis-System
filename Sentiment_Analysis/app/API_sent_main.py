from typing import List
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
import sys
import os
from Sentiment_Analysis import get_sentiment, get_sentiment_list

# Ensure the module is accessible
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

try:
    from Sentiment_Analysis import get_sentiment, get_sentiment_list
except ImportError:
    raise ValueError("Sentiment_Analysis module not found. Ensure it is in the correct directory.")

app = FastAPI()

# Models
class Sentiment(BaseModel):
    text: str

class SentimentList(BaseModel):
    texts: List[str]


# **Global Exception Handler**
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "An unexpected error occurred.",
            "code": 500,
            "error": repr(exc)
        },
    )


# **Custom Exception Handler for HTTPException**
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": exc.detail,
            "code": exc.status_code,
            "error": repr(exc)
        },
    )


# **Custom Exception Handler for Validation Errors**
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "message": "Validation error occurred.",
            "code": 422,
            "error": exc.errors()
        },
    )


@app.get("/")
def read_root():
    return {"message": "Hello from our API"}


@app.post("/get-sentiment/")
async def get_sentiment_api(request: Sentiment):
    try:
        if not request.text.strip():
            raise HTTPException(
                status_code=400,
                detail="No text provided"
            )
        sentiment = get_sentiment(request.text)
        return {"success": True, "sentiment": sentiment}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise e  # Global handler will catch this


@app.post("/get-sentiment-list/")
async def get_sentiment_list_api(request: SentimentList):
    try:
        if not request.texts or not all(isinstance(text, str) and text.strip() for text in request.texts):
            raise HTTPException(
                status_code=400,
                detail="Invalid or empty text list provided"
            )
        sentiment_list = get_sentiment_list(request.texts)
        return {"success": True, "sentiment_list": sentiment_list}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise e  # Global handler will catch this
