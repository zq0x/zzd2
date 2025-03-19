from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from datetime import datetime
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI()

llm_instance = None

@app.get("/")
async def root():
    return f'Hello from server :DDDDDDDDDDDDDDDDDDDDD {os.getenv("VLLM_PORT")}!'

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("VLLM_PORT")))