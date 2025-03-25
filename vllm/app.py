from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from datetime import datetime
import os
import logging

LOG_PATH= './logs'
LOGFILE_CONTAINER = f'{LOG_PATH}/logfile_container_vllm.log'
os.makedirs(os.path.dirname(LOGFILE_CONTAINER), exist_ok=True)
logging.basicConfig(filename=LOGFILE_CONTAINER, level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [START] vLLM 222 started logging in {LOGFILE_CONTAINER}')


app = FastAPI()

llm_instance = None

@app.get("/")
async def root():
    return f'Hello from server :DDDDDDDDDDDDDDDDDDDDD {os.getenv("VLLM_PORT")}!'

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("VLLM_PORT")))