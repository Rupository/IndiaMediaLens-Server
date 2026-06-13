import os
import gc
import asyncio
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import json
import traceback
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel, Field
from nicegui import ui, app
from datetime import datetime as dt
from rich.progress import Progress, SpinnerColumn, TextColumn
import logging

logging.basicConfig(
    filename='runs.log',
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

spinner = Progress(SpinnerColumn(speed=1.5), TextColumn("[bold green]{task.description}"), transient=True)

from historical import get_cumulative_stance_data, assign_hist_stance
import visualization

from current import decoding_block, scraping_block, shutdown_selenium_pool
from nlp import ner_block, tone_block, label_stories

class ColourRequest(BaseModel):
    stories: list[dict[str,str]] = Field(..., max_length=80)

class ErrorResponse(BaseModel):
    error: str

def get_ip(request:Request):
    return request.headers.get("CF-Connecting-IP")

api = FastAPI(title="IndiaMediaLens Server API", version="0.1.4")

limiter = Limiter(key_func=get_ip)
api.state.limiter = limiter
api.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

BLOCK_DECODE = asyncio.BoundedSemaphore(2)
BLOCK_SCRAPE = asyncio.BoundedSemaphore(2)
BLOCK_NER = asyncio.BoundedSemaphore(2)
BLOCK_TONE = asyncio.BoundedSemaphore(2)

cumulative_stance_data = None

@app.on_startup
def startup_event():
    global cumulative_stance_data
    cumulative_stance_data = get_cumulative_stance_data(start_date=dt(2019, 1, 1), end_date=dt(2024, 12, 31))
    spinner.start()

@app.on_shutdown
def shutdown_event():
    logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)
    shutdown_selenium_pool()
    gc.collect()

@limiter.limit('1/minute')
def _rate_limit_check(request:Request):
    pass

async def queued_pipeline(request_stories: list[dict[str, str]], queue: asyncio.Queue, task_id):
    try:
        queue.put_nowait({"status": "running", "msg": "Waiting..."})
        async with BLOCK_DECODE:
            queue.put_nowait({"status": "running", "msg": f"Decoding {len(request_stories)} URLs (1/4)"})
            spinner.update(task_id, advance=1, description=f'Decoding {len(request_stories)} URLs (1/4)')
            logging.info(f"Decoding {len(request_stories)} URLs (1/4)")
            decoded_stories = await asyncio.to_thread(decoding_block, request_stories)

        queue.put_nowait({"status": "running", "msg": "Waiting..."})
        async with BLOCK_SCRAPE:
            queue.put_nowait({"status": "running", "msg": f"Scraping {len(decoded_stories)} Articles (2/4)"})
            spinner.update(task_id, advance=1, description=f'Scraping {len(decoded_stories)} Articles (2/4)')
            logging.info(f"Scraping {len(decoded_stories)} Articles (2/4)")
            parsed_stories = await asyncio.to_thread(scraping_block, decoded_stories)

        queue.put_nowait({"status": "running", "msg": "Waiting..."})
        async with BLOCK_NER:
            queue.put_nowait({"status": "running", "msg": f"Performing NER on {len(parsed_stories)} Articles (3/4)"})
            spinner.update(task_id, advance=1, description=f'Performing NER on {len(parsed_stories)} Articles (3/4)')
            logging.info(f"Performing NER on {len(parsed_stories)} Articles (3/4)")
            data, story_datapoints_tracker = await asyncio.to_thread(ner_block, parsed_stories, 'EST')
        
        queue.put_nowait({"status": "running", "msg": "Waiting..."})
        async with BLOCK_TONE:
            queue.put_nowait({"status": "running", "msg": f"Analyzing Sentiments in {len(data)} Sentences (4/4)"})
            spinner.update(task_id, advance=1, description=f'Analyzing Sentiments in {len(data)} Sentences (4/4)')
            logging.info(f"Analyzing Sentiments in {len(data)} Sentences (4/4)")
            data, story_datapoints_tracker, sentiments = await asyncio.to_thread(tone_block, data, story_datapoints_tracker)

        queue.put_nowait({"status": "running", "msg": "Tidying Up"})
        spinner.update(task_id, advance=1, description='Tidying Up')
        current_est_stances = label_stories(parsed_stories, 'EST', data, story_datapoints_tracker, sentiments)
        historical_est_stances = assign_hist_stance(request_stories, cumulative_stance_data)
        combined_stanced_data = {}

        for story in request_stories:
            title = story['title']
            combined_stanced_data[title] = {
                "historical": historical_est_stances.get(title, 'unknown'),
                "current": current_est_stances.get(title, 'unknown')
            }
        
        queue.put_nowait({"status": "running", "msg": f"Extracted Sentiments for {len(combined_stanced_data)} Articles"})
        logging.info(f"Extracted Sentiments for {len(combined_stanced_data)} Articles")
        await asyncio.sleep(1.5)

        del request_stories, decoded_stories, parsed_stories
        del data, story_datapoints_tracker, sentiments
        del current_est_stances, historical_est_stances
        gc.collect()

        queue.put_nowait({"status": "finished", "data": combined_stanced_data})

    except Exception as e:
        traceback.print_exc()
        logging.error(f"Pipeline error: {str(e)}")
        queue.put_nowait({"status": "error", "msg": f"Pipeline error - {str(e)}", "error_type": "ServerError"})

async def sse_stream(request: Request, request_stories: list[dict[str, str]]):
    task_id = spinner.add_task("Initializing", total=5)
    
    try:
        _rate_limit_check(request)
        
        if not request_stories:
            raise ValueError("Request stories missing")

        queue = asyncio.Queue()
        worker_task = asyncio.create_task(queued_pipeline(request_stories, queue, task_id))

        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=2.0)
                yield f"data: {json.dumps(item)}\n\n"
                
                if item["status"] in ["finished", "error"]:
                    break 
                    
            except asyncio.TimeoutError:
                yield f": ping\n\n"
                
                if worker_task.done() and worker_task.exception():
                    e = worker_task.exception()
                    logging.error(f"Fatal task crash: {str(e)}")
                    yield f"data: {json.dumps({'status': 'error', 'msg': f'Fatal task crash: {str(e)}', 'error_type': 'ServerError'})}\n\n"
                    break

    except ValueError as e:
        yield f"data: {json.dumps({'status':'error', 'msg': f'Invalid data - {str(e)}', 'error_type':'ValueError'})}\n\n"
        logging.error(f"Invalid data: {str(e)}")
        print("FAIL:     Invalid data")
    
    except RateLimitExceeded as e:
        yield f"data: {json.dumps({'status':'error', 'msg': f'Rate limit exceeded - 1 req/min (429)', 'error_type':'RateLimitError'})}\n\n"
        logging.error(f"Rate limit exceeded: {str(e)}")
        print("FAIL:     Rate limit exceeded")
    
    except Exception as e:
        traceback.print_exc()
        yield f"data: {json.dumps({'status':'error', 'msg': f'Internal server error - {str(e)}', 'error_type':'ServerError'})}\n\n"
        logging.error(f"Internal server error: {str(e)}")
        print("FAIL:     Internal server error")
        
    finally:
        spinner.remove_task(task_id)

@api.post(
    "/api/v0/colour",
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def colour(request: Request, request_data: ColourRequest):
    """Return colour data (streamed)"""
    return StreamingResponse(
        sse_stream(request, request_data.stories), 
        media_type='text/event-stream'
    )

@ui.page("/historical/visualization/{outlet}")
async def data_visualization(outlet:str):
    visualization.create_session(outlet)

api.mount('/ui', app)
ui.run_with(api, mount_path='/ui', storage_secret='findher.ogg', reconnect_timeout=30.0)

if __name__ == "__main__":
    import uvicorn
    import sys
    import asyncio

    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    uvicorn.run(api, host='localhost', port=5000)