import os
import gc
import asyncio
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import json
import traceback
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
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

from current import get_full_stories, shutdown_selenium_pool
from nlp import stories_with_nlp

api = FastAPI(title="IndiaMediaLens Server API", version="0.1.2")

class ColourRequest(BaseModel):
    stories: list[dict[str,str]] = Field(..., max_length=80)

class ErrorResponse(BaseModel):
    error: str

@app.on_startup
def startup_event():
    global cumulative_stance_data
    cumulative_stance_data = get_cumulative_stance_data(start_date=dt(2019, 1, 1), end_date=dt(2024, 12, 31))
    spinner.start()

@app.on_shutdown
def shutdown_event():
    shutdown_selenium_pool()
    gc.collect()

async def app_update_generator(request_stories:list[dict[str,str]]):

    task = spinner.add_task("Intializing", total=3)

    try:
        if not request_stories:
            raise ValueError("Request stories missing")

        yield f"data: {json.dumps({'status':'running', 'msg':'Decoding and Extracting Stories'})}\n\n"
        spinner.update(task, advance=1, description='Decoding and Extracting Stories')
        processed_stories = await asyncio.to_thread(get_full_stories, request_stories, )

        yield f"data: {json.dumps({'status':'running', 'msg':f'Analysing Sentiments in {len(processed_stories)} Articles'})}\n\n"
        spinner.update(task, advance=1, description=f'Analysing Sentiments in {len(processed_stories)} Articles')
        current_est_stances = await asyncio.to_thread(stories_with_nlp, processed_stories, 'EST')

        yield f"data: {json.dumps({'status':'running', 'msg':f'Tidying Up'})}\n\n"
        spinner.update(task, advance=1, description=f'Tidying Up')
        historical_est_stances = assign_hist_stance(request_stories, cumulative_stance_data)
        combined_stanced_data = {}

        for story in request_stories:

            title = story['title']
            hist_est_stance = historical_est_stances.get(title, 'unknown')
            curr_est_stance = current_est_stances.get(title, 'unknown')

            combined_stanced_data[title] = {
                "historical": hist_est_stance,
                "current": curr_est_stance
            }
        
        yield f"data: {json.dumps({'status':'finished', 'data': combined_stanced_data})}\n\n"
    
    except ValueError as e:
        traceback.print_exc()
        yield f"data: {json.dumps({'status':'error', 'msg': f'Invalid data - {str(e)}', 'error_type':'ValueError'})}\n\n"
        logging.error(f"Invalid data: {str(e)}")
    
    except Exception as e:
        traceback.print_exc()
        yield f"data: {json.dumps({'status':'error', 'msg': f'Internal server error - {str(e)}', 'error_type':'ServerError'})}\n\n"
        logging.error("Internal server error")
    
    finally:
        spinner.remove_task(task)


@api.post(
    "/api/v0/colour",
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def colour(request_data: ColourRequest):
    """
    Return colour data (streamed)
    """
    request_stories = request_data.stories
    return StreamingResponse(app_update_generator(request_stories), media_type='text/event-stream')

@ui.page("/historical/visualization/{outlet}")
async def data_visualization(outlet:str):
    visualization.create_session(outlet)

api.mount('/ui', app)
ui.run_with(api, mount_path='/ui', storage_secret='findher.ogg')

if __name__ == "__main__":
    import uvicorn
    import sys
    import asyncio

    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    uvicorn.run(api, host='localhost', port=5000)