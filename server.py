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
from pydantic import BaseModel
from nicegui import ui, app
from datetime import datetime as dt
from rich.progress import Progress, SpinnerColumn, TextColumn

spinner = Progress(SpinnerColumn(), TextColumn("[bold green]{task.description}"), transient=True)

from historical import get_cumulative_stance_data, assign_hist_stance
import visualization

from current import get_full_stories, shutdown_selenium_pool
from nlp import stories_with_nlp

api = FastAPI(title="IndiaMediaLens Server API", version="0.1.2")

class ColourRequest(BaseModel):
    stories: list[dict[str,str]]

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

@api.post(
    "/api/v0/colour",
    response_model=dict,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)

async def app_update_generator():
    try:
        task = spinner.add_task("Intializing...", total=3)

        yield f"data: {json.dumps({'status':'running', 'message':'Request Accepted > Extracting Stories...'})}"
        spinner.update(task, advance=1, description='Request Accepted > Extracting Stories...')
        
        processed_stories = await asyncio.to_thread(get_full_stories, request_stories, )
        yield f"data: {json.dumps({'status':'running', 'message':f'Extracted {len(processed_stories)} Stories > Analysing Sentiments...'})}"
        spinner.update(task, advance=1, description=f'Extracted {len(processed_stories)} Stories > Analysing Sentiments...')

        current_est_stances = await asyncio.to_thread(stories_with_nlp, processed_stories, 'EST')
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
        
        yield f"data: {json.dumps({'status':'finished', 'data': combined_stanced_data})}"
    
    except ValueError as e:
        yield f"data: {json.dumps({'status':'error', 'message': f'Invalid data: {str(e)}'})}"
        raise HTTPException(status_code=400, detail=f"Invalid data: {str(e)}")
    
    except KeyError as e:
        yield f"data: {json.dumps({'status':'error', 'message': f'Missing data: {str(e)}'})}"
        raise HTTPException(status_code=401, detail=f"Missing data: {str(e)}")
    
    except ConnectionError as e:
        yield f"data: {json.dumps({'status':'error', 'message': 'Service temporarily unavailable'})}"
        raise HTTPException(status_code=503, detail="Service temporarily unavailable")
    
    except TimeoutError as e:
        yield f"data: {json.dumps({'status':'error', 'message': 'Request timeout'})}"
        raise HTTPException(status_code=504, detail="Request timeout")
    
    except Exception as e:
        traceback.print_exc()
        yield f"data: {json.dumps({'status':'error', 'message': 'Internal server error'})}"
        raise HTTPException(status_code=500, detail="Internal server error")

async def colour(request_data: ColourRequest):
    """
    Return colour data (streamed)
    """
    request_stories = request_data.stories
    
    if not request_stories:
        raise HTTPException(status_code=400, detail="Request stories missing")

    return StreamingResponse(app_update_generator(), media_type='text/event-stream')

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