import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from nicegui import ui, app
from datetime import datetime as dt

from historical import get_cumulative_stance_data, assign_stance
import visualization

from current import get_full_stories, request_serp_match
from nlp import stories_with_nlp

api = FastAPI(title="IndiaMediaLens Server API", version="0.1.1")

class ColourRequest(BaseModel):
    story_token: str
    stories: dict[str,str]

class ErrorResponse(BaseModel):
    error: str

@app.on_startup
def startup_event():
    global cumulative_stance_data
    cumulative_stance_data = get_cumulative_stance_data(start_date=dt(2019, 1, 1), end_date=dt(2024, 12, 31))

@api.post(
    "/api/v0/colour",
    response_model=dict,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)

def historical_colour(request_data: ColourRequest):
    """
    Return coloured sqaures 
    """
    story_token = request_data.story_token
    request_stories = request_data.stories

    if not story_token:
        raise HTTPException(status_code=400, detail="Story token missing")
    
    if not request_stories:
        raise HTTPException(status_code=400, detail="Request stories missing")

    try:
        print("Fetching colours...", end='\n\n\n')
        serp_stories = get_full_stories(story_token)
        labeled_serp_stories = stories_with_nlp(serp_stories, 'EST')
        current_est_stances = request_serp_match(request_stories, labeled_serp_stories, 'EST_label')

        historical_est_stances = assign_stance(request_stories, cumulative_stance_data)

        combined_stanced_data = {}

        for title in request_stories.keys():
            hist__est_stance = historical_est_stances.get(title, 'unknown')
            curr__est_stance = current_est_stances.get(title, 'unknown')

            combined_stanced_data[title] = {
                "historical": hist__est_stance,
                "current": curr__est_stance
            }
        
        return combined_stanced_data
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid data: {str(e)}")
    
    except KeyError as e:
        raise HTTPException(status_code=401, detail=f"Missing data: {str(e)}")
    
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail="Service temporarily unavailable")
    
    except TimeoutError as e:
        raise HTTPException(status_code=504, detail="Request timeout")
    
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error")

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

    uvicorn.run(api, host='127.0.0.1', port=5000)