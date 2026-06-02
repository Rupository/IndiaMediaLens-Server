import os
import threading
from typing import Literal
from dotenv import load_dotenv
import serpapi
from newspaper import Article, Config
from newspaper.article import ArticleException
from concurrent.futures import ThreadPoolExecutor, as_completed
from selenium import webdriver
from selenium.webdriver import ChromeOptions

load_dotenv()
serp_api_key = os.environ.get('SERP_API_KEY')
serp_client = serpapi.Client(api_key=serp_api_key)

def get_selenium_html(url):
    with threading.Lock():
        with webdriver.Chrome(options=options) as driver: 
            driver.get(url)
            article_html = driver.page_source
        return article_html

options = ChromeOptions()
options.page_load_strategy = 'eager'
options.add_argument("--headless=new")

config = Config()
config.browser_user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124  Safari/537.36'
config.request_timeout = 20

def parse_url(url) -> str | None:
    try:
        article = Article(url)
        article.download()
        article.parse()

        text = " ".join(article.text.split()[:200])
        if text == '':
            raise LookupError(f"First Attempt: Unable to extact article text for {url}. Retrying...") # if fail, go into second try

        return text
    
    except (ArticleException, LookupError) as e1:
        try:
            if str(e1).find('403') != -1 or isinstance(e1, LookupError):
                article = Article(url, config=config)
                article.download()
                article.parse()

                text = " ".join(article.text.split()[:200])
                if text == '':
                    raise LookupError(f"Second Attempt: Unable to extact article text for {url}. Retrying...") # if fail, go into third try
                else:
                    print(f'Retry succeeded for {url}!')
                
                return text
            
            else:
                print(e1)
                return ''
            
        except (ArticleException, LookupError) as e2:
            if str(e2).find('403') != -1 or isinstance(e2, LookupError):
                article = Article(url, config=config)
                article.download(input_html = get_selenium_html(url))
                article.parse()
                text = " ".join(article.text.split()[:200])
                if text == '':
                    text = ''
                    print(f"Final Attempt: Unable to extact article text for {url}") # if it still fails, can't circumvent.
                else:
                    print(f'Retry succeeded for {url}!')
                
                return text

            else:
                print(e2)
                return ''
    

def get_serp_stories(story_token:str):
    params = {
    "engine": "google_news",
    "gl": "in",
    "hl": "en",
    "story_token": story_token,
    "api_key": f"{serp_api_key}",
    #"no_cache": "true",
    "json_restrictor": "news_results[].stories[].{source.name, title, link, iso_date}, news_results[].{source.name, title, link, iso_date}"
    }

    search = serp_client.search(params)
    #print(search.get_dict())

    stories = []

    for heading in search.get_dict()["news_results"]:
        if heading.get('title') == 'Posts on X':
            continue

        results = heading.get('stories', [heading])

        for story in results:
            stories.append({
                'title': story.get('title'),
                'outlet': story.get('source').get('name'),
                'url': story.get('link'),
                'publish_date': story.get('iso_date').partition('T')[0]
                })
            
    return stories

def parse_stories(stories:list[dict[str,str]]):
    for story in stories:
        url = story.get('url')
        text = parse_url(url)
        story['text'] = text if text is not None else ''
    
    return stories

def parse_stories_parallel(stories, max_threads=80):
    
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        future_to_story = {
            executor.submit(parse_url, story['url']): story 
            for story in stories
        }
        
        for future in as_completed(future_to_story):
            story = future_to_story[future]
            try:
                text = future.result()
                story['text'] = text
                
            except Exception as e:
                print(f"Unexpected failure processing {story['url']}: {e}")
                story['text'] = ''

    return stories

def get_full_stories(story_token:str):
    stories = get_serp_stories(story_token)
    return parse_stories_parallel(stories)

def request_serp_match(request_stories:dict[str,str], serp_stories:list[dict[str,str]], 
                       assign_key:Literal['EST_label', 'OPP_label']):
    
    serp_set = {story['title'] for story in serp_stories}
    request_set = set(request_stories.keys())
    intersection = serp_set & request_set

    final_stories = dict[str,str]()

    for story in serp_stories:
        title = story['title']
        if title in intersection:
            final_stories[title] = story[assign_key]
    
    return final_stories
    

