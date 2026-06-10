import os
import gc
import queue
import threading
from typing import Literal
from googlenewsdecoder import gnewsdecoder
from newspaper import Article, Config
from newspaper.article import ArticleException
from concurrent.futures import ThreadPoolExecutor, as_completed
from selenium import webdriver
from selenium.webdriver import ChromeOptions
import logging

logging.basicConfig(
    filename='runs.log',
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

options = ChromeOptions()
options.page_load_strategy = 'eager'
options.add_argument("--headless=new")
options.add_argument("--disable-gpu")

options.add_argument("--disable-blink-features=AutomationControlled")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--disable-features=IsolateOrigins,site-per-process")
options.add_argument("--disable-extensions")
options.add_argument("--blink-settings=imagesEnabled=false")

options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option('useAutomationExtension', False)

prefs = {
    "profile.managed_default_content_settings.images": 2,
    "profile.managed_default_content_settings.stylesheets": 2,
    "profile.managed_default_content_settings.fonts": 2,
    "profile.managed_default_content_settings.media_stream": 2
}

THREAD_EXEC = ThreadPoolExecutor(max_workers=20)

# make a pre warmed pool of 10 drivers (chromium heads)
DRIVER_POOL = queue.Queue()
for _ in range(10):DRIVER_POOL.put(webdriver.Chrome(options=options))

def get_selenium_html(url):
    driver = DRIVER_POOL.get()
    try:
        driver.get(url)
        article_html = driver.page_source
        return article_html
    except:
        raise
    finally:
        driver.delete_all_cookies() # denavigate from opened news links, may guzzle RAM
        driver.get("about:blank")
        DRIVER_POOL.put(driver)

config = Config()
config.browser_user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124  Safari/537.36'
config.request_timeout = 20

def parse_url(url) -> str | None:
    try: # first attempt
        article = Article(url)
        article.download()
        article.parse()

        text = " ".join(article.text.split()[:200])
        del article

        if text == '':
            raise LookupError(f"\n[INFO] First Attempt: Unable to extact article text for [{url}]. Retrying...") # if fail, go into second try

        return text
    
    except (ArticleException, LookupError) as e1:
        try: # second attempt (identical)
            if str(e1).find('403') != -1 or isinstance(e1, LookupError):
                article = Article(url)
                article.download()
                article.parse()

                text = " ".join(article.text.split()[:200])
                del article

                if text == '':
                    raise LookupError(f"\n[INFO] Second Attempt: Unable to extact article text for [{url}]. Retrying...") # if fail, go into third try
                else:
                    logging.info(f'Retry succeeded for [{url}]!')
                    
                return text
            
            else:
                logging.error(str(e1))
                return ''
            
        except (ArticleException, LookupError) as e2:
            try: # 3rd attempt (with config)
                if str(e2).find('403') != -1 or isinstance(e2, LookupError):
                    article = Article(url, config=config)
                    article.download()
                    article.parse()
                    text = " ".join(article.text.split()[:200])
                    del article

                    if text == '':
                        raise LookupError(f"\n[INFO] Third Attempt: Unable to extact article text for [{url}]. Retryig...") # if fail, final try
                    else:
                        logging.info(f'Retry succeeded for [{url}]!')
                    
                    return text

                else:
                    logging.error(str(e2))
                    return ''

            except (ArticleException, LookupError) as e3:
                # final attempt (config and selenium)
                if str(e3).find('403') != -1 or isinstance(e3, LookupError):
                    article = Article(url, config=config)
                    article.download(input_html = get_selenium_html(url))
                    article.parse()
                    text = " ".join(article.text.split()[:200])
                    del article

                    if text == '':
                        text = ''
                        logging.error(f"Final Attempt: Unable to extact article text for [{url}]") # if it still fails, can't circumvent.
                    else:
                        logging.info(f'Retry succeeded for [{url}]!')
                    
                    return text

                else:
                    logging.error(str(e3))
                    return ''

def decode_url(url:str):
    return gnewsdecoder(url)['decoded_url']

def parse_stories_parallel(stories, max_threads=15):
    future_to_story = {
        THREAD_EXEC.submit(parse_url, story['url']): story 
        for story in stories
    }
    
    for future in as_completed(future_to_story):
        story = future_to_story[future]
        try:
            text = future.result()
            story['text'] = text
            
        except Exception as e:
            logging.error(f"Unexpected failure processing [{story['url']}]: {e}")
            story['text'] = ''

    return stories

def decode_urls_parallel(stories:list[dict[str|str]], max_threads=15):
    future_to_story = {
        THREAD_EXEC.submit(decode_url, story['url']): story 
        for story in stories
    }
    
    for future in as_completed(future_to_story):
        story = future_to_story[future]
        try:
            url = future.result()
            story['url'] = url
            
        except Exception as e:
            logging.error(f"Unexpected failure processing [{story['url']}]: {e}")
            story['url'] = ''

    return stories

def get_full_stories(request_stories:list[dict[str,str]]):
    stories = decode_urls_parallel(request_stories)
    stories = [story for story in stories if story['url'] != '']
    parsed_stories = parse_stories_parallel(stories)
    processed_stories = [story for story in stories if story['text'] != '']
    gc.collect()

    return processed_stories

def shutdown_selenium_pool():
    while not DRIVER_POOL.empty():
        driver = DRIVER_POOL.get()
        driver.quit()
    print("INFO:     Headerless browsers quit.")