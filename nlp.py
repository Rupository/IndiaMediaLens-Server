import os
import gc
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import spacy
from spacy.language import Language
from spacy.tokens import Span
import pandas as pd
from thefuzz import fuzz, process
from NewsSentiment import TargetSentimentClassifier
import numpy as np
from typing import Literal
from functools import lru_cache
import logging

import torch
torch.set_num_threads(4)

logging.basicConfig(
    filename='runs.log',
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

newsmtsc_classifier = TargetSentimentClassifier()

EST = {
    'BJP',
    'AIADMK',
    'GOV',
    'NCP',
    'ABVP',
    'TDP',
    'AMMK',
    'JDU',
    'RSS',
    'JDS',
    'MNS',
    'LJP',
    'VHP'
}

OPP = {
    'INC',
    'DMK',
    'AAP',
    'SP',
    'SAMAJWADI',
    'SAMAJWADI ',
    'BJD',
    'TRS',
    'CPIM',
    'AITC',
    'TMC',
    'RJD',
    'AIMIM',
    'JKNC',
    'BSP',
    'JKPDP',
    'JMM',
    'CPIML'
}

df_corpus = pd.read_json('data/current/nivaduck_with_display_names.json')
df_corpus = df_corpus.dropna(subset='display_name')
corpus = df_corpus['display_name'].to_dict()

@lru_cache(maxsize=2048)
def get_cached_fuzzy_match(entity_text):
    return process.extractOne(entity_text, corpus, scorer=fuzz.ratio, score_cutoff=95)

nlp = spacy.load('en_core_web_trf', disable=['tagger', 'attribute_ruler', 'lemmatizer'])
@Language.component("fuzzy_affiliation")
def fuzzy_affiliates(doc):
    ents = []
    
    for ent in doc.ents:
        if ent.label_ not in {"PERSON", "ORG"}:
            ents.append(ent)
            continue
        
        match = get_cached_fuzzy_match(ent.text)

        if not match:
            ents.append(ent)
            continue

        _, _, index =  match # pyright: ignore[reportAssignmentType]
        party = str(df_corpus.loc[index, 'party'])

        if party in EST:
            aff = 'EST'
        elif party in OPP:
            aff = 'OPP'
        else:
            ents.append(ent)
            continue
        
        new_ent = Span(doc, ent.start, ent.end, label=aff)
        ents.append(new_ent)
            
    doc.ents = ents
    return doc

nlp.add_pipe("fuzzy_affiliation", after='ner')

def batch_nlp(stories:list[dict[str,str]], entity_type:Literal['EST', 'OPP'], batch_size=16):
    data = []
    texts = [story.get('text', '') for story in stories]
    #for text in texts: print(text, end='\n\n')
    story_datapoints_tracker = [] # for the story at the i'th index, how many newsmtsc tuples it has
    
    with yaspin(text='Resolving Entities...', color='green'):
        with torch.no_grad():
            for doc in nlp.pipe(texts, batch_size=batch_size):
                data_count = 0

                for ent in doc.ents:
                    if ent.label_ == entity_type:
                        sentence = ent.sent
                        left = doc.text[sentence.start_char : ent.start_char]
                        entity_str = ent.text
                        right= doc.text[ent.end_char : sentence.end_char]

                        data.append((left, entity_str, right))
                        data_count += 1
                
                story_datapoints_tracker.append(data_count)
                del doc
    
    if data:

        with yaspin(text='Analysing Sentiment...', color='green'):
            with torch.no_grad():
                sentiments = newsmtsc_classifier.infer(targets=data, batch_size=batch_size, disable_tqdm=True)
        return data, story_datapoints_tracker, sentiments
    
    else:

        logging.warning("<NLP> Could not find ANY relavant entities!")
        return [], [], []


def label_stories(stories:list[dict[str,str]], entity_type:Literal['EST', 'OPP'],
                  data, story_datapoints_tracker, sentiments):

    label = f'{entity_type}_label'
    if data == []:
        for story in stories:
            story[label] = 'unknown'
        return stories
    
    i = 0
    for story, k in zip(stories, story_datapoints_tracker):

        vectors = []

        for datapoint, sentiment in zip(data[i:i+k],sentiments[i:i+k]):
            probs = {item['class_label']: item['class_prob'] for item in sentiment}

            if max(probs.values()) < 0.75: continue

            vector = [probs['positive'], probs['neutral'], probs['negative']]
            vectors.append(vector)
        
        if not vectors:
            story[label] = 'unknown'
            i += k
            continue

        vectors = np.array(vectors)
        article_vec = np.mean(vectors, axis=0)
        class_index = np.argmax(article_vec)

        if class_index == 0:
            story[label] = 'pro'
        elif class_index == 1:
            story[label] = 'neutral'
        elif class_index == 2:
            story[label] = 'anti'
        
        i += k

    
    return stories

def stories_with_nlp(stories:list[dict[str,str]], entity_type:Literal['EST', 'OPP'], batch_size=16):
    data, tracker, sentiments = batch_nlp(stories, entity_type, batch_size)
    final_stories = label_stories(stories, entity_type, data, tracker, sentiments)
    formatted_stories = {story['title']:story[f'{entity_type}_label'] for story in final_stories}

    del data, tracker, sentiments, final_stories
    gc.collect()
    return formatted_stories