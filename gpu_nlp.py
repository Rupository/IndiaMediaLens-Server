import modal
from typing import Literal

if modal.is_local():
    from dotenv import load_dotenv
    load_dotenv()

container = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.10")
    .pip_install(
        "cupy-cuda12x==12.3.0",
        "spacy[cuda12x]==3.8.14",
        "NewsSentiment==1.2.28",
        "pandas==2.3.3",
        "numpy==1.26.4",
        "thefuzz==0.22.1",
        "typing==3.7.4.3",
        "sentencepiece==0.2.1"
    )
    .run_commands(
        "python -m spacy download en_core_web_sm",
        "python -m spacy download en_core_web_trf",
        "python -c 'from NewsSentiment import TargetSentimentClassifier; TargetSentimentClassifier()'",
    )
    .add_local_file('data/current/nivaduck_with_display_names.json', '/mnt/data/current/nivaduck_with_display_names.json')
)

app = modal.App("ViewFinderNLP")

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


@app.cls(image=container, gpu="T4", 
        enable_memory_snapshot=True, 
        experimental_options={"enable_gpu_snapshot": True})
class NLP_processor:

    @modal.enter(snap=True)
    def boot(self):
        import warnings
        warnings.filterwarnings("ignore", category=FutureWarning)

        import spacy
        from spacy.language import Language
        from spacy.tokens import Span
        import pandas as pd
        from thefuzz import fuzz, process
        from NewsSentiment import TargetSentimentClassifier
        spacy.require_gpu()

        self.newsmtsc_classifier = TargetSentimentClassifier()
        self.nlp = spacy.load('en_core_web_trf')
        self.df_corpus = pd.read_json('/mnt/data/current/nivaduck_with_display_names.json')
        self.df_corpus = self.df_corpus.dropna(subset='display_name')
        self.corpus = self.df_corpus['display_name'].to_dict()

        if not Language.has_factory("fuzzy_affiliation"):
            @Language.component("fuzzy_affiliation")
            def fuzzy_affiliates(doc):
                ents = []
                
                for ent in doc.ents:
                    if ent.label_ not in {"PERSON", "ORG"}:
                        ents.append(ent)
                        continue
                    
                    match = process.extractOne(ent.text, self.corpus, scorer=fuzz.ratio, score_cutoff=95)

                    if not match:
                        ents.append(ent)
                        continue

                    _, _, index =  match # pyright: ignore[reportAssignmentType]
                    party = str(self.df_corpus.loc[index, 'party'])

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
        
        if "fuzzy_affiliation" not in self.nlp.pipe_names:
            self.nlp.add_pipe("fuzzy_affiliation", after='ner')


    def batch_nlp(self, stories:list[dict[str,str]], entity_type:Literal['EST', 'OPP'], batch_size=16):
        data = []
        texts = [story.get('text', '') for story in stories]
        #for text in texts: print(text, end='\n\n')
        story_datapoints_tracker = [] # for the story at the i'th index, how many newsmtsc tuples it has
        
        for doc in self.nlp.pipe(texts, batch_size=batch_size):
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
        
        if data:

            sentiments = self.newsmtsc_classifier.infer(targets=data, batch_size=batch_size)
            return data, story_datapoints_tracker, sentiments
        
        else:

            print("Could not find ANY relavant entities!")
            return [], [], []


    def label_stories(self, stories:list[dict[str,str]], entity_type:Literal['EST', 'OPP'],
                    data, story_datapoints_tracker, sentiments):
        
        import numpy as np
        print()
        print(data)
        print()

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
                print(datapoint)
                print(vector)
                vectors.append(vector)
            
            if not vectors:
                story[label] = 'unknown'
                i += k
                continue

            print()

            vectors = np.array(vectors)
            article_vec = np.mean(vectors, axis=0)
            print('Outlet:', story['outlet'])
            print('URL:', story['url'])
            print('Article Vector:', article_vec)
            class_index = np.argmax(article_vec)

            if class_index == 0:
                story[label] = 'pro'
            elif class_index == 1:
                story[label] = 'neutral'
            elif class_index == 2:
                story[label] = 'anti'
            print('ENT_label:', story[label])
            
            i += k
            print()
            print()
        
        return stories

    @modal.method()
    def stories_with_nlp(self, stories:list[dict[str,str]], entity_type:Literal['EST', 'OPP'], batch_size=16):
        data, tracker, sentiments = self.batch_nlp(stories, entity_type, batch_size)
        return self.label_stories(stories, entity_type, data, tracker, sentiments)