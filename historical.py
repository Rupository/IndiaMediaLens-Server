import pandas as pd
import numpy as np
from datetime import datetime as dt
from sklearn.metrics.pairwise import cosine_similarity
from typing import Any, Optional

HISTORICAL_DF = None

OUTLET_TO_DOMAIN = {
        "The Times of India" : "timesofindia.indiatimes.com",
        "Times of India" : "timesofindia.indiatimes.com",
        #"Times Now": "timesnownews.com",
        "The Economic Times" : "economictimes.indiatimes.com",
        #"india.com" : "india.com",
        #"Zee News" : "zeenews.india.com",
        #"BBC" : "bbc.com", # may drop due to low article counts
        "NDTV" : "ndtv.com",
        "India Today" : "indiatoday.in",
        "Hindustan Times" : "hindustantimes.com",
        #"Republic World" : "republicworld.com",
        "The Hindu" : "thehindu.com",
        #"CNN" : "cnn.com", # may drop due to low article counts
        "The Indian Express" : "indianexpress.com",
        "ThePrint" : "theprint.in",
        "Rediff MoneyWiz" : "rediff.com", # odd lising of rediff on gnews, may need to be dropped
        #"Firstpost" : "firstpost.com",
        "The News Minute" : "thenewsminute.com",
        "The Quint" : "thequint.com",
        #"TheWire.in" : "thewire.in", # may need to drop because doesn't seem to show up much in gnews clusters
        #"OpIndia" : "opindia.com",
        #"DNA India" : "dnaindia.com",
        #"Telegraph India" : "telegraphindia.com"
    }

def source_correction(row):
    if row['media_name'] == 'indiatimes.com':
        if 'economictimes.indiatimes.com' in row['url']:
            row['media_name'] = "economictimes." + row['media_name']
        else:
            row['media_name'] = "timesofindia." + row['media_name']

    elif row['media_name'] == 'india.com':
        if 'zeenews.india.com' in row['url']:
            row['media_name'] = "zeenews." + row['media_name']
        else:
            pass
    return row


def load_global_data():
    global HISTORICAL_DF
    if HISTORICAL_DF is None:
        df = pd.read_parquet('data/historical/elections_opinions_annotated_new.parquet')
        df['publish_date'] = pd.to_datetime(df['publish_date'])
        #df['vectors'] = df['vectors'].apply(parse_nparray_str)
        df = df.apply(source_correction, axis=1)

        HISTORICAL_DF = df
    return HISTORICAL_DF


def get_cumulative_stance_data(start_date:dt, end_date:dt) -> pd.DataFrame:
    if start_date > end_date:
        raise ValueError("start_date must be before or equal to end_date")
    
    df = load_global_data()
    
    mask = (df['publish_date'] >= start_date) & \
           (df['publish_date'] <= end_date) & \
           (df['EST_label'] != 'unknown')
    
    df = df.loc[mask].copy()

    counts = df.groupby(['media_name', 'EST_label']).size().unstack(fill_value=0)
    for col in ['pro', 'neutral', 'anti']:
        if col not in counts.columns:
            counts[col] = 0

    counts['total'] = counts.sum(axis=1)
    stance_columns = [col for col in counts.columns if col != 'total']
    percentages = counts[stance_columns].div(counts['total'], axis=0)
    percentages['cumulative stance'] = percentages.idxmax(axis=1)
    counts = counts.add_suffix(' count')
    percentages = percentages.add_suffix(' %')

    outlet_stance = pd.concat([counts, percentages], axis=1)
    return outlet_stance

def get_stance_series(start_date:dt, end_date:dt, scale:str, domain:str):
    if start_date > end_date:
        raise ValueError("start_date must be before or equal to end_date")
    
    df = load_global_data()
    
    mask = (df['media_name'] == domain) & \
           (df['publish_date'] >= start_date) & \
           (df['publish_date'] <= end_date) & \
           (df['EST_label'] != 'unknown')
           
    df = df.loc[mask].copy()
    
    if df.empty:
        return pd.DataFrame()
    
    freq = {'Month': 'ME', '1/4 Year': 'QE', 
            '1/2 Year': '6ME', 'Year': 'YE'}.get(scale, 'ME')
    
    grouper = pd.Grouper(key='publish_date', freq=freq)
    
    counts = df.groupby([grouper, 'EST_label']).size().unstack(fill_value=0).rename_axis(None, axis=1)
    for col in ['pro', 'neutral', 'anti']:
        if col not in counts.columns:
            counts[col] = 0

    percentages = counts.div(counts.sum(axis=1), axis=0) * 100
    results = percentages.reset_index()
    results['publish_date'] = pd.to_datetime(results['publish_date']).dt.strftime('%Y-%m')
    results['total_articles'] = counts.sum(axis=1).values

    return results

def get_outlet_centroid(start_date:dt, end_date:dt, domain:str) -> tuple[Optional[np.ndarray], int]:
    df = load_global_data()

    mask = (df['media_name'] == domain) & \
           (df['publish_date'] >= start_date) & \
           (df['publish_date'] <= end_date)
           
    df = df.loc[mask].copy()

    if df.empty:
        return None, 0
        
    vectors = np.vstack(df['vectors'].tolist())
    centroid = vectors.mean(axis=0)
    return centroid, len(df)

def compute_similarity_matrix(start_date: dt, end_date: dt) -> pd.DataFrame:

    domains = sorted(list(set(OUTLET_TO_DOMAIN.values())))
    valid_domains = []
    vectors = []
    
    counts = {}
    for domain in domains:
        centroid, count = get_outlet_centroid(start_date, end_date, domain)
        if count > 0:
            valid_domains.append(domain)
            vectors.append(centroid)
            counts[domain] = count

    if not vectors:
        return pd.DataFrame()
    
    matrix = np.vstack(vectors)
    # load vectors into a matrix

    global_mean = matrix.mean(axis=0)
    # take their mean, a general "news about elections across outlets" vector
    
    centered_matrix = matrix - global_mean
    # subtract this mean component to arrive at distinctions between outlets
    
    sim_matrix = cosine_similarity(centered_matrix)

    df_sim = pd.DataFrame(
        sim_matrix,
        index=valid_domains,
        columns=valid_domains
    )

    return df_sim

def get_similarity_graph_data(start_date: dt, end_date: dt, focus_domain: str) -> dict[str, Any]:
    df_sim = compute_similarity_matrix(start_date, end_date)

    if df_sim.empty:
        return {"nodes": [], "links": [], "meta": {"most_sim": "None", "least_sim": "None"}}

    stance_df = get_cumulative_stance_data(start_date, end_date)

    most_sim_domain = None
    least_sim_domain = None
    
    if focus_domain and focus_domain in df_sim.index:
        others = df_sim.loc[focus_domain].drop(focus_domain)
        if not others.empty:
            most_sim_domain = others.idxmax()
            least_sim_domain = others.idxmin()

    nodes = []
    valid_stance_df = stance_df[stance_df.index.isin(df_sim.index)]
    
    if not valid_stance_df.empty:
        max_c = valid_stance_df['total count'].max()
        min_c = valid_stance_df['total count'].min()
    else:
        max_c, min_c = 1, 0

    standard_label = {'show': False}
    highlight_label = {
        'show': True,
        'position': 'top',
        'fontStyle': 'italic',
        'fontSize': 9,
        'color': '#333'
    }

    for domain in df_sim.index:
        count = 0
        stance_cat = 'unknown'
        tooltip_str = f"<b>{domain}</b><br/>No data"

        if domain in stance_df.index:
            row = stance_df.loc[domain]
            
            count = int(row['total count'])
            stance_cat = row['cumulative stance %']
            
            pro_pct = round(100 * row.get('pro %', 0), 1)
            neu_pct = round(100 *  row.get('neutral %', 0), 1)
            anti_pct = round(100 * row.get('anti %', 0), 1)

            tooltip_str = (
                f"<div style='text-align:center;'>"
                f"<b>{domain}</b> <br/>"
                f"articles: <b>{count}</b><br/>"
                f"pro: <b>{pro_pct}</b> %<br/>"
                f"neutral: <b>{neu_pct}</b> %<br/>"
                f"anti: <b>{anti_pct}</b> %"
                f"</div>"
            )

        if max_c == min_c:
            size = 30
        else:
            size = 20 + ((count - min_c) / (max_c - min_c)) * 40
        
        item_style = {'borderColor': '#fff', 'borderWidth': 1}
        label_style = standard_label

        if domain == focus_domain:
            item_style = {'borderColor': "#000000", 'borderWidth': 2}
            label_style = highlight_label
        elif domain == most_sim_domain:
            item_style = {'borderColor': "#000000", 'borderWidth': 1}
            label_style = highlight_label
        elif domain == least_sim_domain:
            item_style = {'borderColor': "#000000", 'borderWidth': 1}
            label_style = highlight_label

        nodes.append({
            "name": domain,
            "value": count,
            "symbolSize": size,
            "category": stance_cat,
            "itemStyle": item_style,
            "label": label_style,
            "tooltip": {'formatter': tooltip_str}
        })

    links = []
    threshold = 0.4

    for i, source in enumerate(df_sim.index):
        for j, target in enumerate(df_sim.columns):
            if j <= i: continue 
            
            score = df_sim.iat[i, j]
            
            if score > threshold:  # type: ignore
                links.append({
                    "source": source,
                    "target": target,
                    "value": round(score, 2),  # type: ignore
                    "tooltip": {'formatter': 'similarity: <b>{c}</b>'}
                })

    return {
        "nodes": nodes, 
        "links": links, 
        "meta": {
            "most_sim": most_sim_domain if most_sim_domain is not None else "None", 
            "least_sim": least_sim_domain if least_sim_domain is not None else "None"
        }
    }

def assign_hist_stance(stories:list[dict[str,str]],
                  cumulative_stance_data:pd.DataFrame):

    hist_stanced_stories = dict[str,str]()

    for story in stories:
        if story['outlet'] not in OUTLET_TO_DOMAIN:
            hist_stanced_stories[story['title']] = 'unknown'
            continue

        domain = OUTLET_TO_DOMAIN[story['outlet']]

        if domain not in cumulative_stance_data.index:
            hist_stanced_stories[story['title']] = 'unknown'
            continue

        hist_stanced_stories[story['title']] = str(cumulative_stance_data.loc[domain, 'cumulative stance %'])

    return hist_stanced_stories