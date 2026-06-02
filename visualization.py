from nicegui import ui
import pandas as pd
import asyncio
from datetime import datetime as dt
import calendar
from historical import OUTLET_TO_DOMAIN, get_stance_series, get_similarity_graph_data

MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
YEARS = [str(y) for y in range(2019, 2025)]
SCALES = ['Month', '1/2 Year', '1/4 Year', 'Year']
COLOR_MAP = {"pro":"#33cc33", "anti":"#ff5050", "neutral":"#ffcc00",}

selections = {
    'start_month': MONTHS[0],
    'start_year': YEARS[0],
    'end_month': MONTHS[-1],
    'end_year': YEARS[-1],
    'scale': SCALES[-1]
}

async def handle_reload(outlet: str, 
                        stance_chart: ui.echart, 
                        graph_chart: ui.echart,
                        graph_spinner: ui.spinner,
                        stance_spinner: ui.spinner,
                        ):
    
    graph_chart.set_visibility(False)
    graph_spinner.set_visibility(True)
    stance_chart.set_visibility(False)
    stance_spinner.set_visibility(True)
    
    await asyncio.sleep(0.1)

    reload_bar(outlet, stance_chart)
    reload_graph(outlet, graph_chart)

    graph_chart.set_visibility(True)
    graph_spinner.set_visibility(False)
    stance_chart.set_visibility(True)
    stance_spinner.set_visibility(False)

def get_graph_options(nodes: list, links: list, meta: dict):
    if not nodes:
        return {
            'title': {
                'text': 'Data unavailable',
                'left': 'center', 
                'top': 'center'
            },
            'xAxis': {'show': False}, 'yAxis': {'show': False}
        }
    
    most_sim = meta.get('most_sim', 'None')
    least_sim = meta.get('least_sim', 'None')

    info_text_block = {
        'text': f"{{b|closest}} - {{i|{most_sim}}}\n{{b|furthest}} - {{i|{least_sim}}}",
        
        'bottom': '20',
        'left': 'center',        
        'textStyle': {
            'fontSize': 10,
            'lineHeight': 20,
            'color': '#333',
            'rich': {
                'b': {
                    'fontWeight': 'bold',
                },
                'i': {
                    'fontStyle': 'italic',
                    'fontWeight': 'normal',
                }
            }
        },
        'backgroundColor': 'rgba(255, 255, 255, 0.9)',
        'borderColor': '#ccc',
        'borderWidth': 1,
        'padding': [15, 15], 
        'borderRadius': 10
    }

    return {
        'title': [info_text_block],
        
        'legend': [{'data': ['pro', 'neutral', 'anti'], 'top': '20'}],
        'tooltip': {
            'trigger': 'item',
            'confine': True,
            'padding': 4,
            'textStyle': {'fontSize': 11}
        },
        'series': [{
            'labelLayout': {
                'hideOverlap': True
            },
            'type': 'graph',
            'layout': 'force',
            'data': nodes,
            'links': links,
            'categories': [
                {'name': 'pro', 'itemStyle': {'color': COLOR_MAP['pro']}},
                {'name': 'anti', 'itemStyle': {'color': COLOR_MAP['anti']}},
                {'name': 'neutral', 'itemStyle': {'color': COLOR_MAP['neutral']}},
                {'name': 'unknown', 'itemStyle': {'color': '#ccc'}}
            ],
            'roam': True,
            'force': {
                'repulsion': 300,
                'gravity': 0.2,
                'edgeLength': 100
            },
            'lineStyle': {
                'curveness': 0.2,
                'width': 3,
                'color': "#58A9FF"
            }
        }]
    }

def get_plot_options(df: pd.DataFrame):
    if df.empty:
        return {
            'title': {'text': 'Data unavailable', 'left': 'center', 'top': 'center'},
            'xAxis': {'show': False}, 'yAxis': {'show': False}
        }

    x_axis_data = df['publish_date'].astype(str).tolist()

    tooltip_formatter = (
        '<div style="text-align:center;">'
        '<b>{b}</b><br/>'
        'articles: <b>{c0}</b><br/>'
        'pro: <b>{c1}</b> %<br/>'
        'neutral: <b>{c2}</b> %<br/>'
        'anti: <b>{c3}</b> %'
        '</div>'
    )

    return {
        'tooltip': {
            'trigger': 'axis', 
            'confine': True, 
            'padding': 4, 
            'textStyle': {'fontSize': 11},
            'formatter': tooltip_formatter
        },

        'legend': {'data': ['pro', 'neutral', 'anti'], 'top': '20'},
        'grid': {'left': '10%', 'right': '4%', 'bottom': '20%', 'containLabel': True},
        'xAxis': {'type': 'category', 'data': x_axis_data},
        'yAxis': [{
                    'type': 'value',
                    'max': 100,
                    'name': 'Percentage (%)',
                    'nameLocation': 'middle',
                    'nameGap': 35
                },
                {
                    'type': 'value',
                    'show': False,
                    'minInterval': 1
                }],
        'dataZoom': [
            {
                'type': 'slider',
                'show': True,
                'xAxisIndex': [0],
                'start': 0,
                'end': 100,
                'showDataShadow': False,
                'height': 20,
                'left': 'center', 
                'width': '85%',
                'textStyle': {
                    'color': 'transparent'
                },
            },
            {'type': 'inside', 'xAxisIndex': [0], 'start': 0, 'end': 100}
        ],
        'series': [
            {
                'name': 'articles',
                'type': 'line',        
                'yAxisIndex': 1,       
                'symbol': 'none',     
                'lineStyle': {'width': 0},
                'itemStyle': {'opacity': 0, 'color': 'transparent'},
                'data': df['total_articles'].tolist()
            },
            {'name': 'pro', 'type': 'bar', 'stack': 'total', 'itemStyle': {'color': COLOR_MAP['pro']}, 'data': df['pro'].round(1).tolist()},
            {'name': 'neutral', 'type': 'bar', 'stack': 'total', 'itemStyle': {'color': COLOR_MAP['neutral']}, 'data': df['neutral'].round(1).tolist()},
            {'name': 'anti', 'type': 'bar', 'stack': 'total', 'itemStyle': {'color': COLOR_MAP['anti']}, 'data': df['anti'].round(1).tolist()},
        ]
    }

def reload_graph(outlet: str, chart_element: ui.echart):
    try:
        start_str = f"{selections['start_month']} {selections['start_year']}"
        end_str = f"{selections['end_month']} {selections['end_year']}"

        if outlet in OUTLET_TO_DOMAIN:
            domain = OUTLET_TO_DOMAIN[outlet]
        else: 
            chart_element.options.clear()
            chart_element.options.update({
            'title': {
                'text': 'Data unavailable',
                'left': 'center', 
                'top': 'center'
            },
            'xAxis': {'show': False}, 'yAxis': {'show': False}
        })
            return

        start_date = dt.strptime(start_str, "%B %Y")
        end_date = dt.strptime(end_str, "%B %Y")
        _, last_day = calendar.monthrange(end_date.year, end_date.month)
        end_date = end_date.replace(day=last_day, hour=23, minute=59, second=59)
        
        if start_date > end_date:
            ui.notify("Start date must be before end date!", type='warning')
            return

        graph_data = get_similarity_graph_data(start_date, end_date, focus_domain=domain)
        new_options = get_graph_options(graph_data['nodes'], graph_data['links'], meta=graph_data.get('meta', {}))
        chart_element.options.clear()
        chart_element.options.update(new_options)
        
    except Exception as e:
        ui.notify(f"Graph Error: {str(e)}", type='negative')
        print(f"Error: {e}")

def reload_bar(outlet: str, chart_element: ui.echart):
    try:
        start_str = f"{selections['start_month']} {selections['start_year']}"
        end_str = f"{selections['end_month']} {selections['end_year']}"
        scale = selections['scale']

        if outlet in OUTLET_TO_DOMAIN:
            domain = OUTLET_TO_DOMAIN[outlet]
        else: 
            chart_element.options.clear()
            chart_element.options.update({
            'title': {
                'text': 'Data unavailable',
                'left': 'center', 
                'top': 'center'
            },
            'xAxis': {'show': False}, 'yAxis': {'show': False}
        })
            return

        start_date = dt.strptime(start_str, "%B %Y")
        end_date = dt.strptime(end_str, "%B %Y")
        _, last_day = calendar.monthrange(end_date.year, end_date.month)
        end_date = end_date.replace(day=last_day, hour=23, minute=59, second=59)
        
        if start_date > end_date:
            ui.notify("Start date must be before end date!", type='warning')
            return

        df = get_stance_series(start_date, end_date, scale, domain)
        new_options = get_plot_options(df)
        chart_element.options.clear()
        chart_element.options.update(new_options)

    except Exception as e:
        ui.notify(f"Plot Error: {str(e)}", type='negative')
        print(f"Error: {e}")


def create_session(outlet:str):
    with ui.row().classes('w-full justify-center'):
        ui.markdown(f"#### **{outlet}**")

        with ui.tabs().classes('w-full rounded-lg') as tabs:
            graph = ui.tab('Graph', label='Similarity', icon='sym_r_bubble_chart')
            plot = ui.tab('Plots', label='Time series', icon='sym_r_stacked_bar_chart')

        with ui.tab_panels(tabs, value=graph).classes('w-full'):
            with ui.tab_panel(graph):
                with ui.card().classes('w-full p-0 border-2 h-110 relative'):
                    graph_chart = ui.echart({'title': {'text': ''}}).classes('h-full w-full')
                    
                    graph_spinner = ui.spinner(size='4em', color='DeepSkyBlue') \
                        .classes('absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 z-10') \
                        .props('thickness=5') 
                    graph_spinner.set_visibility(False)

            with ui.tab_panel(plot):
                with ui.card().classes('w-full p-0 border-2 h-110 relative'):
                    stance_chart = ui.echart({'title': {'text': ''}}).classes('h-full w-full') 
                    
                    stance_spinner = ui.spinner(size='4em', color='DeepSkyBlue') \
                        .classes('absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 z-10') \
                        .props('thickness=5')
                    stance_spinner.set_visibility(False)

        with ui.column().classes('w-full mt-2'):
            with ui.row().classes('items-center no-wrap self-center'):
                ui.select(options=MONTHS, label='Start Month', value=MONTHS[0]).classes('w-32').bind_value(selections, 'start_month')
                ui.select(options=YEARS, label='Start Year', value=YEARS[0]).classes('w-32').bind_value(selections, 'start_year')

            with ui.row().classes('items-center no-wrap self-center'):
                ui.select(options=MONTHS, label='End Month', value=MONTHS[-1]).classes('w-32').bind_value(selections, 'end_month')
                ui.select(options=YEARS, label='End Year', value=YEARS[-1]).classes('w-32').bind_value(selections, 'end_year')

            with ui.row().classes('items-center no-wrap self-center'):
                ui.select(options=SCALES, label='Scale', value=SCALES[-1]).classes('w-32 self-center').bind_value(selections, 'scale')\
                    .bind_value(selections, 'scale')\
                    .bind_enabled_from(tabs, 'value', backward=lambda v: v == 'Plots')
                ui.button(icon='sym_r_replay', on_click=lambda: handle_reload(outlet, stance_chart, graph_chart, graph_spinner, stance_spinner), color='DeepSkyBlue').classes('self-right text-white')
        
    ui.timer(0.1, lambda: handle_reload(outlet, stance_chart, graph_chart, graph_spinner, stance_spinner), once=True)