import pandas as pd
import numpy as np

def format_netflow_ts(df: pd.DataFrame):
    # Conversão temporal
    df['first_utc'] = pd.to_datetime(df['first'], utc=True)
    df['last_utc']  = pd.to_datetime(df['last'], utc=True)
    
    df['first_local'] = df['first_utc'].dt.tz_convert('America/Sao_Paulo')
    df['last_local']  = df['last_utc'].dt.tz_convert('America/Sao_Paulo')
    
    # Duração do fluxo
    df['flow_duration_s'] = (
        df['last_local'] - df['first_local']
    ).dt.total_seconds()
    
    df['bytes_por_fluxo'] = df['in_bytes']
    df['is_short_flow'] = df['flow_duration_s'] < 1

    df = df.set_index('first_local').sort_index()
    
    return df

def format_starlink_ts(df: pd.DataFrame):
    # Tratamento para horário local
    df['dt_utc'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
    df['dt_local'] = df['dt_utc'].dt.tz_convert('America/Sao_Paulo')
    
    df = df.set_index('dt_local')

    # Minutos Decorridos
    df = df.sort_values('timestamp')
    t0 = df['timestamp'].min()
    df['elapsed_min'] = (df['timestamp'] - t0) / 60

    df['hora'] = df.index.hour
    df['dia']  = df.index.strftime('%d/%m')

    return df

def netflow_agg(df: pd.DataFrame, freq='10min'):
    df_agg = df.resample(freq).agg({
        'first_utc': 'count',       
        'flow_duration_s': 'mean',  
        'bytes_por_fluxo': 'mean', 
        'is_short_flow': 'mean'     
    })
    
    df_agg = df_agg.rename(columns={
        'first_utc': 'count_fluxos',
        'flow_duration_s': 'duracao_media_s',
        'bytes_por_fluxo': 'bytes_medio',
        'is_short_flow': 'pct_fluxos_curtos'
    })
    
    seconds = pd.to_timedelta(freq).total_seconds()
    df_agg['fluxos_por_segundo'] = df_agg['count_fluxos'] / seconds
    df_agg['pct_fluxos_curtos'] = df_agg['pct_fluxos_curtos'] * 100 
    
    return df_agg

def starlink_agg(df: pd.DataFrame, freq='10min'):
    df_agg = df.resample(freq).mean(numeric_only=True)

    return df_agg

def split_by_gap(df_main, df_sec, gap_threshold_hours=4):
    time_diff = df_main.index.to_series().diff()
    
    max_gap_size = time_diff.max()
    gap_end_time = time_diff.idxmax() 

    loc_end = df_main.index.get_loc(gap_end_time)

    if isinstance(loc_end, slice):
        loc_end = loc_end.start
    elif hasattr(loc_end, "__len__"): 
        loc_end = loc_end[0]
    
    gap_start_time = df_main.index[loc_end - 1]
    
    df_main_pre_gap = df_main[df_main.index < gap_start_time]
    df_main_post_gap = df_main[df_main.index >= gap_end_time]
    
    df_sec_pre_gap = df_sec[df_sec.index < gap_start_time]
    df_sec_post_gap = df_sec[df_sec.index >= gap_end_time]
    
    return (df_main_pre_gap, df_sec_pre_gap), (df_main_post_gap, df_sec_post_gap)

def sync_time_boundaries(df1, df2):
    start_common = max(df1.index.min(), df2.index.min())
    end_common = min(df1.index.max(), df2.index.max())
    
    df1_synced = df1[(df1.index >= start_common) & (df1.index <= end_common)]
    df2_synced = df2[(df2.index >= start_common) & (df2.index <= end_common)]
    
    return df1_synced, df2_synced