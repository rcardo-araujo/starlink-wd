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
    
    return df

def format_starlink_ts(df: pd.DataFrame):
    # Tratamento para horário local
    df['dt_utc'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
    df['dt_local'] = df['dt_utc'].dt.tz_convert('America/Sao_Paulo')

    # Minutos Decorridos
    df = df.sort_values('timestamp')
    t0 = df['timestamp'].min()
    df['elapsed_min'] = (df['timestamp'] - t0) / 60

    df['hora'] = df['dt_local'].dt.hour
    df['dia'] = df['dt_local'].dt.strftime('%d/%m')

    return df