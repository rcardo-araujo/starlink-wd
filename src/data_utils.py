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

def netflow_agg(df: pd.DataFrame, freq='10min'):
    # Define o índice temporal para o resample
    # Usamos 'first_local' como referência de quando o fluxo começou
    df_agg = df.set_index('first_local').resample(freq).agg({
        'first_utc': 'count',       # Conta quantos fluxos ocorreram (Volume)
        'flow_duration_s': 'mean',  # Duração média
        'bytes_por_fluxo': 'mean',  # Tamanho médio
        'is_short_flow': 'mean'     # Fração de fluxos curtos (0 a 1)
    })
    
    # Renomeia para ficar claro no plot
    df_agg = df_agg.rename(columns={
        'first_utc': 'count_fluxos',
        'flow_duration_s': 'duracao_media_s',
        'bytes_por_fluxo': 'bytes_medio',
        'is_short_flow': 'pct_fluxos_curtos'
    })
    
    # Cálculos derivados
    seconds = pd.to_timedelta(freq).total_seconds()
    df_agg['fluxos_por_segundo'] = df_agg['count_fluxos'] / seconds
    df_agg['pct_fluxos_curtos'] = df_agg['pct_fluxos_curtos'] * 100 # Transforma 0.5 em 50%
    
    return df_agg