import pandas as pd
from pathlib import Path

def format_ts(df: pd.DataFrame):
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