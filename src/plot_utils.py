import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import os
from scipy import stats

# Configurações
COLOR_SL = "#2354A1"
COLOR_NF = "#C23138"
COLOR_GREEN = "#80A123"

FIGURE_SIZE = (10, 10)
DPI = 140
FONT_SIZE_LABELS = 14
FONT_SIZE_TICKS = 12
LINE_WIDTH = 2
LINE_WIDTH_PLOT = 1.5
BORDER_WIDTH = 2.0

BOXPLOT_PALETTE = "Blues_d" 

def apply_plot_config():
    plt.rcParams.update({
        'figure.figsize': FIGURE_SIZE,
        'figure.dpi': DPI,
        'axes.labelsize': FONT_SIZE_LABELS,
        'xtick.labelsize': FONT_SIZE_TICKS,
        'ytick.labelsize': FONT_SIZE_TICKS,
        'legend.fontsize': FONT_SIZE_TICKS,
        'lines.linewidth': LINE_WIDTH,
        'axes.titlesize': FONT_SIZE_LABELS + 2
    })

def apply_axis_style(ax, xlabel, ylabel, color_y='black'):
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FONT_SIZE_LABELS)
    else:
        ax.set_xlabel("", fontsize=FONT_SIZE_LABELS) 

    ax.set_ylabel(ylabel, fontsize=FONT_SIZE_LABELS)
    
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICKS)
    ax.tick_params(axis='y', labelcolor=color_y)
    
    ax.grid(True, linestyle='-', alpha=0.5)

    for side in ['top', 'bottom', 'left', 'right']:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color('black')
        ax.spines[side].set_linewidth(BORDER_WIDTH)

def plot_correlation_series(df_sl, df_nf, output_folder, tag=""):
    apply_plot_config()
    
    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    if not df_sl.empty and not df_nf.empty:
        t0 = min(df_sl.index.min(), df_nf.index.min())
    else:
        # Fallback seguro se um dos DFs estiver vazio
        if not df_sl.empty: t0 = df_sl.index.min()
        elif not df_nf.empty: t0 = df_nf.index.min()
        else: return # Nada a plotar

    x_sl = (df_sl.index - t0).total_seconds() / 60.0
    x_nf = (df_nf.index - t0).total_seconds() / 60.0

    # Configuração dos Plots
    plots_config = [
        (
            'Signal Quality (SNR) vs Flow Volume',    
            'is_snr_above_noise_floor', 'SNR > Noise Floor', 
            'fluxos_por_segundo', 'Flows/s',
            'corr_snr_vs_volume.png'
        ),
        (
            'Latency vs Flow Duration',              
            'popPingLatencyMs', 'Latency (ms)',       
            'duracao_media_s', 'Avg Duration (s)',    
            'corr_latency_vs_duration.png'
        ),
        (
            'Packet Loss vs Flow Size',               
            'pingDropRate', 'Packet Loss Rate',       
            'bytes_medio', 'Avg Bytes/Flow',          
            'corr_loss_vs_size.png'
        ),
        (
            'Obstruction vs Short Flows',             
            'fraction_obstructed', 'Obstruction',    
            'pct_fluxos_curtos', '% Short Flows',     
            'corr_obstruction_vs_short.png'
        )
    ]
    
    for title, sl_col, sl_lbl, nf_col, nf_lbl, filename in plots_config:
        
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
  
        if sl_col in df_sl.columns:
            ax.plot(x_sl, df_sl[sl_col], color=COLOR_SL, linewidth=LINE_WIDTH_PLOT, label=sl_lbl)
            apply_axis_style(ax, 'Elapsed time (min)', sl_lbl, color_y=COLOR_SL)
        else:
            ax.text(0.5, 0.5, f"Dados ausentes: {sl_col}", ha='center', transform=ax.transAxes)

        ax2 = ax.twinx()
        if nf_col in df_nf.columns:
            ax2.plot(x_nf, df_nf[nf_col], color=COLOR_NF, linestyle='-', linewidth=LINE_WIDTH_PLOT, label=nf_lbl)
            apply_axis_style(ax2, 'Elapsed time (min)', nf_lbl, color_y=COLOR_NF)
            ax2.grid(False)

        lines_1, labels_1 = ax.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', frameon=True)
        
        full_title = f"{title} ({tag})" if tag else title

        if tag:
            base, ext = os.path.splitext(filename)
            final_filename = f"{base}_{tag}{ext}" 
        else:
            final_filename = filename

        if output_folder:
            save_path = os.path.join(output_folder, final_filename)
            plt.savefig(save_path, dpi=DPI)
        
        plt.close(fig)

def classify_by_quantiles(df):
    try:
        df['link_state'] = pd.qcut(
            df['popPingLatencyMs'], 
            q=3, 
            labels=['High SNR', 'Medium SNR', 'Low SNR']
        )
        df['link_state'] = df['link_state'].astype(str)
        
        # Handover: 10% piores casos de perda
        loss_threshold = df['pingDropRate'].quantile(0.90)
        if loss_threshold == 0: loss_threshold = 0.001
            
        mask_handover = df['pingDropRate'] > loss_threshold
        df.loc[mask_handover, 'link_state'] = 'Handover/Instability'
        
        return df
        
    except Exception as e:
        print(f"Erro na classificação: {e}. Usando fallback.")
        df['link_state'] = 'High SNR'
        return df
        
    except Exception as e:
        print(f"Erro na classificação por quantis: {e}")
        df['link_state'] = 'High SNR'
        return df, 0

def plot_physical_states_boxplot(df_sl, df_nf, output_folder):
    apply_plot_config()

    df = pd.merge(df_sl, df_nf, left_index=True, right_index=True, how='inner')
    
    if df.empty:
        print("Erro: Dataframe vazio.")
        return
        
    df = classify_by_quantiles(df)
    
    order_states = ['High SNR', 'Medium SNR', 'Low SNR', 'Handover/Instability']
    existing = df['link_state'].unique()
    final_order = [s for s in order_states if s in existing]

    metrics = [
        ('throughput_bps', 'Throughput (bps)', 'box_troughput_snrt.png'),
        ('duracao_media_s', 'Average Flow Duration (s)', 'box_duration_snr.png'),
        ('pacotes_medio', 'Average Packets per Flow', 'box_packets_snr.png') 
    ]

    for col_y, label_y, filename in metrics:
        if col_y not in df.columns:
            print(f"Skipping {col_y}: coluna não encontrada.")
            continue

        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        
        sns.boxplot(
            data=df, 
            x='link_state', 
            y=col_y, 
            hue='link_state',
            legend=False,
            order=final_order,
            palette=BOXPLOT_PALETTE,
            ax=ax,
            linewidth=2,
            showfliers=False
        )
        
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        apply_axis_style(ax, "", label_y)
        
        save_path = os.path.join(output_folder, filename)
        
        fig.set_size_inches(FIGURE_SIZE)
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        plt.close(fig)

def prepare_scatter_data(df_sl, df_nf):
    df = pd.merge(df_sl, df_nf, left_index=True, right_index=True, how='inner')
    
    if 'jitter_ms' not in df.columns:
        df['jitter_ms'] = df['popPingLatencyMs'].rolling(window=3).std().fillna(0)
        
    return df

def plot_correlation_scatter(df_sl, df_nf, output_folder):
    apply_plot_config()
    
    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder)

    df = prepare_scatter_data(df_sl, df_nf)
    
    if df.empty:
        print("Erro: DataFrame vazio.")
        return

    pairs = [
        ('popPingLatencyMs', 'Latency (ms)', 
         'duracao_media_s', 'Average Flow Duration (s)', 
         'scatter_latency_vs_duration.png'),

        ('jitter_ms', 'Jitter (Latency Std Dev)', 
         'throughput_bps', 'Throughput (bps)', 
         'scatter_jitter_vs_throughput.png'),

        ('pingDropRate', 'Packet Loss Rate', 
         'pacotes_medio', 'Average Packets per Flow', 
         'scatter_loss_vs_packets.png'),
         
        ('popPingLatencyMs', 'Latency (ms)', 
         'throughput_bps', 'Throughput (bps)', 
         'scatter_latency_vs_throughput.png')
    ]

    for col_x, label_x, col_y, label_y, filename in pairs:
        if col_x not in df.columns or col_y not in df.columns:
            continue

        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        
        sns.scatterplot(
            data=df, x=col_x, y=col_y, 
            ax=ax, color=COLOR_NF, s=100, alpha=0.7, edgecolor='black', linewidth=0.8
        )
        
        clean_data = df[[col_x, col_y]].dropna()
        if len(clean_data) > 2:
            r, p_value = stats.pearsonr(clean_data[col_x], clean_data[col_y])
            
            text_str = f'Pearson r = {r:.2f}'
            ax.text(0.95, 0.95, text_str, transform=ax.transAxes, fontsize=16,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        apply_axis_style(ax, label_x, label_y)
        
        save_path = os.path.join(output_folder, filename)
        fig.set_size_inches(FIGURE_SIZE)
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        plt.close(fig)

def classify_simple_status(df):
    lat_threshold = df['popPingLatencyMs'].quantile(0.75) 
    loss_threshold = df['pingDropRate'].quantile(0.90)    
    if loss_threshold == 0: loss_threshold = 0.001

    def get_status(row):
        if row['pingDropRate'] > loss_threshold or row['popPingLatencyMs'] > lat_threshold:
            return 'Degraded'
        else:
            return 'Stable'

    df['network_status'] = df.apply(get_status, axis=1)
    return df

def plot_flow_duration_histogram(df_sl, df_nf, output_folder="histograms"):
    apply_plot_config()
    
    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder)

    df = pd.merge(df_sl, df_nf, left_index=True, right_index=True, how='inner')

    df = classify_simple_status(df)

    if 'duracao_media_s' not in df.columns:
        print("Erro: Coluna 'duracao_media_s' não encontrada.")
        return

    df = df[df['duracao_media_s'] > 0.1]

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    
    sns.histplot(
        data=df,
        x="duracao_media_s",
        hue="network_status",
        hue_order=["Stable", "Degraded"],
        palette={"Stable": COLOR_SL, "Degraded": COLOR_NF},
        element="step",     
        stat="percent",     
        common_norm=False,   
        bins=30,             
        alpha=0.4,           
        ax=ax
    )
    
    sns.move_legend(ax, "upper right", title=None, frameon=True)

    apply_axis_style(ax, "Average Flow Duration (s)", "Frequency (%)")
    
    save_path = os.path.join(output_folder, "hist_flow_duration.png")
    fig.set_size_inches(FIGURE_SIZE)
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)

def plot_temporal_heatmap(df_sl, df_nf, output_folder="heatmaps"):
    apply_plot_config()
    
    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder)

    df = pd.merge(df_sl, df_nf, left_index=True, right_index=True, how='inner')
    
    if df.empty:
        print("Erro: DataFrame vazio.")
        return

    df['hour'] = df.index.hour

    df['date_str'] = df.index.strftime('%Y-%m-%d')

    metrics = [
        ('popPingLatencyMs', 'Avg Latency (ms)', 'coolwarm', 'heat_latency.png'),
        ('pingDropRate', 'Packet Loss Rate (Failure)', 'Reds', 'heat_packet_loss.png')
    ]

    for col, cbar_label, cmap, filename in metrics:
        if col not in df.columns:
            continue
            
        heatmap_data = df.pivot_table(index='date_str', columns='hour', values=col, aggfunc='mean')

        num_days = len(heatmap_data)
        height = max(6, num_days * 0.5) 
        fig, ax = plt.subplots(figsize=(12, height))

        # 3. Plotar Heatmap
        sns.heatmap(
            heatmap_data,
            cmap=cmap,
            annot=False,
            fmt=".1f" if col == 'popPingLatencyMs' else ".3f",
            linewidths=0.5,
            linecolor='white',
            cbar_kws={'label': ''}, 
            ax=ax
        )

        current_y_labels = heatmap_data.index
        new_y_labels = [pd.to_datetime(d).strftime('%d/%m') for d in current_y_labels]
        ax.set_yticklabels(new_y_labels, rotation=45)

        ax.set_ylabel("") 

        ax.set_xlabel("Hour of Day", fontsize=FONT_SIZE_LABELS, fontweight='normal', labelpad=15)
        plt.xticks(rotation=0) 

        save_path = os.path.join(output_folder, filename)
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        plt.close(fig)

def extract_event_windows(df, window_pre=3, window_post=6):
    thresh_loss = df['pingDropRate'].quantile(0.90)
    if thresh_loss == 0: thresh_loss = 0.005 # Fallback
    
    # Cria uma série booleana: 1 se for evento, 0 se não
    is_event = (df['pingDropRate'] > thresh_loss) | (df['fraction_obstructed'] > 0.001)
    
    event_starts = is_event.astype(int).diff() == 1
    event_indices = df.index[event_starts]
    
    if len(event_indices) == 0:
        return pd.DataFrame() # Sem eventos

    slices = []

    df_reset = df.reset_index()
    
    for start_time in event_indices:
        idx_loc = df.index.get_loc(start_time)

        idx_min = max(0, idx_loc - window_pre)
        idx_max = min(len(df), idx_loc + window_post + 1)
        
        subset = df.iloc[idx_min:idx_max].copy()

        t0 = df.index[idx_loc]
        subset['rel_time_min'] = (subset.index - t0).total_seconds() / 60.0

        subset['event_id'] = t0
        
        slices.append(subset)
        
    if not slices:
        return pd.DataFrame()
        
    return pd.concat(slices)

def plot_recovery_analysis(df_sl, df_nf, output_folder="recovery_plots"):
    apply_plot_config()
    
    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder)

    df = pd.merge(df_sl, df_nf, left_index=True, right_index=True, how='inner')
    
    if 'throughput_bps' not in df.columns:
        duration = df['duracao_media_s'].replace(0, 0.001)
        df['throughput_bps'] = (df['bytes_medio'] * df['count_fluxos'] * 8) / duration
    df['throughput_mbps'] = df['throughput_bps'] / 1e6
    
    try:
        window_sec = df.index.to_series().diff().median().total_seconds()
        if pd.isna(window_sec) or window_sec == 0: window_sec = 600
    except:
        window_sec = 600
        
    df['flows_per_sec'] = df['count_fluxos'] / window_sec

    df_aligned = extract_event_windows(df, window_pre=4, window_post=8)
    
    if df_aligned.empty:
        print("Aviso: Nenhum evento significativo encontrado para gerar o gráfico de recuperação.")
        return

    fig, axes = plt.subplots(3, 1, figsize=FIGURE_SIZE, sharex=True)
    
    metrics = [
        ('throughput_mbps', 'Throughput (Mbps)'),
        ('flows_per_sec', 'Flow Initiation Rate (flows/s)'),
        ('duracao_media_s', 'Avg Flow Duration (s)')
    ]
    
    for ax, (col, label) in zip(axes, metrics):
        sns.lineplot(
            data=df_aligned,
            x='rel_time_min',
            y=col,
            ax=ax,
            color=COLOR_SL,
            linewidth=2.5,
            errorbar=None
        )
        
        ax.axvline(x=0, color=COLOR_NF, linestyle='--', linewidth=2, label='Event Onset (t₀)')
        
        ax.set_ylabel(label, fontsize=FONT_SIZE_LABELS)
        ax.grid(True, linestyle='--', alpha=0.5)

        if ax != axes[-1]:
            ax.set_xlabel("")
    
    axes[0].legend(loc='upper right')
    
    axes[-1].set_xlabel("Time Relative to Event (minutes)", fontsize=FONT_SIZE_LABELS)

    plt.tight_layout()
    
    save_path = os.path.join(output_folder, "recovery_analysis.png")
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)

PALETTE_PHASES = {
    "Before": COLOR_SL, 
    "During": COLOR_NF, 
    "After":  COLOR_GREEN 
}

def prepare_phases_data(df, window_intervals=3):
    thresh_loss = df['pingDropRate'].quantile(0.95)
    if thresh_loss == 0: thresh_loss = 0.005

    is_event = (df['pingDropRate'] > thresh_loss) | (df.get('fraction_obstructed', 0) > 0)

    state_changes = is_event.astype(int).diff()
    
    starts = df.index[state_changes == 1]
    ends = df.index[state_changes == -1]

    if len(starts) == 0 and len(ends) == 0:
        return pd.DataFrame() 
        
    phases_data = []

    for start in starts:
        valid_ends = ends[ends > start]
        if len(valid_ends) == 0:
            end = df.index[-1] 
        else:
            end = valid_ends[0]
            
        idx_start = df.index.get_loc(start)
        idx_end = df.index.get_loc(end)

        idx_b_start = max(0, idx_start - window_intervals)
        df_before = df.iloc[idx_b_start : idx_start].copy()
        df_before['Phase'] = 'Before'

        df_during = df.iloc[idx_start : idx_end + 1].copy()
        df_during['Phase'] = 'During'

        idx_a_end = min(len(df), idx_end + 1 + window_intervals)
        df_after = df.iloc[idx_end + 1 : idx_a_end].copy()
        df_after['Phase'] = 'After'
        
        phases_data.extend([df_before, df_during, df_after])
        
    if not phases_data:
        return pd.DataFrame()
        
    df_viz = pd.concat(phases_data)
    return df_viz

def plot_causal_violin(df_sl, df_nf, output_folder="causal_plots"):
    apply_plot_config()
    
    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder)

    df = pd.merge(df_sl, df_nf, left_index=True, right_index=True, how='inner')
    
    if 'throughput_bps' not in df.columns:
        duration = df['duracao_media_s'].replace(0, 0.001)
        if 'in_bytes' in df.columns:
             df['throughput_bps'] = (df['in_bytes'] * 8) / duration
        else:
             df['throughput_bps'] = (df['bytes_medio'] * df['count_fluxos'] * 8) / duration
    df['throughput_mbps'] = df['throughput_bps'] / 1e6

    df_viz = prepare_phases_data(df, window_intervals=6)
    
    if df_viz.empty:
        print("Aviso: Nenhum evento detectado para análise causal (Before/After).")
        return

    metrics = [
        (
            'throughput_mbps', 
            'Throughput (Mbps)', 
            'violin_handover_vs_throughput.png'
        ),
        (
            'duracao_media_s', 
            'Flow Duration (s)', 
            'violin_handover_vs_duration.png'
        )
    ]
    
    order_list = ['Before', 'During', 'After']

    for col, label, filename in metrics:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.violinplot(
            data=df_viz,
            x='Phase',
            y=col,
            order=order_list,
            palette=PALETTE_PHASES,
            ax=ax,
            inner='box',
            linewidth=1.5,
            alpha=0.8
        )
        
        ax.set_ylabel(label, fontsize=12)
        ax.set_xlabel("Handover Phase", fontsize=12, labelpad=10)
        ax.grid(True, axis='y', linestyle='-', alpha=0.5)

        save_path = os.path.join(output_folder, filename)
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        plt.close(fig) 

def plot_capacity_elasticity(df_sl, df_nf, output_folder="capacity_plots", use_log_scale=True):
    apply_plot_config()
    if output_folder and not os.path.exists(output_folder): os.makedirs(output_folder)

    df = pd.merge(df_sl, df_nf, left_index=True, right_index=True, how='inner')
    
    if df.empty:
        print("Erro: DataFrame vazio após merge. Verifique se os índices de tempo são idênticos.")
        return

    if 'downloadBps' in df.columns:
        df['sl_mbps'] = df['downloadBps'] / 1e6
    else:
        print("Erro: 'downloadBps' não encontrado.")
        return

    if 'in_bytes' in df.columns:
        vol_bytes = df['in_bytes']
    elif 'bytes_medio' in df.columns and 'count_fluxos' in df.columns:
        vol_bytes = df['bytes_medio'] * df['count_fluxos']
    else:
        print("Erro: Colunas de bytes não encontradas no NetFlow.")
        return

    df['nf_mbps'] = (vol_bytes * 8) / 600 / 1e6

    fig, ax = plt.subplots(figsize=(6, 6))
    
    sns.scatterplot(
        data=df,
        x='sl_mbps',
        y='nf_mbps',
        ax=ax,
        color=COLOR_SL,
        alpha=0.6,
        s=50,
        linewidth=0,
        label='10min samples'
    )

    max_val = max(df['sl_mbps'].max(), df['nf_mbps'].max())
    limit = max(max_val * 1.2, 1.0)
    
    ax.plot([0, limit], [0, limit], color=COLOR_NF, linestyle='--', linewidth=2, label='y=x (Saturation)')

    if use_log_scale:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(left=0.01, right=limit)
        ax.set_ylim(bottom=0.01, top=limit)
        scale_txt = "(Log Scale)"
    else:
        ax.set_xlim(0, limit)
        ax.set_ylim(0, limit)
        scale_txt = "(Linear Scale)"

    ax.set_xlabel("Starlink Dish Download (Mbps)", fontsize=12, labelpad=10)
    ax.set_ylabel("Router NetFlow Usage (Mbps)", fontsize=12, labelpad=10)
    
    ax.grid(True, which="major", linestyle='-', alpha=0.5)
    ax.legend()

    plt.tight_layout()
    filename = "scatter_capacity_usage.png"
    save_path = os.path.join(output_folder, filename)
    plt.savefig(save_path, dpi=140)
    plt.close(fig)

def plot_host_impact_scatter(df_sl, df_nf, output_folder="host_plots", top_n=20):
    apply_plot_config()
    
    if output_folder and not os.path.exists(output_folder): os.makedirs(output_folder)

    possible_cols = [
        'src4_addr', 'src6_addr',       # Seus nomes de coluna (Prioridade)
        'dst4_addr', 'dst6_addr',       # Destino (caso queira analisar quem recebe)
        'sa', 'src_ip', 'srcip', 'source_address', 
        'da', 'dst_ip', 'dstip'
    ]

    host_col = next((c for c in possible_cols if c in df_nf.columns), None)
    
    FREQ = '10min'
    
    # Prepara Starlink
    df_sl_agg = df_sl.resample(FREQ).agg({'popPingLatencyMs': 'mean'})
    df_sl_agg = df_sl_agg.rename(columns={'popPingLatencyMs': 'window_latency'})
    
    # Prepara NetFlow
    df_nf_proc = df_nf.copy()
    
    if not isinstance(df_nf_proc.index, pd.DatetimeIndex):
        if 'first_local' in df_nf_proc.columns:
            df_nf_proc = df_nf_proc.set_index('first_local')
        elif 'first_utc' in df_nf_proc.columns:
             df_nf_proc = df_nf_proc.set_index('first_utc')
    
    df_nf_proc['time_bin'] = df_nf_proc.index.round(FREQ)
    
    # Garante in_bytes
    if 'in_bytes' not in df_nf_proc.columns:
        # Tenta usar bytes_por_fluxo se existir
        if 'bytes_por_fluxo' in df_nf_proc.columns:
            df_nf_proc['in_bytes'] = df_nf_proc['bytes_por_fluxo']
        else:
            df_nf_proc['in_bytes'] = 0

    df_merged = pd.merge(
        df_nf_proc.reset_index(), 
        df_sl_agg, 
        left_on='time_bin', 
        right_index=True, 
        how='left'
    )
    
    df_merged = df_merged.dropna(subset=['window_latency'])

    df_hosts = df_merged.groupby(host_col).agg({
        'in_bytes': 'sum',
        'window_latency': 'mean',
        'time_bin': 'count'
    }).rename(columns={'time_bin': 'flow_count', 'window_latency': 'avg_latency_seen'})
    
    # 5. FILTRAGEM
    df_top = df_hosts.nlargest(top_n, 'in_bytes')
    
    df_top['vol_mb'] = df_top['in_bytes'] / 1e6

    # 6. PLOTAGEM
    fig, ax = plt.subplots(figsize=(10, 8))

    if df_top['avg_latency_seen'].nunique() <= 1:
        print("Aviso: Todos os hosts viram a mesma latência média (pouca variação temporal).")

    sns.scatterplot(
        data=df_top,
        x='avg_latency_seen',
        y='vol_mb',
        size='flow_count',
        hue='flow_count',
        sizes=(200, 2000),
        palette='Blues_d',
        alpha=0.7,
        ax=ax,
        edgecolor='black',
        legend='brief'
    )

    for line in range(0, df_top.shape[0]):
        ax.text(
            df_top.avg_latency_seen.iloc[line], 
            df_top.vol_mb.iloc[line], 
            str(df_top.index[line]),
            horizontalalignment='left', 
            size='small', 
            color='black', 
            ha='center'
        )

    ax.set_xlabel("Avg Starlink Latency Experienced (ms)", fontsize=12, labelpad=10)
    ax.set_ylabel("Total Transferred Volume (MB)", fontsize=12, labelpad=10)
    
    ax.set_yscale('log')
    ax.grid(True, linestyle='-', alpha=0.5)
    
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, title="Flow Count", labelspacing=1.0, markerscale=0.5, title_fontsize=14, borderpad=0.7)
    
    plt.tight_layout()
    
    filename = "scatter_host_vs_quality.png"
    save_path = os.path.join(output_folder, filename)
    plt.savefig(save_path, dpi=140, bbox_inches='tight')
    plt.close(fig)

def plot_availability_gap(df_sl_agg, df_nf_agg, output_folder="availability_plots"):
    apply_plot_config()
    if output_folder and not os.path.exists(output_folder): os.makedirs(output_folder)

    df = pd.merge(df_sl_agg, df_nf_agg, left_index=True, right_index=True, how='inner')
    
    if df.empty:
        print("Erro: DataFrame vazio após merge. Verifique o alinhamento temporal.")

    df['phys_avail_pct'] = (1 - df['pingDropRate'].clip(0, 1)) * 100

    if 'pct_fluxos_curtos' in df.columns:
        df['perceived_avail_pct'] = 100 - df['pct_fluxos_curtos']
    else:
        print("Aviso: 'pct_fluxos_curtos' não encontrado. Usando Throughput normalizado como fallback.")
        max_th = df['throughput_bps'].max()
        df['perceived_avail_pct'] = (df['throughput_bps'] / max_th) * 100

    df['phys_smooth'] = df['phys_avail_pct'].rolling(3, center=True).mean()
    df['perc_smooth'] = df['perceived_avail_pct'].rolling(3, center=True).mean()

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.fill_between(
        df.index, 
        df['phys_smooth'], 
        color=COLOR_GREEN, 
        alpha=0.3, 
        label='Physical Link Availability (Starlink)'
    )
    ax.plot(df.index, df['phys_smooth'], color='#2E7D32', linewidth=1.5, alpha=0.8)

    # Linha Azul (Percebido)
    ax.plot(
        df.index, 
        df['perc_smooth'], 
        color=COLOR_SL, 
        linewidth=2.5, 
        label='Perceived Success (% Long Flows)'
    )
    
    gap = df['phys_smooth'] - df['perc_smooth']
    ax.fill_between(
        df.index, 
        df['phys_smooth'], 
        df['perc_smooth'],
        where=(gap > 20), 
        color=COLOR_NF,  
        alpha=0.3,
        interpolate=True,
        label='Quality Gap (Phantom Availability)'
    )

    ax.set_ylim(0, 105)
    ax.set_ylabel("Availability / Success Rate (%)", fontsize=12, labelpad=10)
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%d/%m'))
    ax.grid(True, linestyle='-', alpha=0.5)
    
    ax.legend(loc='upper left', frameon=True, framealpha=0.9)

    plt.tight_layout()
    
    filename = "link_availability_gap.png"
    save_path = os.path.join(output_folder, filename)
    plt.savefig(save_path, dpi=140, bbox_inches='tight')
    plt.close(fig)