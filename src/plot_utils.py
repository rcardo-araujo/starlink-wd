import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import os
from scipy import stats

# Configurações
COLOR_SL = "#2354A1"
COLOR_NF = "#C23138"

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

def plot_correlation_series(df_sl, df_nf, output_path=None):
    apply_plot_config()
    
    if not df_sl.empty and not df_nf.empty:
        t0 = min(df_sl.index.min(), df_nf.index.min())
    else:
        t0 = df_sl.index.min() if not df_sl.empty else df_nf.index.min()

    x_sl = (df_sl.index - t0).total_seconds() / 60.0
    x_nf = (df_nf.index - t0).total_seconds() / 60.0

    fig, axes = plt.subplots(4, 1, figsize=FIGURE_SIZE, sharex=True, constrained_layout=True)
    
    plots_config = [
        (
            'Signal Quality (SNR) vs Flow Volume',   
            'is_snr_above_noise_floor', 'SNR > Noise Floor', 
            'fluxos_por_segundo', 'Flows/s'          
        ),
        (
            'Latency vs Flow Duration',             
            'popPingLatencyMs', 'Latency (ms)',      
            'duracao_media_s', 'Avg Duration (s)'    
        ),
        (
            'Packet Loss vs Flow Size',              
            'pingDropRate', 'Packet Loss Rate',      
            'bytes_medio', 'Avg Bytes/Flow'          
        ),
        (
            'Obstruction vs Short Flows',            
            'fraction_obstructed', 'Obstruction',   
            'pct_fluxos_curtos', '% Short Flows'     
        )
    ]
    
    for ax, (title, sl_col, sl_lbl, nf_col, nf_lbl) in zip(axes, plots_config):
        ax.tick_params(labelbottom=True)
        
        # Eixo Starlink
        if sl_col in df_sl.columns:
            ax.plot(x_sl, df_sl[sl_col], color=COLOR_SL, linewidth=LINE_WIDTH_PLOT, label=sl_lbl)
            apply_axis_style(ax, 'Elapsed time (min)', sl_lbl, color_y=COLOR_SL)
        else:
            ax.text(0.5, 0.5, f"Dados ausentes: {sl_col}", ha='center', transform=ax.transAxes)
        
        # Eixo NetFlow
        ax2 = ax.twinx()
        if nf_col in df_nf.columns:
            ax2.plot(x_nf, df_nf[nf_col], color=COLOR_NF, linestyle='-', linewidth=LINE_WIDTH_PLOT, label=nf_lbl)
            apply_axis_style(ax2, 'Elapsed time (min)', nf_lbl, color_y=COLOR_NF)
            ax2.grid(False)

        lines_1, labels_1 = ax.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', frameon=True)
    
    if output_path:
        plt.savefig(output_path, dpi=DPI)

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