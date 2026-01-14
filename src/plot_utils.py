import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd

# Configurações
COLOR_SL = "#2354A1"
COLOR_NF = "#C23138"

FIGURE_SIZE = (14, 16)
DPI = 140
FONT_SIZE_LABELS = 14
FONT_SIZE_TICKS = 12
LINE_WIDTH = 2
LINE_WIDTH_PLOT = 1.5
BORDER_WIDTH = 2.0

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
    ax.set_xlabel(xlabel, fontsize=FONT_SIZE_LABELS)
    
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
    else:
        plt.show()