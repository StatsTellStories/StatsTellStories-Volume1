import numpy as np
import pandas as pd
import math

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import seaborn as sns 


def px_lines(df, x, y, title=None, color=None, color_map=None, line_color=None, figsize=None, save_path=None):
    fig = px.line(
        df, 
        x=x, 
        y=y, 
        title=title, 
        color=color, 
        color_discrete_map=color_map
    )
    fig.update_layout(title_x=0.5)
    if line_color:
        fig.update_traces(line_color=line_color)
    if figsize:
        fig.update_layout(width=figsize[0], height=figsize[1])
    if save_path:
        fig.write_image(save_path)
    fig.show()

def px_scatter(df, x, y, title=None, color=None, color_map=None):
    fig = px.scatter(
        df, 
        x=x, 
        y=y, 
        title=title, 
        color=color, 
        color_discrete_map=color_map
    )
    fig.show()
    
# #### 3 d plot
def px_scatter_3d(df, x, y, z, color = None, color_map = None):
    fig = px.scatter_3d(df, x=x, y=y, z=z,
        color=color,
        title=f"Labels: {x} vs {y} vs {z}",
        color_discrete_map=color_map)
    fig.show()
    
    
    
def plot_hist(s: pd.Series, xlabel, ylabel, **kwargs):

    fontsizeTitle = 22
    fontsizeLabel = 20
    fontsizeValues = 18

    if not isinstance(s, pd.Series):
        raise TypeError("plot_hist expects a pandas Series (e.g. df['col']).")

    data = s.dropna()

    figsize       = kwargs.get('figsize', (15, 5))
    save_path     = kwargs.get('save_path', None)
    color         = kwargs.get('color', '#A4262C')
    edgecolor     = kwargs.get('edgecolor', 'white')
    linewidth     = kwargs.get('linewidth', 0.5)
    bins          = kwargs.get('bins', "auto")
    ylim          = kwargs.get('ylim', None)
    transparent   = kwargs.get('transparent', True)
    dpi           = kwargs.get('dpi', 300)
    dark_mode     = kwargs.get('dark_mode', True)
    bg_color      = kwargs.get('bg_color', "#3E3D3D")
    text_color    = kwargs.get('text_color', 'white')
    title         = kwargs.get('title', f"Distribution of {xlabel}")

    fig, ax = plt.subplots(figsize=figsize)

    if dark_mode:
        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)

    ax.hist(
        data,
        bins=bins,
        color=color,
        edgecolor=edgecolor,
        linewidth=linewidth
    )

    if dark_mode:
        ax.set_title(title, fontsize=fontsizeTitle, pad=20, color=text_color)
        ax.set_xlabel(xlabel, fontsize=fontsizeLabel, color=text_color, labelpad=20)
        ax.set_ylabel(ylabel, fontsize=fontsizeLabel, color=text_color, labelpad=20)
        ax.tick_params(axis="y", labelsize=fontsizeValues, colors=text_color)
        ax.tick_params(axis="x", labelsize=fontsizeValues, colors=text_color)
        ax.spines[["left", "bottom"]].set_edgecolor(text_color)
    else:
        ax.set_title(title, fontsize=fontsizeTitle, pad=20)
        ax.set_xlabel(xlabel, fontsize=fontsizeLabel, labelpad=20)
        ax.set_ylabel(ylabel, fontsize=fontsizeLabel, labelpad=20)
        ax.tick_params(axis="y", labelsize=fontsizeValues)
        ax.tick_params(axis="x", labelsize=fontsizeValues)

    ax.spines[["top", "right"]].set_visible(False)

    if ylim is not None:
        ax.set_ylim(ylim)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(
            save_path,
            bbox_inches="tight",
            transparent=transparent,
            dpi=dpi,
            facecolor='none'
        )
    plt.show()


def plot_bar(s: pd.Series, title, xlabel, ylabel, **kwargs):
    fontsizeTitle = 22  
    fontsizeLabel = 20  
    fontsizeValues = 18
    if not isinstance(s, pd.Series):
        raise TypeError("plot_bar expects a pandas Series (e.g. df['col']).")
    
    data = s.dropna()
    
    figsize        = kwargs.get('figsize', (15, 7))
    save_path      = kwargs.get('save_path', None)
    color          = kwargs.get('color', '#A4262C') 
    edgecolor      = kwargs.get('edgecolor', 'white')
    linewidth      = kwargs.get('linewidth', 0.5)
    rotate_xticks  = kwargs.get('rotate_xticks', 90)
    ylim           = kwargs.get('ylim', None)
    transparent    = kwargs.get('transparent', False)
    dpi            = kwargs.get('dpi', 300)
    dark_mode      = kwargs.get('dark_mode', False)
    bg_color       = kwargs.get('bg_color', "#3E3D3D")
    text_color     = kwargs.get('text_color', 'white')
    
    fig, ax = plt.subplots(figsize=figsize)
    if dark_mode:
        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)
    else:
        ax.set_facecolor('#F5F5F5')   # dezenter grauer Plot-Hintergrund
    
    data.plot(
        kind="bar",
        figsize=figsize,
        color=color,
        edgecolor=edgecolor,
        linewidth=linewidth,
        ax=ax
    )
    
    if dark_mode:
        ax.set_title(title, fontsize=fontsizeTitle, pad=20, color=text_color)
        ax.set_xlabel(xlabel, fontsize=fontsizeLabel, color=text_color, labelpad=20)
        ax.set_ylabel(ylabel, fontsize=fontsizeLabel, color=text_color, labelpad=20)
        ax.tick_params(axis="y", labelsize=fontsizeValues, colors=text_color)
        ax.tick_params(axis="x", labelsize=fontsizeValues, rotation=rotate_xticks, colors=text_color)
        ax.spines[["left", "bottom"]].set_edgecolor(text_color)
    else:
        ax.set_title(title, fontsize=fontsizeTitle, pad=20)
        ax.set_xlabel(xlabel, fontsize=fontsizeLabel, labelpad=20)
        ax.set_ylabel(ylabel, fontsize=fontsizeLabel, labelpad=20)
        ax.tick_params(axis="y", labelsize=fontsizeValues)
        ax.tick_params(axis="x", labelsize=fontsizeValues, rotation=rotate_xticks)
    
    ax.grid(axis='y', linestyle='--', alpha=0.6, color=text_color if dark_mode else '#BBBBBB')
    ax.set_axisbelow(True)
    
    ax.spines[["top", "right"]].set_visible(False)
    
    if ylim is not None:
        ax.set_ylim(ylim)
    
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(
            save_path,
            bbox_inches="tight",
            transparent=transparent,
            dpi=dpi,
        )
    plt.show()


def plot_line(df: pd.DataFrame, x: str, y: str, title: str, xlabel: str = None, ylabel: str = None, **kwargs):
    
    if not isinstance(df, pd.DataFrame):
        raise TypeError("plot_line expects a pandas DataFrame.")
    
    if x not in df.columns or y not in df.columns:
        raise ValueError(f"Columns '{x}' and/or '{y}' not found in DataFrame.")

    data = df[[x, y]].dropna()

    xlabel = xlabel or x
    ylabel = ylabel or y

    fontsizeTitle  = 26
    fontsizeLabel  = 24
    fontsizeValues = 22

    figsize     = kwargs.get('figsize', (20, 8))
    color       = kwargs.get('color', '#A4262C')
    save_path   = kwargs.get('save_path', None)
    ylim        = kwargs.get('ylim', None)
    transparent = kwargs.get('transparent', True)
    dpi         = kwargs.get('dpi', 300)
    dark_mode   = kwargs.get('dark_mode', False)
    bg_color    = kwargs.get('bg_color', '#3E3D3D')
    text_color  = kwargs.get('text_color', 'white')
    linewidth   = kwargs.get('linewidth', 3.5)

    fig, ax = plt.subplots(figsize=figsize)

    if dark_mode:
        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)

    ax.plot(data[x], data[y], color=color, linewidth=linewidth)
    ax.set_xlim(data[x].min(), data[x].max())

    if ylim is not None:
        ax.set_ylim(ylim)

    if dark_mode:
        ax.set_title(title, fontsize=fontsizeTitle, pad=20, color=text_color)
        ax.set_xlabel(xlabel, fontsize=fontsizeLabel, labelpad=20, color=text_color)
        ax.set_ylabel(ylabel, fontsize=fontsizeLabel, labelpad=20, color=text_color)
        ax.tick_params(axis="both", labelsize=fontsizeValues, colors=text_color)
        ax.spines[["left", "bottom"]].set_edgecolor(text_color)
        ax.grid(color='#666666', linewidth=0.5, alpha=0.5)
    else:
        ax.set_title(title, fontsize=fontsizeTitle, pad=20)
        ax.set_xlabel(xlabel, fontsize=fontsizeLabel, labelpad=20)
        ax.set_ylabel(ylabel, fontsize=fontsizeLabel, labelpad=20)
        ax.tick_params(axis="both", labelsize=fontsizeValues)
        ax.grid(linewidth=0.5, alpha=0.5)

    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(
            save_path,
            bbox_inches="tight",
            transparent=transparent,
            dpi=dpi,
            facecolor='none'
        )
    plt.show()

def plot_two_lines(data, col1, col2, title, xlabel, ylabel, **kwargs):

    fontsizeTitle  = 24
    fontsizeLabel  = 22
    fontsizeValues = 20

    figsize        = kwargs.get('figsize', (20, 8))
    save_path      = kwargs.get('save_path', None)
    fill_between   = kwargs.get('fill_between', False)
    dpi            = kwargs.get('dpi', 300)
    dark_mode      = kwargs.get('dark_mode', True)
    bg_color       = kwargs.get('bg_color', '#3E3D3D')
    text_color     = kwargs.get('text_color', 'white')
    color1         = kwargs.get('color1', "#D07F0E")
    color2         = kwargs.get('color2', "Slateblue")
    linewidth      = kwargs.get('linewidth', 2.5)
    ylim           = kwargs.get('ylim', None)
    transparent    = kwargs.get('transparent', True)


    fig, ax = plt.subplots(figsize=figsize)

    if dark_mode:
        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)

    if fill_between:
        ax.fill_between(data.index, data[col2], color="skyblue", alpha=0.35)
        ax.fill_between(data.index, data[col1], color=color1, alpha=0.55)
        
    # if fill_between:
    #     ax.fill_between(data.index, data[col2], color="skyblue", alpha=0.2)
    
    ax.plot(data.index, data[col2], color=color2, marker=".", alpha=0.8, linewidth=linewidth, label=col2)

    # if fill_between:
    #     ax.fill_between(data.index, data[col1], color=color1, alpha=0.55)
    
    ax.plot(data.index, data[col1], color=color1, marker=".", alpha=0.8, linewidth=linewidth, label=col1)

    ax.set_xlim(data.index.min(), data.index.max())

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.spines[["top", "right"]].set_visible(False)

    if dark_mode:
        ax.set_title(title, fontsize=fontsizeTitle, pad=20, color=text_color)
        ax.set_xlabel(xlabel, fontsize=fontsizeLabel, labelpad=20, color=text_color)
        ax.set_ylabel(ylabel, fontsize=fontsizeLabel, labelpad=20, color=text_color)
        ax.tick_params(axis="both", labelsize=fontsizeValues, colors=text_color)
        ax.spines[["left", "bottom"]].set_edgecolor(text_color)
        ax.legend(fontsize=fontsizeValues, facecolor=bg_color, labelcolor=text_color, edgecolor=text_color)
    else:
        ax.set_title(title, fontsize=fontsizeTitle, pad=20)
        ax.set_xlabel(xlabel, fontsize=fontsizeLabel, labelpad=20)
        ax.set_ylabel(ylabel, fontsize=fontsizeLabel, labelpad=20)
        ax.tick_params(axis="both", labelsize=fontsizeValues)
        ax.legend(fontsize=fontsizeTitle)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(
            save_path,
            bbox_inches="tight",
            transparent=transparent,
            dpi=dpi,
            facecolor='none'
        )
    plt.show()

def plot_two_lines_twin(data, col1, col2, title, xlabel, ylabel1, ylabel2, **kwargs):

    fontsizeTitle  = 24
    fontsizeLabel  = 22
    fontsizeValues = 20

    figsize        = kwargs.get('figsize', (20, 8))
    save_path      = kwargs.get('save_path', None)
    dpi            = kwargs.get('dpi', 300)
    dark_mode      = kwargs.get('dark_mode', True)
    bg_color       = kwargs.get('bg_color', '#3E3D3D')
    text_color     = kwargs.get('text_color', 'white')
    color1         = kwargs.get('color1', "#D07F0E")
    color2         = kwargs.get('color2', "#3F82D4")
    linewidth      = kwargs.get('linewidth', 2.5)
    transparent    = kwargs.get('transparent', True)
    fill_between = kwargs.get('fill_between', False)

    if not dark_mode:
        text_color = 'black'
        bg_color   = 'white'

    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()

    if dark_mode:
        fig.patch.set_facecolor(bg_color)
        ax1.set_facecolor(bg_color)
    
    if fill_between:
        ax2.fill_between(data.index, data[col2], color=color2, alpha=0.35)
        ax1.fill_between(data.index, data[col1], color=color1, alpha=0.55)
  

    ax1.plot(data.index, data[col1], color=color1, marker=".", alpha=0.8, linewidth=linewidth, label=col1)
    ax2.plot(data.index, data[col2], color=color2, marker=".", alpha=0.8, linewidth=linewidth, label=col2)

    ax1.set_xlim(data.index.min(), data.index.max())
    ax1.spines[["top"]].set_visible(False)
    ax1.set_title(title, fontsize=fontsizeTitle, pad=20, color=text_color)
    ax1.set_xlabel(xlabel, fontsize=fontsizeLabel, labelpad=20, color=text_color)
    ax1.set_ylabel(ylabel1, fontsize=fontsizeLabel, labelpad=20, color=text_color)
    ax2.set_ylabel(ylabel2, fontsize=fontsizeLabel, labelpad=20, color=text_color)
    ax1.tick_params(axis="both", labelsize=fontsizeValues, colors=text_color)
    ax2.tick_params(axis="y", labelsize=fontsizeValues, colors=color2)
    ax1.spines[["left", "bottom"]].set_edgecolor(text_color)
    ax2.spines[["top"]].set_visible(False)
    ax2.spines["right"].set_edgecolor(color2)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               fontsize=fontsizeTitle, facecolor=bg_color,
               labelcolor=text_color, edgecolor=text_color)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", transparent=transparent, dpi=dpi, facecolor='none')
    plt.show()

import plotly.express as px

def plot_geo_scatter(df, locations_col, values_col, **kwargs):
    
    save_path       = kwargs.get('save_path', None)
    projection      = kwargs.get('projection', 'natural earth')
    locationmode    = kwargs.get('locationmode', 'country names')
    min_value       = kwargs.get('min_value', None)
    color_scale     = kwargs.get('color_scale', 'Reds')
    q_low           = kwargs.get('q_low', 0.1)
    q_high          = kwargs.get('q_high', 0.9)
    title           = kwargs.get('title', None)

    range_color = [
        df[values_col].quantile(q_low),
        df[values_col].quantile(q_high)
    ]

    fig = px.scatter_geo(
        df,
        locations=locations_col,
        locationmode=locationmode,
        size=values_col,
        color=values_col,
        title=title,
        projection=projection,
        color_continuous_scale=color_scale,
        range_color=range_color,
    )

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=14),
        title=dict(font=dict(size=22), x=0.5),
        coloraxis_colorbar=dict(
            title=values_col.capitalize(),
            tickfont=dict(color='white'),
            title_font=dict(color='white'),
        ),
        geo=dict(
            bgcolor='rgba(0,0,0,0)',
            landcolor='#3E3D3D',
            oceancolor='#2B2B2B',
            showocean=True,
            countrycolor='#666666',
            coastlinecolor='#666666',
            framecolor='#666666',
        )
    )

    fig.show()

    if save_path is not None:
        fig.write_image(save_path, scale=3, width=1600, height=900)




def plot_numeric_columns(
    df: pd.DataFrame,
    x_col: str | None = None,
    n_cols: int = 3,
    title: str | None = "Numeric columns overview",
    xlabel: str | None = None,
    ylabel: str | None = None,
    theme: str = "light",
    transparent: bool = False,
    save_path: str | None = None,
    palette: str = "tab10",
    max_xticks: int | None = None,
) -> None:
    # Pick x values
    if x_col is not None:
        if x_col not in df.columns:
            raise ValueError(f"x_col '{x_col}' not found in dataframe.")
        x_values = df[x_col]
        numeric_df = df.drop(columns=[x_col]).select_dtypes(include="number")
        default_xlabel = x_col
    else:
        x_values = df.index
        numeric_df = df.select_dtypes(include="number")
        default_xlabel = "index"

    if numeric_df.empty:
        print("No numeric columns to plot.")
        return

    xlabel = xlabel if xlabel is not None else default_xlabel

    # Theme colors
    if theme == "dark":
        fg = "#E6E6E6"
        bg = "#1B1B1F"
        grid = "#3A3A40"
    else:
        fg = "#1B1B1F"
        bg = "#FFFFFF"
        grid = "#D9D9D9"

    n = numeric_df.shape[1]
    n_cols = min(n_cols, n)
    n_rows = math.ceil(n / n_cols)

    fig_w = 4.5 * n_cols
    fig_h = 3.0 * n_rows

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_w, fig_h),
        constrained_layout=True,
    )
    axes = axes.flatten() if n > 1 else [axes]

    # Backgrounds
    fig.patch.set_facecolor("none" if transparent else bg)
    cmap = plt.get_cmap(palette)

    for i, col in enumerate(numeric_df.columns):
        ax = axes[i]
        color = cmap(i % cmap.N)

        ax.plot(x_values, numeric_df[col], color=color, linewidth=1.6, alpha=0.95)
        ax.fill_between(x_values, numeric_df[col], alpha=0.10, color=color)

        ax.set_title(col, fontsize=11, fontweight="medium", loc="left", color=fg)
        ax.set_xlabel(xlabel, fontsize=9, color=fg)
        
        if max_xticks is not None:
            ax.xaxis.set_major_locator(plt.MaxNLocator(max_xticks))

        if ylabel is not None:
            ax.set_ylabel(ylabel, fontsize=9, color=fg)

        ax.tick_params(axis="both", labelsize=9, colors=fg)
        ax.grid(True, color=grid, linewidth=0.6, alpha=0.7)
        ax.set_axisbelow(True)
        ax.set_facecolor("none" if transparent else bg)

        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(grid)

        # ax.xaxis.set_major_locator(plt.MaxNLocator(6))
        # if pd.api.types.is_datetime64_any_dtype(x_values):
        #     ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

        ax.margins(x=0)

    # Hide unused axes
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=14, fontweight="medium", color=fg)

    if save_path:
        fig.savefig(
            save_path,
            dpi=150,
            bbox_inches="tight",
            transparent=transparent,
            facecolor="none" if transparent else bg,
        )

    plt.show()


def plot_corr(df, transparent=False, save_path=None, dark=False):
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    corr = df[numeric_cols].corr()
    
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    
    if dark:
        text_color = "#cdd6f4"
        bg_color = "#1e1e2e"
        grid_color = "#45475a"
    else:
        text_color = "#1e1e2e"
        bg_color = "#ffffff"
        grid_color = "#cccccc"
    
    fig, ax = plt.subplots(figsize=(11, 9))
    
    if not transparent:
        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)
    
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=1.5,
        linecolor=bg_color if not transparent else "#1e1e2e",
        cbar_kws={
            "shrink": 0.7,
            "label": "Correlation",
            "ticks": [-1, -0.5, 0, 0.5, 1],
        },
        annot_kws={"size": 17, "weight": "bold"},
        ax=ax,
    )
    
    ax.set_title(
        "Feature Correlation Matrix",
        fontsize=26,
        fontweight="bold",
        color=text_color,
        pad=20,
    )
    
    ax.tick_params(colors=text_color, labelsize=17)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0)
    
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.set_tick_params(color=text_color, labelsize=15)
    cbar.outline.set_edgecolor(grid_color)
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=text_color)
    cbar.set_label("Correlation", color=text_color, fontsize=17, weight="bold")
    
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(
            save_path,
            transparent=transparent,
            dpi=200,
            bbox_inches="tight",
            facecolor=fig.get_facecolor() if not transparent else "none",
        )
    
    plt.show()

