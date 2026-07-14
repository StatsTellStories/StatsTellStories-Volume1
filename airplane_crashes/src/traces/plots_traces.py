import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import matplotlib.dates as mdates

def plot_top_icao(df, top_icaos, col, xCol, n=8):
    icaos = top_icaos.head(n).index.tolist()
    nan_counts = top_icaos.head(n).values.tolist()

    rows = int(np.ceil(n / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(14, rows * 4))

    for ax, icao, nan_count in zip(axes.flat, icaos, nan_counts):
        sub = df[df["icao"] == icao].sort_values(xCol)
        ax.plot(sub[xCol], sub[col])
        #ax.plot(df[df["icao"] == icao][xCol], df[df["icao"] == icao][col])
        ax.set_title(f"{icao} ({nan_count} NaNs) — {col}", fontsize=14)
        ax.set_xlabel(xCol, fontsize=12)
        ax.set_ylabel(col, fontsize=12)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m %H:%M"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.tick_params(axis='x', rotation=45, labelsize=9)
        ax.tick_params(axis='y', labelsize=11)

    plt.tight_layout()
    plt.show()

def plot_flight_map(flight, color_cols=('altitude',), title='Flight Trace'):
    if isinstance(color_cols, str):
        color_cols = [color_cols]

    fig = px.scatter_map(
        flight,
        lat='lat',
        lon='lon',
        color=color_cols[0],
        color_continuous_scale='turbo',
        zoom=6,
        map_style='dark',
        hover_data={'timestamp': True, 'lat': False, 'lon': False, **{c: True for c in color_cols}}
    )
    fig.update_traces(marker=dict(size=6, opacity=0.8))
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        coloraxis_colorbar=dict(title=color_cols[0].capitalize()),
        margin=dict(l=0, r=0, t=60, b=0),
        #width=1200,
        height=700,
        paper_bgcolor="#f0f7cf",
    )

    if len(color_cols) > 1:
        buttons = []
        for col in color_cols:
            buttons.append(dict(
                label=col.capitalize(),
                method='update',
                args=[
                    {'marker.color': [flight[col]]},
                    {'coloraxis.colorbar.title.text': col.capitalize()},
                ],
            ))
        fig.update_layout(
            updatemenus=[dict(
                buttons=buttons,
                direction='down',
                x=0.01, y=1.08,
                xanchor='left', yanchor='top',
                bgcolor="#ffffff",
                bordercolor="#6b6b6b",
                font=dict(color='#000000', size=13),
                active=0,
            )]

        )

    fig.show()


def plot_density(df, lat_col='lat', lon_col='lon', z_col='point_count', 
                 radius=5, center=dict(lat=20, lon=0), zoom=1,
                 map_style='carto-positron', title='Point Density',
                 save_path=None, transparent=False):
    fig = px.density_map(
        df,
        lat=lat_col, lon=lon_col, z=z_col,
        radius=radius,
        center=center,
        zoom=zoom,
        map_style=map_style,
        title=title
    )
    fig.update_layout(height=700)
    
    if transparent:
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)')
    
    if save_path is not None:
        fig.write_image(save_path)
    
    fig.show()



def plot_top_flights(df, flight_ids, col, xCol, n=10):
    rows = int(np.ceil(n / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(14, rows * 4))

    for ax, fid in zip(axes.flat, flight_ids):
        sub = df.filter(pl.col("flight_id") == fid).sort(xCol)        
        timestamps = sub[xCol].to_numpy()
        values = sub[col].to_numpy().astype(float)
        
        if len(timestamps) > 1:
            diffs = np.diff(timestamps.astype("int64")) / 1e6
            max_gap_idx = np.argmax(diffs)
            max_gap_min = diffs[max_gap_idx] / 60
            ax.axvspan(timestamps[max_gap_idx], timestamps[max_gap_idx + 1], alpha=0.2, color='red', label=f"max gap: {max_gap_min:.1f}min")
            ax.axvline(x=timestamps[max_gap_idx], color='red', alpha=0.7, linewidth=1.5)
            ax.axvline(x=timestamps[max_gap_idx + 1], color='red', alpha=0.7, linewidth=1.5)
            ax.legend(fontsize=9)
        
        ax.plot(timestamps, values)
        ax.set_title(f"{fid} — {col}", fontsize=14)
        ax.set_xlabel(xCol, fontsize=12)
        ax.set_ylabel(col, fontsize=12)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m %H:%M"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.tick_params(axis='x', rotation=45, labelsize=9)
        ax.tick_params(axis='y', labelsize=11)

    plt.tight_layout()
    plt.show()



def plot_top_flights_pd(df, flight_ids, col, xCol, n=10):
    rows = int(np.ceil(n / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(14, rows * 4))

    for ax, fid in zip(axes.flat, flight_ids):
        sub = df[df['flight_id'] == fid].sort_values(xCol)
        timestamps = sub[xCol].to_numpy()
        values = sub[col].to_numpy().astype(float)
        
        if len(timestamps) > 1:
            diffs = np.diff(timestamps) / np.timedelta64(1, 's')
            max_gap_idx = np.argmax(diffs)
            max_gap_min = diffs[max_gap_idx] / 60
            ax.axvspan(timestamps[max_gap_idx], timestamps[max_gap_idx + 1], alpha=0.2, color='red', label=f"max gap: {max_gap_min:.1f}min")
            ax.axvline(x=timestamps[max_gap_idx], color='red', alpha=0.7, linewidth=1.5)
            ax.axvline(x=timestamps[max_gap_idx + 1], color='red', alpha=0.7, linewidth=1.5)
            ax.legend(fontsize=9)
        
        ax.plot(timestamps, values)
        ax.set_title(f"{fid} — {col}", fontsize=14)
        ax.set_xlabel(xCol, fontsize=12)
        ax.set_ylabel(col, fontsize=12)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m %H:%M"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.tick_params(axis='x', rotation=45, labelsize=9)
        ax.tick_params(axis='y', labelsize=11)

    plt.tight_layout()
    plt.show()
