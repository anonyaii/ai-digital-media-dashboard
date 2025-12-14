import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio

# Load data
CSV_PATH = "impact_of_ai_on_digital_media.csv"
df = pd.read_csv(CSV_PATH)


# Find first column whose name contains all keywords
def find_col(df, keywords, required=True):
    keywords = [k.lower() for k in keywords]
    for col in df.columns:
        clean = col.lower().replace(" ", "_")
        if all(k in clean for k in keywords):
            return col
    if required:
        raise ValueError(
            f"Could not find column with keywords {keywords} in {list(df.columns)}"
        )
    return None


# Detect columns
YEAR_COL = find_col(df, ["year"])
COUNTRY_COL = find_col(df, ["country"])
INDUSTRY_COL = find_col(df, ["industry"])
ADOPTION_COL = find_col(df, ["adoption"])
JOBLOSS_COL = find_col(df, ["job", "loss"])
REVENUE_COL = find_col(df, ["revenue"])
TRUST_COL = find_col(df, ["trust"])
VOLUME_COL = find_col(df, ["content", "volume"], required=False)
TOOLS_COL = find_col(df, ["tool"], required=False)

# Clean types
df[YEAR_COL] = pd.to_numeric(df[YEAR_COL], errors="coerce").astype("Int64")
for col in [ADOPTION_COL, JOBLOSS_COL, REVENUE_COL, TRUST_COL]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
if VOLUME_COL is not None:
    df[VOLUME_COL] = pd.to_numeric(df[VOLUME_COL], errors="coerce")

df = df.dropna(
    subset=[
        YEAR_COL,
        COUNTRY_COL,
        INDUSTRY_COL,
        ADOPTION_COL,
        JOBLOSS_COL,
        REVENUE_COL,
        TRUST_COL,
    ]
)

# Aggregations
global_year = (
    df.groupby(YEAR_COL)
    .agg(
        adoption_value=(ADOPTION_COL, "mean"),
        jobloss_value=(JOBLOSS_COL, "mean"),
        revenue_value=(REVENUE_COL, "mean"),
    )
    .reset_index()
    .sort_values(YEAR_COL)
)

industry_stats = (
    df.groupby(INDUSTRY_COL)
    .agg(
        adoption_value=(ADOPTION_COL, "mean"),
        revenue_value=(REVENUE_COL, "mean"),
        jobloss_value=(JOBLOSS_COL, "mean"),
        trust_value=(TRUST_COL, "mean"),
    )
    .reset_index()
    .dropna(subset=["adoption_value"])
)

country_year = (
    df.groupby([YEAR_COL, COUNTRY_COL])[ADOPTION_COL]
    .mean()
    .reset_index()
    .sort_values([COUNTRY_COL, YEAR_COL])
)

all_countries = sorted(df[COUNTRY_COL].unique().tolist())
N_COUNTRIES = len(all_countries)

country_year_all = country_year[country_year[COUNTRY_COL].isin(all_countries)]
min_y = country_year_all[ADOPTION_COL].min()
max_y = country_year_all[ADOPTION_COL].max()
y_pad = (max_y - min_y) * 0.05 if max_y > min_y else 1
shared_y_range = [min_y - y_pad, max_y + y_pad]

heat_df = (
    df.groupby([INDUSTRY_COL, COUNTRY_COL])[ADOPTION_COL]
    .mean()
    .reset_index()
)
heat_pivot = heat_df.pivot(index=INDUSTRY_COL, columns=COUNTRY_COL, values=ADOPTION_COL)
heat_pivot = heat_pivot.reindex(columns=all_countries).fillna(0.0)
heat_z = heat_pivot.values
heat_x = list(heat_pivot.columns)
heat_y = list(heat_pivot.index)

# Volume by industry
volume_by_industry = None
if VOLUME_COL is not None:
    volume_by_industry = (
        df.groupby(INDUSTRY_COL)[VOLUME_COL]
        .mean()
        .reset_index()
        .sort_values(VOLUME_COL, ascending=False)
    )

# Top tools
tools_counts = None
if TOOLS_COL is not None:
    tools_expanded = (
        df[[TOOLS_COL]]
        .dropna()
        .assign(tool=lambda d: d[TOOLS_COL].astype(str).str.split(","))
        .explode("tool")
    )
    tools_expanded["tool"] = tools_expanded["tool"].str.strip()
    tools_counts = (
        tools_expanded["tool"]
        .value_counts()
        .reset_index(name="count")
        .head(10)
    )

# Figure structure
N_COLS = max(N_COUNTRIES + 2, 6)
g_span = max(2, int(0.3 * N_COLS))
s_span = max(2, int(0.35 * N_COLS))
if g_span + s_span >= N_COLS - 1:
    s_span = max(2, N_COLS - g_span - 2)
h_span = N_COLS - g_span - s_span
if h_span < 2:
    h_span = 2
    s_span = max(2, N_COLS - g_span - h_span)

specs_row1 = [None] * N_COLS
specs_row1[0] = {"type": "xy", "colspan": g_span}
specs_row1[g_span] = {"type": "xy", "colspan": s_span}
specs_row1[g_span + s_span] = {"type": "heatmap", "colspan": h_span}

specs_row2 = []
for i in range(N_COLS):
    if i < N_COUNTRIES:
        specs_row2.append({"type": "xy"})
    elif i == N_COUNTRIES:
        specs_row2.append({"type": "xy"})
    elif i == N_COUNTRIES + 1:
        specs_row2.append({"type": "xy"})
    else:
        specs_row2.append(None)

subplot_titles = (
    "Global Trends: AI Adoption, Job Loss, Revenue (2020–2025)",
    "Industry Profiles: Adoption vs Revenue",
    "AI Adoption by Industry & Country"
) + tuple(all_countries) + (
    "AI-Generated Content Volume by Industry",
    "Top AI Tools Used",
)

fig = make_subplots(
    rows=2,
    cols=N_COLS,
    column_widths=[1.0 / N_COLS] * N_COLS,
    row_heights=[0.55, 0.45],
    specs=[specs_row1, specs_row2],
    subplot_titles=subplot_titles,
    vertical_spacing=0.12,
    horizontal_spacing=0.05,
)

# Global trends lines
year_vals = global_year[YEAR_COL]

fig.add_trace(
    go.Scatter(
        x=year_vals,
        y=global_year["adoption_value"],
        mode="lines+markers",
        name="AI Adoption (%)",
        line=dict(color="#1f77b4", width=2),
        hovertemplate="Year %{x}<br>AI Adoption: %{y:.1f}%<extra></extra>",
        showlegend=True,
    ),
    row=1,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=year_vals,
        y=global_year["jobloss_value"],
        mode="lines+markers",
        name="Job Loss (%)",
        line=dict(color="#d62728", width=2),
        hovertemplate="Year %{x}<br>Job Loss: %{y:.1f}%<extra></extra>",
        showlegend=True,
    ),
    row=1,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=year_vals,
        y=global_year["revenue_value"],
        mode="lines+markers",
        name="Revenue Increase (%)",
        line=dict(color="#2ca02c", width=2),
        hovertemplate="Year %{x}<br>Revenue Increase: %{y:.1f}%<extra></extra>",
        showlegend=True,
    ),
    row=1,
    col=1,
)

fig.update_xaxes(title_text="Year", row=1, col=1, rangeslider=dict(visible=False))
fig.update_yaxes(title_text="Percentage (%)", row=1, col=1, title_standoff=8)

# Industry scatter
scatter_col = g_span + 1
trust = industry_stats["trust_value"].fillna(industry_stats["trust_value"].median())
if len(trust) > 0:
    size_scaled = 8 + 22 * (trust - trust.min()) / (trust.max() - trust.min() + 1e-9)
else:
    size_scaled = 10

fig.add_trace(
    go.Scatter(
        x=industry_stats["adoption_value"],
        y=industry_stats["revenue_value"],
        mode="markers+text",
        marker=dict(
            size=size_scaled,
            color=industry_stats["jobloss_value"],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(
                title="Job Loss (%)",
                tickfont=dict(size=8),
                len=0.55,
                y=0.77,
                x=0.66,
            ),
            opacity=0.8,
        ),
        text=industry_stats[INDUSTRY_COL],
        textposition="top center",
        hovertemplate=(
            "Industry: %{text}<br>"
            "AI Adoption: %{x:.1f}%<br>"
            "Revenue Increase: %{y:.1f}%<br>"
            "Job Loss (colour): %{marker.color:.1f}%<br>"
            "Trust (size): %{customdata[0]:.1f}%<extra></extra>"
        ),
        customdata=np.expand_dims(industry_stats["trust_value"].values, axis=1),
        showlegend=False,
    ),
    row=1,
    col=scatter_col,
)

fig.update_xaxes(title_text="AI Adoption (%)", row=1, col=scatter_col)
fig.update_yaxes(title_text="Revenue Increase (%)", row=1, col=scatter_col, title_standoff=8)

# Heatmap
heatmap_col = g_span + s_span + 1
heatmap = go.Heatmap(
    z=heat_z,
    x=heat_x,
    y=heat_y,
    colorscale="YlOrRd",
    colorbar=dict(
        title="AI Adoption (%)",
        tickfont=dict(size=9),
        len=0.55,
        y=0.77,
        x=1.02,
    ),
    hovertemplate="Industry: %{y}<br>Country: %{x}<br>AI Adoption: %{z:.1f}%<extra></extra>",
)
fig.add_trace(heatmap, row=1, col=heatmap_col)

fig.update_xaxes(title_text="Country", row=1, col=heatmap_col, title_standoff=0)
fig.update_yaxes(title_text="", row=1, col=heatmap_col)

# Custom y label for heatmap
fig.add_annotation(
    x=0.75,
    y=0.77,
    xref="paper",
    yref="paper",
    text="Industry",
    showarrow=False,
    textangle=-90,
    font=dict(size=20),
)

# Country mini-charts
for idx, country_name in enumerate(all_countries):
    r, c = 2, idx + 1
    sub = country_year_all[country_year_all[COUNTRY_COL] == country_name]
    fig.add_trace(
        go.Scatter(
            x=sub[YEAR_COL],
            y=sub[ADOPTION_COL],
            mode="lines+markers",
            line=dict(color="#6baed6", width=2),
            hovertemplate="Year %{x}<br>AI Adoption: %{y:.1f}%<extra></extra>",
            showlegend=False,
        ),
        row=r,
        col=c,
    )
    fig.update_xaxes(title_text="Year", row=r, col=c)
    if c == 1:
        fig.update_yaxes(
            title_text="AI Adoption (%)",
            row=r,
            col=c,
            range=shared_y_range,
            title_standoff=6,
        )
    else:
        fig.update_yaxes(
            showticklabels=False,
            row=r,
            col=c,
            range=shared_y_range,
        )

# Volume bar
vol_col = N_COUNTRIES + 1
if volume_by_industry is not None and not volume_by_industry.empty:
    fig.add_trace(
        go.Bar(
            x=volume_by_industry[INDUSTRY_COL],
            y=volume_by_industry[VOLUME_COL],
            hovertemplate="Industry: %{x}<br>AI Content Volume: %{y:.1f} TB/year<extra></extra>",
            showlegend=False,
        ),
        row=2,
        col=vol_col,
    )
    fig.update_xaxes(title_text="Industry", row=2, col=vol_col, tickangle=-45)
    fig.update_yaxes(title_text="Content Volume (TB/year)", row=2, col=vol_col, title_standoff=8)

# Tools bar
tools_col = N_COUNTRIES + 2
if tools_counts is not None and not tools_counts.empty:
    fig.add_trace(
        go.Bar(
            x=tools_counts["tool"],
            y=tools_counts["count"],
            hovertemplate="Tool: %{x}<br>Count: %{y}<extra></extra>",
            showlegend=False,
        ),
        row=2,
        col=tools_col,
    )
    fig.update_xaxes(title_text="AI Tool", row=2, col=tools_col, tickangle=-45)
    fig.update_yaxes(title_text="Occurrences", row=2, col=tools_col, title_standoff=8)

# Domains for top row
fig.update_xaxes(domain=[0.00, 0.30], row=1, col=1)
fig.update_xaxes(domain=[0.35, 0.65], row=1, col=scatter_col)
fig.update_xaxes(domain=[0.80, 1.00], row=1, col=heatmap_col)

# Domains for bottom row
gap_country = 0.01
left_end = 0.70
n_c = len(all_countries)
if n_c > 0:
    total_gaps = gap_country * (n_c - 1)
    width_c = (left_end - total_gaps) / n_c
    for idx in range(n_c):
        start = idx * (width_c + gap_country)
        end = start + width_c
        fig.update_xaxes(domain=[start, end], row=2, col=idx + 1)

vol_col = N_COUNTRIES + 1
tools_col = N_COUNTRIES + 2
if volume_by_industry is not None and not volume_by_industry.empty:
    fig.update_xaxes(domain=[0.74, 0.86], row=2, col=vol_col)
if tools_counts is not None and not tools_counts.empty:
    fig.update_xaxes(domain=[0.91, 1.00], row=2, col=tools_col)

# Align subplot titles
annotations = list(fig.layout.annotations)
annotations[0].x = 0.15
annotations[1].x = 0.50
annotations[2].x = 0.90

if n_c > 0:
    for idx in range(n_c):
        start = idx * (width_c + gap_country)
        end = start + width_c
        centre = (start + end) / 2.0
        ann_idx = 3 + idx
        annotations[ann_idx].x = centre
    vol_ann_idx = 3 + n_c
    tools_ann_idx = 4 + n_c
    annotations[vol_ann_idx].x = 0.80
    annotations[tools_ann_idx].x = 0.955

for ann in annotations:
    ann.font.size = 9

fig.layout.annotations = tuple(annotations)

# Remove axis titles in row 2 and add shared ones as annotations
for c in range(1, N_COUNTRIES + 1):
    fig.update_xaxes(row=2, col=c, title_text="")
fig.update_xaxes(row=2, col=vol_col, title_text="")
fig.update_xaxes(row=2, col=tools_col, title_text="")

bottom_title_y = -0.12
if fig.layout.margin is None:
    fig.update_layout(margin=dict(b=90))
else:
    current_b = fig.layout.margin.b if fig.layout.margin.b is not None else 0
    if current_b < 90:
        fig.update_layout(margin=dict(b=90))

if n_c > 0:
    for idx in range(n_c):
        start = idx * (width_c + gap_country)
        end = start + width_c
        centre = (start + end) / 2.0
        fig.add_annotation(
            x=centre,
            y=bottom_title_y,
            xref="paper",
            yref="paper",
            text="Year",
            showarrow=False,
            font=dict(size=10),
        )

fig.add_annotation(
    x=(0.74 + 0.86) / 2.0,
    y=bottom_title_y,
    xref="paper",
    yref="paper",
    text="Industry",
    showarrow=False,
    font=dict(size=10),
)

fig.add_annotation(
    x=(0.91 + 1.00) / 2.0,
    y=bottom_title_y,
    xref="paper",
    yref="paper",
    text="AI Tool",
    showarrow=False,
    font=dict(size=10),
)

# Layout and export
fig.update_layout(
    hovermode="x unified",
    title=(
        "Impact of AI on Digital Media (2020–2025)<br>"
        "Global Patterns, Industry Profiles, Country Trends and AI Tooling"
    ),
    autosize=True,
    showlegend=True,
    legend=dict(
        orientation="v",
        x=0.30,
        y=1.00,
        xanchor="right",
        yanchor="top",
        font=dict(size=7),
        bgcolor="rgba(255,255,255,0.7)",
        bordercolor="rgba(0,0,0,0.2)",
        borderwidth=0.2,
    ),
    margin=dict(l=0, r=0, t=80, b=0),
    paper_bgcolor="#f2f2f2",
    font=dict(size=10),
)

output_file = "ai_digital_media_dashboard.html"
pio.write_html(
    fig,
    file=output_file,
    auto_open=False,
    include_plotlyjs="cdn",
    full_html=True,
    default_width="100%",
    default_height="100%",
)
