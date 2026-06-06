"""
charts.py  –  All 10 required chart types + bonus pair-plot
Course   : Exploratory Data Analysis  |  Instructor: Ali Hassan Sherazi
Dataset  : aidsinfo.unaids.org.csv   |  UNAIDS HIV/AIDS Global Estimates

WHY each chart is chosen for the HIV/AIDS dataset is documented inside each
function. Matplotlib + Seaborn are used exclusively (as per rubric).
"""

import matplotlib
matplotlib.use("Agg")           # non-interactive backend required for Streamlit
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# ── Colour palette: clinical-modern crimson/teal tones ───────────────────────
PALETTE  = ["#C0392B", "#2980B9", "#16A085", "#8E44AD", "#E67E22",
            "#27AE60", "#2C3E50", "#D35400", "#1ABC9C", "#E74C3C"]
BG       = "#0D1117"    # deep navy background
CARD_BG  = "#161B22"
TEXT     = "#E6EDF3"
ACCENT   = "#C0392B"    # HIV-red accent


def _style(fig, ax_list=None):
    """Apply consistent dark-clinical theme to any figure."""
    fig.patch.set_facecolor(BG)
    if ax_list is None:
        ax_list = fig.get_axes()
    for ax in ax_list:
        ax.set_facecolor(CARD_BG)
        ax.tick_params(colors=TEXT, labelsize=9)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(ACCENT)
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363D")


def _sample(df: pd.DataFrame, n: int = 50_000) -> pd.DataFrame:
    """
    Return a random sample of up to n rows.
    WHY: The full dataset has 2.5M rows; Matplotlib/Seaborn scatter/violin
         plots become extremely slow. Sampling preserves distribution shape
         while keeping render time < 2 seconds.
    """
    return df.sample(n=min(n, len(df)), random_state=42) if len(df) > n else df


# ── 1. PIE CHART ─────────────────────────────────────────────────────────────
def pie_chart(df: pd.DataFrame) -> plt.Figure:
    """
    WHY PIE: Shows proportional split of records by Unit type (Rate/Number/Percent).
    Categorical composition is immediately readable in pie form.
    COLUMNS USED: Unit (categorical)
    """
    counts = df["Unit"].value_counts().head(6)
    fig, ax = plt.subplots(figsize=(6, 5))
    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=counts.index,
        autopct="%1.1f%%",
        colors=PALETTE[:len(counts)],
        startangle=140,
        wedgeprops=dict(edgecolor=BG, linewidth=1.5),
    )
    for t in texts + autotexts:
        t.set_color(TEXT)
        t.set_fontsize(9)
    ax.set_title("Distribution of HIV Indicator Units", fontsize=13, pad=12)
    _style(fig, [ax])
    fig.tight_layout()
    return fig


# ── 2. HISTOGRAM ─────────────────────────────────────────────────────────────
def histogram(df: pd.DataFrame) -> plt.Figure:
    """
    WHY HISTOGRAM: Data_Value is the core measurement across all indicators.
    Histogram reveals whether values are normally distributed, skewed, or
    concentrated at specific ranges (e.g. prevalence rates near 0).
    COLUMNS USED: Data_Value (continuous numerical)
    """
    # Cap at 99th percentile to avoid extreme outliers dominating the axis
    cap = df["Data_Value"].quantile(0.99)
    sub = df[df["Data_Value"] <= cap]["Data_Value"]
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(sub, bins=40, kde=True,
                 color=PALETTE[0], edgecolor=BG, ax=ax)
    if ax.lines:
        ax.lines[0].set_color(PALETTE[1])
    ax.set_xlabel("Data Value (capped at 99th percentile)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of HIV/AIDS Data Values", fontsize=13)
    _style(fig, [ax])
    fig.tight_layout()
    return fig


# ── 3. LINE CHART ─────────────────────────────────────────────────────────────
def line_chart(df: pd.DataFrame) -> plt.Figure:
    """
    WHY LINE: Tracks the average data value per year across all indicators.
    A time-series line reveals global HIV/AIDS trends from 1990 to 2024.
    COLUMNS USED: Time_Period (x), Data_Value (y, averaged)
    """
    yearly = (
        df.groupby("Time_Period")["Data_Value"]
        .mean()
        .reset_index()
        .sort_values("Time_Period")
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(yearly["Time_Period"], yearly["Data_Value"],
            color=ACCENT, linewidth=2.5, marker="o", markersize=4,
            markerfacecolor=PALETTE[1])
    ax.fill_between(yearly["Time_Period"], yearly["Data_Value"],
                    alpha=0.15, color=ACCENT)
    ax.set_xlabel("Year")
    ax.set_ylabel("Average Data Value")
    ax.set_title("Global HIV/AIDS Indicator Trends Over Time", fontsize=13)
    _style(fig, [ax])
    fig.tight_layout()
    return fig


# ── 4. BAR CHART ─────────────────────────────────────────────────────────────
def bar_chart(df: pd.DataFrame) -> plt.Figure:
    """
    WHY BAR: Compares average data values across top 10 reporting areas.
    Horizontal bar chart is optimal for ranking categories with long names.
    COLUMNS USED: Area (categorical), Data_Value (aggregated mean)
    """
    # Exclude aggregate regions like 'All countries' for country-level analysis
    sub = df[df["Area_Level"] >= 2]
    top = (
        sub.groupby("Area")["Data_Value"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(top["Area"], top["Data_Value"],
                   color=PALETTE[:10], edgecolor=BG)
    ax.set_xlabel("Average Data Value")
    ax.set_title("Top 10 Reporting Areas by Average Data Value", fontsize=13)
    ax.invert_yaxis()
    for bar, val in zip(bars, top["Data_Value"]):
        ax.text(val + val * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:,.1f}", va="center", color=TEXT, fontsize=8)
    _style(fig, [ax])
    fig.tight_layout()
    return fig


# ── 5. SCATTER PLOT ───────────────────────────────────────────────────────────
def scatter_plot(df: pd.DataFrame) -> plt.Figure:
    """
    WHY SCATTER: Plots Data_Value over Time_Period colored by Unit.
    Reveals how different unit types (Rate, Number, Percent) evolve over time
    and whether their magnitudes differ systematically.
    COLUMNS USED: Time_Period (x), Data_Value (y), Unit (color)
    """
    sub = _sample(df, 30_000)
    cap = sub["Data_Value"].quantile(0.95)
    sub = sub[sub["Data_Value"] <= cap]
    units = sub["Unit"].unique()[:3]
    color_map = dict(zip(units, PALETTE))
    fig, ax = plt.subplots(figsize=(7, 5))
    for unit in units:
        s = sub[sub["Unit"] == unit]
        ax.scatter(s["Time_Period"], s["Data_Value"],
                   label=unit, color=color_map[unit],
                   alpha=0.4, s=15, edgecolors="none")
    ax.set_xlabel("Year")
    ax.set_ylabel("Data Value")
    ax.set_title("HIV/AIDS Data Values Over Time by Unit Type", fontsize=13)
    ax.legend(fontsize=8, facecolor=CARD_BG, labelcolor=TEXT, edgecolor="#30363D")
    _style(fig, [ax])
    fig.tight_layout()
    return fig


# ── 6. BOX PLOT ───────────────────────────────────────────────────────────────
def box_plot(df: pd.DataFrame) -> plt.Figure:
    """
    WHY BOX PLOT: Compares spread, median, and outliers of Data_Value across
    the three unit types (Rate, Number, Percent). Reveals how much variability
    exists within each measurement category.
    COLUMNS USED: Unit (categorical x), Data_Value (numerical y)
    """
    cap = df["Data_Value"].quantile(0.95)
    sub = df[df["Data_Value"] <= cap].copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=sub, x="Unit", y="Data_Value", hue="Unit",
                palette=PALETTE[:3], ax=ax, legend=False,
                flierprops=dict(marker="o", color=ACCENT, markersize=3))
    ax.set_xlabel("Unit Type")
    ax.set_ylabel("Data Value")
    ax.set_title("Data Value Spread by Unit Type", fontsize=13)
    _style(fig, [ax])
    fig.tight_layout()
    return fig


# ── 7. HEATMAP ────────────────────────────────────────────────────────────────
def heatmap(df: pd.DataFrame) -> plt.Figure:
    """
    WHY HEATMAP: Correlation matrix between Time_Period, Area_Level, and
    Data_Value. Reveals whether time or geographic level predicts the magnitude
    of HIV/AIDS statistics.
    COLUMNS USED: Time_Period, Area_Level, Data_Value (all numerical)
    """
    num_cols = ["Time_Period", "Area_Level", "Data_Value"]
    # Pivot: average Data_Value per Indicator × Time_Period (top 8 indicators)
    top_indicators = df["Indicator"].value_counts().head(8).index.tolist()
    pivot_df = (
        df[df["Indicator"].isin(top_indicators)]
        .groupby(["Indicator", "Time_Period"])["Data_Value"]
        .mean()
        .unstack(level=0)
        .dropna(how="all")
    )
    # Correlation between indicators over time
    corr = pivot_df.corr()
    # Shorten labels for display
    corr.columns = [c[:25] for c in corr.columns]
    corr.index   = [i[:25] for i in corr.index]
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn",
                linewidths=0.5, linecolor=BG,
                annot_kws={"size": 6, "color": "#0D1117"},
                ax=ax)
    ax.set_title("Correlation Between HIV/AIDS Indicators Over Time", fontsize=12)
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.tick_params(axis="y", rotation=0,  labelsize=7)
    _style(fig, [ax])
    fig.tight_layout()
    return fig


# ── 8. AREA CHART ─────────────────────────────────────────────────────────────
def area_chart(df: pd.DataFrame) -> plt.Figure:
    """
    WHY AREA: Cumulative number of HIV/AIDS records reported per year.
    The filled area emphasises growth in global surveillance and reporting
    capacity over time.
    COLUMNS USED: Time_Period (time), count of records (aggregated)
    """
    yearly = (
        df.groupby("Time_Period").size()
        .reset_index(name="Count")
        .sort_values("Time_Period")
    )
    yearly["Cumulative"] = yearly["Count"].cumsum()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.fill_between(yearly["Time_Period"], yearly["Cumulative"],
                    color=PALETTE[0], alpha=0.7)
    ax.plot(yearly["Time_Period"], yearly["Cumulative"],
            color=PALETTE[1], linewidth=2)
    ax.set_xlabel("Year")
    ax.set_ylabel("Cumulative Records")
    ax.set_title("Cumulative HIV/AIDS Data Records Over Time", fontsize=13)
    _style(fig, [ax])
    fig.tight_layout()
    return fig


# ── 9. COUNT PLOT ─────────────────────────────────────────────────────────────
def count_plot(df: pd.DataFrame) -> plt.Figure:
    """
    WHY COUNT PLOT: Shows the frequency of records per data Source.
    Categorical count plot reveals which organisations contribute most data,
    indicating data coverage bias.
    COLUMNS USED: Source (categorical)
    """
    # Shorten source names for display
    sub = df.copy()
    sub["Source_Short"] = sub["Source"].str[:30]
    order = sub["Source_Short"].value_counts().index
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(data=sub, y="Source_Short", hue="Source_Short",
                  order=order, legend=False,
                  palette=PALETTE[:len(order)], ax=ax,
                  edgecolor=BG)
    ax.set_xlabel("Count")
    ax.set_ylabel("Data Source")
    ax.set_title("Source Contribution Analysis", fontsize=13)
    _style(fig, [ax])
    fig.tight_layout()
    return fig


# ── 10. VIOLIN PLOT ───────────────────────────────────────────────────────────
def violin_plot(df: pd.DataFrame) -> plt.Figure:
    """
    WHY VIOLIN: Shows the full probability density of Data_Value across
    Area Levels (1=Global, 2=Regional, 3=National). Violin reveals
    bimodality and spread that a simple box plot cannot show.
    COLUMNS USED: Area_Level (categorical x), Data_Value (numerical y)
    """
    cap = df["Data_Value"].quantile(0.90)
    sub = _sample(df[df["Data_Value"] <= cap], 20_000)
    sub["Area_Level_Label"] = sub["Area_Level"].map(
        {1: "Global", 2: "Regional", 3: "National"}
    ).fillna(sub["Area_Level"].astype(str))
    order = ["Global", "Regional", "National"]
    order = [o for o in order if o in sub["Area_Level_Label"].unique()]
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.violinplot(data=sub, x="Area_Level_Label", y="Data_Value",
                   hue="Area_Level_Label",
                   order=order, legend=False,
                   palette=PALETTE[:3], inner="quartile", ax=ax)
    ax.set_xlabel("Area Level")
    ax.set_ylabel("Data Value")
    ax.set_title("Data Value Distribution by Area Level", fontsize=13)
    _style(fig, [ax])
    fig.tight_layout()
    return fig


# ── BONUS: PAIR PLOT ──────────────────────────────────────────────────────────
def pair_plot(df: pd.DataFrame) -> plt.Figure:
    """
    WHY PAIR PLOT (BONUS): Multi-dimensional view of Time_Period, Data_Value,
    and Area_Level colored by Unit type. Gives a comprehensive overview of
    pairwise relationships in one image.
    COLUMNS USED: Time_Period, Data_Value, Area_Level + hue = Unit
    """
    cap = df["Data_Value"].quantile(0.95)
    sub = _sample(df[df["Data_Value"] <= cap], 5_000)
    cols = ["Time_Period", "Data_Value", "Area_Level", "Unit"]
    sub = sub[cols].dropna()
    units = sub["Unit"].value_counts().head(3).index
    sub   = sub[sub["Unit"].isin(units)]
    g = sns.pairplot(sub, hue="Unit",
                     palette=dict(zip(units, PALETTE)),
                     plot_kws=dict(alpha=0.4, s=10),
                     diag_kind="kde",
                     vars=["Time_Period", "Data_Value", "Area_Level"])
    g.figure.patch.set_facecolor(BG)
    g.figure.suptitle("Pair Plot: HIV/AIDS Key Variables by Unit Type",
                       y=1.01, color=ACCENT, fontsize=12)
    for ax in g.axes.flatten():
        if ax:
            ax.set_facecolor(CARD_BG)
            ax.tick_params(colors=TEXT)
            ax.xaxis.label.set_color(TEXT)
            ax.yaxis.label.set_color(TEXT)
    return g.figure
