

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings


def clean_dataframe(
        df: pd.DataFrame,
        drop_empty_rows: bool = True,
        drop_empty_cols: bool = True,
        drop_any_nan_rows: bool = False
    ) -> pd.DataFrame:
    """
    Replace blank-like strings with NaN, then drop
    fully-empty rows/columns. Optionally drop rows
    containing *any* NaN.
    """
    # Treat "", " ", "NA", "N/A" as missing
    df = df.replace(r'^\s*$', np.nan, regex=True)
    df = df.replace(["NA", "N/A", "na", "n/a"], np.nan)
    if drop_empty_cols:
        df = df.dropna(axis=1, how="all")
    if drop_empty_rows:
        df = df.dropna(axis=0, how="all")
    if drop_any_nan_rows:
        df = df.dropna(axis=0, how="any")
    return df.reset_index(drop=True)


# ---------- 1. Column-typing ----------
def classify_columns(df: pd.DataFrame) -> dict:
    meta = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            meta[col] = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(df[col]) or col.lower().endswith(("date", "time")):
            meta[col] = "date"
        else:
            unique_ratio = df[col].nunique() / len(df)
            meta[col] = "categorical" if unique_ratio < 0.2 else "text"
    return meta


# ---------- 2. Chart router ----------
ROUTER = {
    ("numeric", "numeric"): "scatter",
    ("categorical", "numeric"): "bar",
    ("date", "numeric"): "line",
    ("numeric", "categorical"): "bar",
}

def pick_chart(x_type, y_type):
    return ROUTER.get((x_type, y_type), "table")

def safe_router(df, x, y):
    meta = classify_columns(df)
    key = (meta.get(x), meta.get(y))
    if key not in ROUTER:
        raise ValueError(f"No chart for {key}")
    if df[[x, y]].dropna().empty:
        raise ValueError("Selected columns have no data after cleaning")
    return ROUTER[key]

def debug_slice(df, x, y):
    print("Rows:", len(df))
    print("Nulls in x:", df[x].isna().sum(), "Nulls in y:", df[y].isna().sum())
    print("Sample:", df[[x, y]].head())

# ---------- 3. Plotting helpers ----------
def make_plot(df, x, y=None, title=None, top_k=10, theme="whitegrid"):
    # Set modern theme and colors
    modern_colors = ['#6366F1', '#8B5CF6', '#EC4899', '#F59E0B', '#10B981']
    sns.set_theme(style="whitegrid", palette=modern_colors)
    import matplotlib.pyplot as plt
    plt.rcParams["axes.facecolor"] = "#FAFAFA"
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.edgecolor"] = "#E5E7EB"
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["grid.color"] = "#F3F4F6"
    plt.rcParams["grid.alpha"] = 0.3
    x_type = classify_columns(df).get(x, "text") if x else None
    y_type = classify_columns(df).get(y, "text") if y else None
    fig, ax = plt.subplots()
    try:
        if x and y:
            # Guard router and data
            chart = safe_router(df, x, y)
            data = df[[x, y]].dropna()
            if chart == "scatter":
                ax = sns.scatterplot(data=data, x=x, y=y, ax=ax)
            elif chart == "bar":
                order = data[x].value_counts().index[:top_k] if x_type == "categorical" else data[y].value_counts().index[:top_k]
                if x_type == "categorical":
                    ax = sns.barplot(data=data[data[x].isin(order)], x=x, y=y, order=order, ax=ax)
                else:
                    ax = sns.barplot(data=data[data[y].isin(order)], x=y, y=x, order=order, ax=ax)
            elif chart == "line":
                if pd.api.types.is_datetime64_any_dtype(data[x]):
                    data = data.set_index(x)
                    if len(data) > 100:
                        data = data.resample("M").mean(numeric_only=True)
                    ax = data[y].plot(ax=ax)
                    ax.set_xlabel(x)
                    ax.set_ylabel(y)
                else:
                    ax = sns.lineplot(data=data, x=x, y=y, ax=ax)
            else:
                raise ValueError(f"No chart for ({x_type}, {y_type})")
        elif x:
            # 1D plot (hist/bar)
            data = df[x].dropna()
            if x_type == "numeric":
                ax = sns.histplot(data, kde=True, ax=ax)
            elif x_type == "categorical":
                order = data.value_counts().index[:top_k]
                ax = sns.barplot(x=order, y=data.value_counts().loc[order], ax=ax)
            else:
                raise ValueError(f"No chart for ({x_type},)")
        else:
            raise ValueError("No columns provided for plotting")
        tidy_ax(ax, x, y, title)
        return fig
    except Exception as e:
        # Fallback: show table with error as title
        warnings.warn(f"Plotting failed: {e}")
        ax.axis("off")
        tbl = df[[c for c in [x, y] if c in df.columns]].head(top_k) if x or y else df.head(top_k)
        ax.table(cellText=tbl.values, colLabels=tbl.columns, loc="center")
        ax.set_title(str(e))
        plt.tight_layout()
        return fig

# ---------- 4. Global theme, labels, layout ----------
def tidy_ax(ax, x=None, y=None, title=None):
    if title:
        ax.set_title(title, fontsize=16, fontweight='bold', color='#1F2937', pad=20)
    if x:
        ax.set_xlabel(x, fontsize=12, color='#4B5563')
    if y:
        ax.set_ylabel(y, fontsize=12, color='#4B5563')
    import matplotlib.pyplot as plt
    plt.tight_layout()
    return ax
