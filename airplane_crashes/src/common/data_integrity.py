import re

def check_duplicates(df):
    dupes = df[df.duplicated(keep=False)]
    print(f"Total rows: {len(df)}")
    print(f"Duplicate rows: {len(dupes)}")
    print(f"Duplicate rate: {len(dupes)/len(df)*100:.3f}%")
    return dupes


def check_nan(df):
    total_rows = len(df)

    nan_summary = (
        df.isna()
        .sum()
        .to_frame(name="NaN_count")
        .assign(
            NaN_percent=lambda x: (x["NaN_count"] / total_rows) * 100
        )
    )

    nan_summary = nan_summary[nan_summary["NaN_count"] >= 0]

    if nan_summary.empty:
        print("✅ No NaN values found in the DataFrame.")
    else:
        nan_summary["NaN_percent"] = nan_summary["NaN_percent"].round(2).astype(str) + " %"
        print("⚠️ NaN summary:")
        print(nan_summary)


def show_high_nan_columns(df, threshold=0.3):
    nan_summary = (
        df.isna()
        .mean()
        .to_frame(name="NaN_percent")
    )

    high_nan = nan_summary[nan_summary["NaN_percent"] > threshold]

    if high_nan.empty:
        print(f"✅ No columns with more than {int(threshold*100)}% NaN values.")
    else:
        high_nan["NaN_percent"] = (high_nan["NaN_percent"] * 100).round(2).astype(str) + " %"
        print(f"⚠️ Columns with more than {int(threshold*100)}% NaN values:")
        print(high_nan)


def drop_high_nan_columns(df, threshold=0.3, protected_cols=None):
    """
    Drops columns with a NaN ratio above the given threshold,
    except for columns listed in protected_cols.

    Parameters
    ----------
    df : pandas.DataFrame
    threshold : float, default 0.3
        Maximum allowed fraction of NaN values (e.g. 0.3 = 30%)
    protected_cols : list or None
        Columns that should not be dropped, even if they exceed the threshold

    Returns
    -------
    pandas.DataFrame
        Cleaned DataFrame
    """
    if protected_cols is None:
        protected_cols = []

    nan_ratio = df.isna().mean()

    cols_to_drop = [
        col for col in nan_ratio.index
        if nan_ratio[col] > threshold and col not in protected_cols
    ]

    print(f"Dropping {len(cols_to_drop)} columns with more than {int(threshold*100)}% NaN values.")
    if cols_to_drop:
        print("Dropped columns:", cols_to_drop)

    return df.drop(columns=cols_to_drop)

def unique_strings_without_numbers(column):
    """
    Takes a pandas Series (column) and returns
    unique strings that do not contain any digits.
    """
    result = column.dropna() \
                   .astype(str) \
                   .unique()
    
    # Keep only strings without digits
    filtered = [x for x in result if not re.search(r'\d', x)]
    
    return filtered

def check_string_values(df, column):
    """
    Checks how many string values are present in a column
    and shows the most common ones.
    """
    col = df[column]

    string_mask = col.apply(lambda x: isinstance(x, str))

    num_strings = string_mask.sum()
    total = len(col)

    print(f"Column '{column}':")
    print(f"- String values: {num_strings} ({(num_strings / total) * 100:.2f}%)")

    if num_strings > 0:
        print("\nMost common string values:")
        print(col[string_mask].value_counts().head())
