import pandas as pd
import math
import constants
import os
import natsort

def ceil_int(n):
    """
    transform to int and ceil

    Parameters
    ----------
    n : float, int

    Returns
    -------
    int
    """
    return int(math.ceil(n))


def convert_to_datetime(date):
    """
    Convert date string in format YYYYMMDD to datetime

    Parameters
    ----------
    date : str YYYYMMDD

    Returns
    -------
    datetime object
    """
    if len(date) == 8:
        return pd.to_datetime(date, format="%Y%m%d", errors="raise", utc=False)
    elif len(date) == 12:
        return pd.to_datetime(date, format="%Y%m%d%H%M", errors="raise", utc=False)
    else:
        return pd.to_datetime(
            date, errors="raise", infer_datetime_format=True, utc=False
        )


def parse_column_names(df):
    """
    Rename columns according to constants.PARSING_DICT

    Parameters
    ----------
    df : input dataframe to rename cols

    Returns
    -------
    df with columns renamed
    """
    for col in df.columns:
        for col_name, col_alternatives in constants.PARSING_DICT:
            if col in col_alternatives:
                df = df.rename(columns={col: col_name})
                continue

    return df


def add_labels(df, label_columns, shift_values, col_to_shift):
    """
    Adds labeled columns to the dataframe by shifting values of an existing column.
    """
    for label_col, shift_val in zip(label_columns, shift_values):
        df[label_col] = df[col_to_shift].shift(shift_val)
    return df


def read_data(
    path,
    pattern_to_skip=None,
    pattern_to_read=["csv"],
    return_separated=True,
    tz_localize=False,
    print_info=False,
):
    """
    Generic function to read files, generically csvs
    If the path is a file, it will be read, "datetime" will be set as index
    and converted to datetime, the dataframe will be sorted by index and it will be returned

    If the path is a directory, there is a optional pattern_to_skip with a list of patterns
    that will be searched in the filename, if they are found the file will be skipped

    Pattern to read is a list of patterns that need to be present on the file name for it to
    be read, defaults to ["csv"]

    Return_separated is a boolean, defaulted to true, only affects when the path is a folder
    if it is true, all the valid files will be returned as a list, if it's false the dataframes
    will be concatenated and sorted by the index, and a joined dataframe will be returned

    Parameters
    ----------
    path : str, path to evaluate
    pattern_to_skip: list, Optional: patterns that will make the file to not be read if they are found
    pattern_to_read: ['csv'], Optional: patterns that need to be found for the file to be read
    return_separated : True, Optional: wether to concatenate the dataframes before returning or not
    tz_localize : Whether to tz_localize to UTC or not

    Returns
    -------
    a single dataframe if it is a file or return_separated = False, otherise, a list of dataframes.
    """

    # Check if the path is a file
    if os.path.isfile(path):
        # If it is a file we don't need to apply the pattern
        if print_info:
            print(f"Reading from file {path}")
        df = pd.read_csv(path, comment="#")
        df.set_index("datetime", inplace=True)
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        if tz_localize:
            df.index = df.index.tz_localize("UTC")
        return df

    # Check if the path is a folder
    if os.path.isdir(path):
        dfs = []
        files = natsort.natsorted(os.listdir(path))
        for f in files:
            skip = False

            # If a pattern to skip is found, skip the file
            if pattern_to_skip is not None:
                for pat in pattern_to_skip:
                    if f.find(pat) >= 0:
                        skip = True
                        break

            # If a pattern to read is not found, skip the file
            if pattern_to_read is not None:
                for pat in pattern_to_read:
                    if f.find(pat) < 0:
                        skip = True

            if skip:
                continue

            fil = os.path.join(path, f)
            if print_info:
                print(f"Reading from file {fil}")
            df1 = pd.read_csv(fil)
            df1.set_index("datetime", inplace=True)
            df1.index = pd.to_datetime(df1.index)
            df1.sort_index(inplace=True)
            if tz_localize:
                df1.index = df1.index.tz_localize("UTC")
            dfs.append(df1)

        if return_separated:
            return dfs
        else:
            df = pd.concat(dfs)
            df.sort_index(inplace=True)
            return df

    print(f"Is neither a valid file or directory")

