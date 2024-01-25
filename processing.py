
"""
File: processing.py
Description: Functions that process data in order to perform certain tasks
Author: Nathan Parker
Date: 1 October 2023
"""

import pandas as pd
import math
def alter_series(df, attribute, function):
    """ Iterates over a series in a dataframe and performs the given function

    Args:
        df (pandas.DataFrame): dataframe that wants to be altered
        attribute (string): column name to be altered in the dataframe
        function (function): Transformation to be applied to each value
    """
    for i in range(len(df)):
        df.loc[i, attribute] = function(df.loc[i, attribute])

# lambda function that rounds the one's place down
round_down = lambda x: (math.floor(x / 10)) * 10


def round_decade(df, attribute):
    """ Rounds the year to a decade in a dataframe series

    Args:
        df (pandas.DataFrame): dataframe that wants to be altered
        attribute (string): column name to be altered in the dataframe
    """
    alter_series(df, attribute, round_down)

def check_zero(number):
    """ checks if a number is equal to zero

    Args:
        number (int): the number being checked

    Returns:
        zero (bool): whether or not the number is equal to 0
    """
    zero = (number == 0)
    return zero

def remove_zero(df, attribute):
    """ Removes rows where the value for an attribute is zero

    Args:
        df (pandas.DateFrame): dataframe that is being altered
        attribute (string): column that is being checked for zero values
    """
    for i in range(len(df)):
        if check_zero(df.loc[i, attribute]):
            df.drop(i, inplace=True)

def filter_data(df, attribute):
    """ Removes rows where the value for an attribute is zero and there NaN

    Args:
        df (pandas.DateFrame): dataframe that is being altered
        attribute (string): column that is being checked for zero values
    """

    # Removes NaN values and reindexes
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Removes rows where the value is zero
    remove_zero(df, attribute)

def aggregate_df(df, source, target, count=0):
    """ Groups by source and target columns, and then aggregates based on the source column

    Args:
        df (pandas.DateFrame): dataframe that is being used
        source (string): column of df that is being aggregated by
        target (string): column of df that is being counted

    Returns:
        aggregate_data (pandas.Series): aggregated data from dataframe
    """
    grouped_data = df.groupby([source, target])
    aggregated_data = grouped_data[source].aggregate("count")

    # Filter the aggregated_data by a certain count
    aggregated_data = aggregated_data[aggregated_data >= count]
    return aggregated_data
