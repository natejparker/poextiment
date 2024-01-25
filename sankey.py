"""
File: sankey.py
Description: Functions that help make sankey diagrams
Author: Nathan Parker
Date: 1 October 2023
"""

import pandas as pd
import plotly.graph_objects as go
import processing as proc

def produce_sankey_lists(aggregate):
    """ Breaks aggregated dataframe into lists for sankey diagram

    Args:
        aggregate (pandas.Series): aggregate data being broken into lists

    Returns:
        sankey_sources (list[any]): list of values for source nodes
        sankey_targets (list[any]): list of values for target nodes
        sankey_values (list[real]): list of values for node connection value
    """
    sankey_sources = list()
    sankey_targets = list()
    sankey_values = list()

    for i in range(len(aggregate.axes[0])):
        sankey_sources.append(aggregate.axes[0][i][0])

    for i in range(len(aggregate.axes[0])):
        sankey_targets.append(aggregate.axes[0][i][1])

    sankey_values = list(aggregate.values)

    return sankey_sources, sankey_targets, sankey_values

def stack_df(df, list_values, count=0):
    """ Combines aggregates of the multiple columns in the dataframe and return sankey lists

    Args:
        df (pd.DataFrame): DataFrame that we're taking values from
        list_values (list[string]): list of columns that we're pulling values from data frame
        count(int): Required number of instances for a sankey value to be displayed

    Returns:
        sankey_sources (list[any]): list of values for source nodes
        sankey_targets (list[any]): list of values for target nodes
        sankey_values (list[real]): list of values for node connection value
    """
    # Set values early
    sankey_sources = list()
    sankey_targets = list()
    sankey_values = list()

    while (len(list_values) >= 2):
        # Pull first value from column list
        source = list_values.pop(0)

        # Aggregate first two columns in column list
        aggregate = proc.aggregate_df(df, source, list_values[0], count=count)

        x, y, z = produce_sankey_lists(aggregate)

        sankey_sources += x
        sankey_targets += y
        sankey_values += z

    return sankey_sources, sankey_targets, sankey_values

def create_lcmap(list1, list2):
    """ Creates labels and a dictionary that maps sankey values with integers that represent their nodes

    Args:
        list1 (list[any]): list of sankey source node values
        list2 (list[any]): list of sankey target node values

    Returns:
        labels (list[any]): list of labels for the final sankey diagram
        lc_map (dict): maps the various node labels to the integer values that represent them
    """
    labels = list(set(list1 + list2))

    # generate n integers for n labels
    codes = list(range(len(labels)))

    # create a map from label to code
    lc_map = dict(zip(labels, codes))

    return labels, lc_map

def replace_list(list, lc_map):
    """ Replace values that are keys in the dict with values

    Args:
        list (list[any]): list of sankey values
        lc_map (dict): map of list to key-value pairs
    """
    for i in range(len(list)):
        list[i] = lc_map[list[i]]

def create_visualization(source, target, values, labels, save=None):
    """ Creates sankey diagram and displays it

    Args:
        source (list[int]): list of source nodes
        target (list[int]): list of target nodes
        values (list[real]): values between nodes
        labels (list[any]): node labels
    """

    link = {'source': source, 'target': target, 'value': values,
            'line': {'color': 'black', 'width': 1}}

    node = {'label': labels, 'pad': 50, 'thickness': 50,
            'line': {'color': 'black', 'width': 1}}

    sk = go.Sankey(link=link, node=node)
    fig = go.Figure(sk)

    # For dashboarding, you will want to return the fig
    # rather than show the fig.

    # fig.show()


    if(save):
        fig.write_image(save)
    else:
        fig.write_html('first_figure.html', auto_open=True)

def make_sankey(df, list_values, save=None, count=20):
    """ Takes a dataframe and a list of columns and turns it into a sankey diagram
    Args:
        df (pd.DataFrame): DataFrame that we're taking values from
        list_values (list[string]): list of columns that we're pulling values from data frame
    """
    # Creates list of sankey source nodes, sankey source target nodes, and sankey values
    sankey_source, sankey_targets, sankey_values = stack_df(df, list_values, count=count)

    #
    labels, lc_map = create_lcmap(sankey_source, sankey_targets)

    replace_list(sankey_source, lc_map)
    replace_list(sankey_targets, lc_map)

    create_visualization(sankey_source, sankey_targets, sankey_values, labels, save=save)

"""
def make_sankey(aggregate):
     Takes a pandas Series and turns it into a sankey diagram
    Args:
        aggregate (pandas.Series): aggregate of dataframe values
    
    # Creates list of sankey source nodes, sankey source target nodes, and sankey values
    sankey_source, sankey_targets, sankey_values = produce_sankey_lists(aggregate)

    #
    labels, lc_map = create_lcmap(sankey_source, sankey_targets)

    replace_list(sankey_source, lc_map)
    replace_list(sankey_targets, lc_map)

    create_visualization(sankey_source, sankey_targets, sankey_values, labels)
"""