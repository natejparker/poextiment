"""
filename: poextiment.py
description: An extensible reusable library for poem (or lyric) analysis and comparison
Credit: Skeleton of code provided by John Rachlin (faculty for CCIS at Northeastern University)
"""

from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import string
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.figure_factory as ff
from functools import reduce
from sankey import create_lcmap, replace_list, create_visualization
import json
from exceptions import *

class Poextiment:

    def __init__(self):
        # string  --> {filename/label --> statistics}
        # "wordcounts" --> {"A": wc_A, "B": wc_B, ....}
        self.data = defaultdict(dict)

        # Stores stopwords to be used for the Poextiment instance
        self.stopwords = list()

        # Stores the names of the difference texts that are loaded
        self.textlist = list()

    # Takes the results from the parser output and stores the data
    def _save_results(self, label, results):
        for k, v in results.items():
            self.data[k][label] = v

        self.textlist.append(label)

    def _print_results(self):
        print(self.data)

    @staticmethod
    def txt_parser(filename, encoding):
        """ Converts a .txt file into a Python string

        Parameters:
            filename (string): name of the txt file
            encoding (string): encoding type of the txt file

        Output:
            text (string): contents of the text file
        """

        f = open(filename, encoding=encoding)

        text = f.read()
        f.close()

        return text

    @staticmethod
    def json_parser(filename, keyname):
        """ Converts a .json file into a Python string

        Parameters:
            filename (string): name of the txt file
            keyname (string): key that stores the poem content

        Output:
            text (string): contents of the poem from the json file
        """
        f = open(filename)
        raw = json.load(f)

        # Error if key is not found in the json
        if not(keyname in raw.keys()):
            raise JsonKeyNotFound(keyname)

        text = raw[keyname]
        f.close()

        return text

    @staticmethod
    def sanitize_text(text):
        """ Removes the capitalization and punctuation from a given file

        Parameters:
            text (string): Poem to be cleaned

        Output:
            clean_text (string): Post-processed poem
        """

        downcase_text = text.lower()
        clean_text = downcase_text.translate(str.maketrans('', '', string.punctuation))

        return clean_text

    @staticmethod
    def word_count(text):
        """ Finds the word count in poem specific text formats

        Parameters:
            text (string): Processed input text

        Output:
            word_count(dict (string : int)): Word counts of
        """

        word_count = Counter(text.replace('\n', ' ').split(' '))
        del word_count['']

        return word_count

    def remove_stopwords(self, worddict):
        """ Removes stop words from the overall word count

        Parameters:
            worddict (dict{string : int}): Word count totals

        Output:
            updateddict (dict{string : int}): Word count totals (minus the stop words)
        """

        updateddict = worddict.copy()
        keylist = list(worddict.keys())
        for key in keylist:
             if key in self.stopwords:
                 updateddict.pop(key, None)

        return updateddict

    @staticmethod
    def stanza_sentiments(stanza_list):
        """ Returns a list of sentiment scores for the stanzas

        Parameters:
            stanza_list (list[string]) : List of stanzas from text file

        Output:
            sentiment_list (list[real]) : List of sentiment scores
        """
        # initialize an empty list and an instance of the sentiment analyzer
        sentiment_list = list()

        sid_obj = SentimentIntensityAnalyzer()

        # Analyzes the sentiment of each stanza, converts to a compound score, stores in list
        for i in range(len(stanza_list)):
            sentiment_list.append(sid_obj.polarity_scores(stanza_list[i])['compound'])

        return sentiment_list

    @staticmethod
    def stanza_to_lines(stanzalist):
        """ Converts a list of stanzas into a list of lines

        Parameters:
            stanzalist (list[string]): List of strings that are stanzas

        Output:
            line_list (list[string]): List of strings that are lines

        """
        line_list = stanzalist
        for i in range(len(line_list)):
            line_list[i] = line_list[i].split('\n')

        # Flatten the lists
        flat_lines = [item for sublist in line_list for item in sublist]

        return flat_lines

    def parser(self, filename, filetype=None, encoding='utf-8', json_tag=None):

        text = ''

        if filetype is None or filetype == 'txt':
            text = self.txt_parser(filename, encoding)
        elif filetype == 'json':
            text = self.json_parser(filename, json_tag)
        else:
            raise InvalidFiletype(filetype)

        return text


    def _processing(self, text):
        """ DEMONSTRATION ONLY:
        Extracting word counts and number of words
        as a random number.
        Replace with a real parser that processes
        your input file fully.  (Remove punctuation,
        convert to lowercase, etc.)   """

        # Sanitize text file
        clean_text = Poextiment.sanitize_text(text)

        # Determine word counts + determine total words
        word_count = Poextiment.word_count(clean_text)
        num_words = sum(word_count.values())

        updated_count = self.remove_stopwords(word_count)

        # Split the text into stanzas
        stanza_list = clean_text.split('\n\n')

        # Calculate the sentiment score for each stanza
        stanza_sentiment = self.stanza_sentiments(stanza_list)

        # Split the text into lines
        line_list = self.stanza_to_lines(stanza_list)

        # Calculate the sentiment score for every line
        line_sentiment = self.stanza_sentiments(line_list)

        results = {
            'wordcount': updated_count,
            'numwords': num_words,
            'stanzasentiment': stanza_sentiment,
            'linesentiment': line_sentiment
        }
        return results

    def load_text(self, filename, filetype=None, label=None, encoding='utf-8', json_tag=None, parser=None):
        """ Registers a text document with the framework
        Extracts and stores data to be used in later
        visualizations. """

        if parser is None:
            # Parse the file for string (supported types are .txt and .json)
            text = self.parser(filename, filetype, encoding, json_tag)
        else:
            # Allows the flexibility of the user to add their own parser
            text = parser(filename)

        # Process the string into the different results
        if not(isinstance(text, str)):
            raise SelfParseNotString()

        results = self._processing(text)

        if label is None:
            label = filename

        # store the results of processing one file
        # in the internal state (data)
        self._save_results(label, results)

    def load_stop_words(self, stopfile, encoding='utf-8'):
        """ Loads a file that contains common words that should be ignored into the data.

        Parameters:
            stopfile (string): The .txt file. Format should be one word per line.

        """

        # Reads the input of the file
        stopwords = self.txt_parser(stopfile, encoding)

        # Converts the input into a list of words
        liststop = stopwords.split('\n')

        # Store in the framework
        self.stopwords = liststop

    def text_to_sankey(self, word_list=None, k=5):
        """ A DEMONSTRATION OF A CUSTOM VISUALIZATION
        A trivially simple barchart comparing number
        of words in each registered text file. """

        if word_list is None:
            # Pull out the word count for each text
            word_counts = self.data['wordcount']

            # Add the counters of the different texts together
            text_counters = reduce((lambda x, y: x + y), list(word_counts.values()))

            # Set of the most common words
            word_set = set(dict(text_counters.most_common(k)).keys())

        else:
            word_set = word_list
        # ERROR IF K IS GREATER THAN AVAILABLE WORDS

        # Lists to create map of strings to nodes
        text_list = self.textlist.copy()
        words = list(word_set)

        labels, lc_map = create_lcmap(text_list, words)

        # Initialize sankey lists
        sankey_source = list()
        sankey_target = list()
        sankey_values = list()

        # Fill sankey source list
        for text in text_list:
            for time in range(len(words)):
                sankey_source.append(text)

        # Fill sankey target list
        sankey_target = words * len(text_list)

        # Fill sankey value list
        for i in range(len(sankey_source)):
            sankey_values.append(self.data['wordcount'][sankey_source[i]][sankey_target[i]])

        # Replace lists with the integers representing them
        replace_list(sankey_source, lc_map)
        replace_list(sankey_target, lc_map)

        # Create final visualization
        create_visualization(sankey_source, sankey_target, sankey_values, labels)

    def hist_subplots(self, remove_neutral=True):

        list_of_texts = self.textlist.copy()

        subcolumns = 0
        subrows = 0

        # Determine subplot dimensions
        text_count = len(list_of_texts)
        for i in range(5, 0, -1):
            if text_count % i == 0:
                subcolumns = i
                subrows = int(text_count / i)
                break

        # Titles of subplots
        titles = [text for text in list_of_texts]

        hist_fig = make_subplots(
            rows=subrows,
            cols=subcolumns,
            subplot_titles=titles,
            shared_xaxes='all'
        )

        current_text = 0

        for row in range(1, (subrows+1)):
            for column in range(1, (subcolumns+1)):
                text_name = list_of_texts[current_text]
                data = self.data['linesentiment'][text_name]

                if remove_neutral is True:
                    data = [i for i in data if i != 0]

                y, x = np.histogram(data, bins=20, range=[-1, 1])
                x = [(a + b) / 2 for a, b in zip(x, x[1:])]

                figure = px.bar(x=x, y=y, color=x, color_continuous_scale=px.colors.diverging.RdYlGn
                                ).update_traces(marker_line_color="black"
                                                ).update_layout(bargap=0, coloraxis_cmid=0)

                for trace in range(len(figure["data"])):
                    hist_fig.add_trace((figure["data"][trace]), row=row, col=column)

                current_text += 1

        hist_fig.update_layout(
            height=600
        )


        hist_fig.write_html('second_figure.html', auto_open=True)


    def swarm_sentiment(self):

        # Initialize an empty DataFrame
        df = pd.DataFrame(columns=['text', 'sentiment'])

        # Create a copy of the text labels
        list_of_texts = self.textlist.copy()

        # Loop that fills the DataFrame with the stanza data
        for text in list_of_texts:
            for stanza in self.data['stanzasentiment'][text]:

                # Insert a row into the dataframe
                df.loc[-1] = [text, stanza]
                df.index = df.index + 1
                df = df.sort_index()

        # Create swarmplot
        sns.set()
        plt.figure(figsize=(17, 8))
        swarm = sns.swarmplot(df, x='sentiment', hue='text', palette='bright', size=10, linewidth=2)
        sns.move_legend(swarm, "upper left", bbox_to_anchor=(0.68, 1))

        plt.savefig('stanzasentiments.png', format='png')