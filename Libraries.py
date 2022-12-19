#Libraries 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
%matplotlib inline
import os
# nltk is natural language toolkit (used when working with human language data)
# Import functions for data preprocessing & data preparation
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from sklearn.feature_extraction.text import CountVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer, LancasterStemmer # converting the words to their root/base/stem form.
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords #function to remove the stop words like in,an,the,is,are etc.
from nltk.corpus import wordnet
import string
from string import punctuation
import nltk
import re
