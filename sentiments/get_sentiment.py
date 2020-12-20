import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer

path = r'C:\Users\USER\Desktop\social\comments\\'

politics_comments = pd.read_table(path+"politics_comments.txt", sep='\n',
                   header=None, names=["comments"])
education_comments = pd.read_table(path+"education_comments.txt", sep='\n',
                   header=None, names=["comments"])
health_comments = pd.read_table(path+"Health_comments.txt", sep='\n',
                   header=None, names=["comments"])

# Using the NLTK VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

def get_sentiment(data):
    """
    Get sentiment for the comments in the data.
    """
    sentiments = []
    sentiment_probab = []
    for comment in data['comments']:
        score = analyzer.polarity_scores(comment)
        sentiment_probab.append(score['compound'])
        if score['compound'] > 0.05:
            sentiments.append('pos')
        elif score['compound'] < -0.05:
            sentiments.append('neg')
        else:
            sentiments.append('neu')
    data['sentiments'] = sentiments
    data['compound'] = sentiment_probab
    return data

politics_sentiment = get_sentiment(politics_comments)
politics_sentiment.to_csv('politics_sentiment.csv', index=False)
education_sentiment = get_sentiment(education_comments)
education_sentiment.to_csv('education_sentiment.csv', index=False)
health_sentiment = get_sentiment(health_comments)
health_sentiment.to_csv('health_sentiment.csv', index=False)
