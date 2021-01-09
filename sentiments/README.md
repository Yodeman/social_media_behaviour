get_sentiment.py is the file where comments from nairaland.com are classsified into positive or negative sentiment,
using the nltk VADER sentiment analyzer which is a pre-trained model that uses rule-based values tuned to sentiments
from social media. It evaluates the text of a message and gives you an assessment of not just positive and negative,
but the intensity of that emotion as well which ranges from -1 to +1.

The emotions.py file is the web based version of the project, built with streamlit. The web page takes in input from
the user and predicts the emotion ans sentiment of the user input.
