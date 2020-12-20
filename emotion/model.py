import pandas as pd
import re
import pickle
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score

data = pd.read_csv("data.csv")
target = data['Emotion']

def preprocess_tokenize(sentence):
    """
    Preprocess the data by removing noise.
    Tokenizes and lemmatize preprocessed sentence.
    """
    
    tokenizer = RegexpTokenizer(r'\w+')
    lemmatizer = WordNetLemmatizer()
    
    # remove hashtags and @names
    sentence = re.sub(r"(#[\d\w\.]+)", " ", sentence)
    sentence = re.sub(r"(@[\d\w\.]+)", " ", sentence)
    
    return ' '.join([lemmatizer.lemmatize(w) for w in tokenizer.tokenize(sentence.lower())])

def cv_score(model, X, y, scoring, k):
    scores = cross_val_score(model, X, y, scoring=scoring, cv=k)
    avg = sum(scores)/len(scores)
    print("Average cross validation score %0.2f" %(avg*100))

train = data['Text'].apply(preprocess_tokenize)

# split the data into training and testing set.
#X_train, X_test, y_train, y_test = train_test_split(data['Text'], data['Emotion'], test_size=0.25, random_state=1)

stop_words = stopwords.words('english')
count_vector = CountVectorizer(stop_words=stop_words)

training_set = count_vector.fit_transform(train)

model = MultinomialNB()
model.fit(training_set, target)

encoder_model = {'encoder':count_vector, 'model':model}

# check the accuracy of the model
cv_score(model, training_set, target, 'accuracy', 10)

# save encoder and model to file.
with open('encoder_model', 'wb') as f:
    pickle.dump(encoder_model, f)