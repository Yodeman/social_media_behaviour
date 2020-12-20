import pandas as pd
import numpy as np
import random
import pickle

path = r'C:\Users\USER\Desktop\social\sentiments\\'

# Load sentiments from file
def load_file():
    politics_sentiments = pd.read_csv(path+"politics_sentiment.csv")
    education_sentiments = pd.read_csv(path+"education_sentiment.csv")
    health_sentiments = pd.read_csv(path+"health_sentiment.csv")

#merge all sentiments
def merge():
    all_data = pd.concat([politics_sentiments, education_sentiments, health_sentiments], ignore_index=True)
    X = all_data['comments']
    y = all_data['sentiments']
    #y = y.map({'pos':1, 'neg':-1, 'neu':0})
    #print(X.head(), y.head())


import nltk
from nltk.corpus import stopwords
def transform(comment):
    """
    Tokenizes comment and then apply lemmatization on the tokenized word.
    Returns a dictionary of lemmatized words mapped to 'True' as expected 
    by the classifier.
    """
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    lemmatizer = nltk.stem.WordNetLemmatizer()

    nltk_stopwords = set(stopwords.words('english'))
    pidgin_stopwords = set(['plz', 'abeg', 'pple', 'ppos', 'pple', 'dey', 'wan', 'wey', 'dis', 'dat', 'den', 'wetin', 'u', 'coz', 
                            'wtf', 'sabi', 'abi', 'diz', 'becuz', 'lol', 'lols', 'lolz', 'b4', 'urself', 'joor', 'deh', 'una', 
                            'abt', 'wat', 'tho'])
    stop_words = nltk_stopwords|pidgin_stopwords
    
    tokens = [lemmatizer.lemmatize(w) for w in tokenizer.tokenize(comment.lower()) if w not in stop_words]
    return {token:True for token in tokens}


# using nltk naive bayes classifier
from nltk import NaiveBayesClassifier, classify

def build_model():
    target = list(y.map({'pos':'positive', 'neg':'negative', 'neu':'neutral'}))
    predictors = list(X.apply(transform))
    model_data = list(zip(predictors, target))

    # mix the data
    random.shuffle(model_data)

    # split the data into training and testing set (80/20)
    threshold = int(len(model_data)*0.8)
    train_data = model_data[:threshold]
    test_data = model_data[threshold:]
    # model
    classifier = NaiveBayesClassifier.train(train_data)
    with open('classifier', 'wb') as f:
        pickle.dump((classifier, len(train_data), classify.accuracy(classifier, test_data)), f)

def custom_comment():
    """
    Classify user input comment.
    """
    print("Do yout wish to try a custom input?\nEnter yes/no or press the return key to exit. ")
    query = input().strip()
    if query.lower() == "yes":
        done = False
        while not done:
            comment = input("Enter comment: ").strip()
            if comment:
                print()
                probabilities = classifier.prob_classify(transform(comment))
                predicted_sentiment = probabilities.max()
                print("Sentiment Probabilities: {", end='')
                for i in probabilities.samples():
                    print(str(i) +":"+ str(round(probabilities.prob(i), 2)), end=', ')
                print("}")
                print("Predicted sentiment:", predicted_sentiment)
                #print("Highest Probability:", round(probabilities.prob(predicted_sentiment), 2))
                print()
            else:
                done = True
    else:
        return None
        

if __name__ == "__main__":
   
    with open("classifier", 'rb') as f:
        classifier, train_data, accuracy = pickle.load(f)
    print("Trained on: ", train_data)
    print("Accuracy: %0.2f" %(100*accuracy)+"%")
    print()
    N = 15
    print('\nTop ' + str(N) + ' most informative words:')
    for i, item in enumerate(classifier.most_informative_features()):
        print(str(i+1)+'. ' +item[0])
        if i==N-1:
            break
    print("\nSentiments Samples:\n")
    comments = ["Nigeria Government sucks", 
                "I pray ASUU call off the strike",
                "To hell with ASUU and the government",
                "Corona virus is fucking real, but people keep playing with it"
              ]
    for comment in comments:
        print("Comment:", comment)
        probabilities = classifier.prob_classify(transform(comment))
        predicted_sentiment = probabilities.max()
        print("Predicted sentiment:", predicted_sentiment)
        print("Probability:", round(probabilities.prob(predicted_sentiment), 2))
        print()
    
    custom_comment()