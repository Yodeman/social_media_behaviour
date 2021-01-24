import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords

#load encoder and model
with open(r'.\encoder_model', 'rb') as f:
    encoder_model = pickle.load(f)
    encoder, model = encoder_model['encoder'], encoder_model['model']

#load sentiment classifier
with open(r"..\classifier\classifier", 'rb') as f:
    classifier, _, _ = pickle.load(f)
    
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

    
def predict(comment):
    encoded_comment = encoder.transform([comment])
    emotion = model.predict(encoded_comment)
    return emotion

def run():
    st.title('Emotion Classifier')
    st.write('This app is created to predict the emotion of the user with the comment.\nIt shows the sentiment of the comment too.')
    comment = st.text_area('Enter text')
    emotion = ''
    if st.button('Predict'):
        emotion = predict(comment)
        probabilities = classifier.prob_classify(transform(comment))
        predicted_sentiment = probabilities.max()
        st.success(f'This user is {emotion[0]}')
        st.write("Predicted sentiment:", predicted_sentiment)
        st.write("Probability:", round(probabilities.prob(predicted_sentiment), 2))

if __name__ == "__main__":
    run()
