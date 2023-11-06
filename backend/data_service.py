import pandas as pd
import spacy
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

model_path = '../data/model.joblib'
vectorizer_path = '../data/vectorizer.joblib'

class DataService:
    def __init__(self):
        # Load Natural Language Processing english Model
        self.nlp = spacy.load('en_core_web_sm')

        # Load AI model
        self.model = self.load_model()


    def treat_comment(self, comment):
        spacy_comment = self.nlp(comment, disable=["parser", "tagger", "ner", "textcat"])
        treated_tokens = [w.text for w in spacy_comment if w.is_alpha and not w.is_stop]
        return " ".join(treated_tokens)


    def load_and_clean_csv(self):
        # Read csv
        dataframe = pd.read_csv('../data/train.csv')

        # If comment.sentiment is True, then the comment is negative
        dataframe['sentiment'] = dataframe[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].any(axis=1).astype(int)
        dataframe = dataframe.drop(["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"], axis=1)
        

        # Reduce dataset to have same number of positive & negative comments
        df_positive = dataframe[dataframe['sentiment'] == 1]
        df_negative = dataframe[dataframe['sentiment'] == 0]
        df_negative = df_negative.sample(n=len(dataframe[dataframe['sentiment'] == 1]), random_state=42)
        dataframe = pd.concat([df_positive, df_negative])

        # Clean all comments
        dataframe['comment_text'] = dataframe['comment_text'].apply(self.treat_comment)

        return dataframe
    

    def train_model(self):
        # Prepare training data
        Y = self.dataframe["sentiment"]
        self.vectorizer = TfidfVectorizer()
        X = self.vectorizer.fit_transform(self.dataframe["comment_text"])
        X, Y

        # Save vectorizer
        joblib.dump(self.vectorizer, vectorizer_path)

        # Train model
        X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.25,random_state=42)
        model = LogisticRegression()
        model.fit(X_train, Y_train)
        return model

    
    def verify_comment(self, comment):
        # Clean comment
        clean_comment = self.treat_comment(comment)

        # Load vectorizer
        self.vectorizer = joblib.load(vectorizer_path)

        # Predict
        encoded_comment = self.vectorizer.transform([clean_comment])
        prediction = self.model.predict(encoded_comment)
        return prediction[0]


    def load_model(self):
        if os.path.exists(model_path):    
            return joblib.load(model_path)
        else:
            # Load CSV data
            self.dataframe = self.load_and_clean_csv()
            
            # Train AI model
            model = self.train_model()
            
            # Save model
            joblib.dump(model, model_path)
            return model 
    