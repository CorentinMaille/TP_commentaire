import pandas as pd
import spacy
import os
import joblib
from pandarallel import pandarallel
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
        # nlp configuration
        spacy_comment = self.nlp(comment, disable=["parser", "tagger", "ner", "textcat"])
        # keep some stop words
        treated_tokens = [w.text for w in spacy_comment if (w.is_alpha and not w.is_stop) or w.text.lower() in ["not", "no", "never", "none", "nothing", "don't", "myself", "like"]]
        return " ".join(treated_tokens)

    def load_and_clean_toxic(self):
        # Read csv
        toxic_df = pd.read_csv('../data/toxic.csv')

        # Clean dataframe
        toxic_df['status'] = toxic_df[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].any(axis=1).astype(int)
        toxic_df = toxic_df.drop(["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"], axis=1)

        # Reduce dataset
        positive_df = toxic_df[toxic_df['status'] == 1]
        negative_df = toxic_df[toxic_df['status'] == 0]
        negative_df = negative_df.sample(n=len(toxic_df[toxic_df['status'] == 1]), random_state=42)

        toxic_df = pd.concat([positive_df, negative_df])
        toxic_df = toxic_df.reset_index()
        return toxic_df

    def load_and_clean_suicide(self):
        suicide_df = pd.read_csv('../data/suicide.csv')
        suicide_df.replace({'non-suicide': 0, 'suicide': 2},inplace=True)
        suicide_df.rename(columns={'text': 'comment_text', 'class': 'status'}, inplace=True)
        suicide_df.drop('Unnamed: 0', axis=1, inplace=True)
        return suicide_df


    def load_dataframe(self):
        # load and clean all datasets
        toxic_df = self.load_and_clean_toxic()
        suicide_df = self.load_and_clean_suicide()

        # Merge datasets
        dataframe = pd.concat([suicide_df, toxic_df])
        self.dataframe = dataframe.sample(frac=1).reset_index(drop=True) # Shuffle data

        # Clean all comments
        pandarallel.initialize(progress_bar=True)

        self.dataframe['comment_text'] = self.dataframe['comment_text'].parallel_apply(self.treat_comment)

        return self.dataframe
    

    def train_model(self):
        # Prepare training data
        Y = self.dataframe["status"]
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

        return self.get_status_label_from_code(prediction[0]);


    def load_model(self):
        if os.path.exists(model_path):    
            return joblib.load(model_path)
        else:
            # Load CSV data
            self.dataframe = self.load_dataframe()
            
            # Train AI model
            model = self.train_model()
            
            # Save model
            joblib.dump(model, model_path)
            return model 
    

    def get_status_label_from_code(self, status_code):
        if status_code == 0:
            return 'positive'
        elif status_code == 1:
            return 'negative'
        elif status_code == 2:
            return 'suicidal'