import nltk
import pandas as pd
import pickle
import re
import sys

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sqlalchemy import create_engine

from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline


def load_data(database_filepath):
    """Load data from a provided database

    Args:
        database_filepath (str): Path to an SQLite database with training data.

    Returns:
        pandas.DataFrame: DataFrame containing incomming text messages.
        pandas.DataFrame: DataFrame with classified categories.
        list: A list of category names.
    """
    engine = create_engine("sqlite:///{}".format(database_filepath))
    with engine.connect() as connection:
        df = pd.read_sql_table(table_name="messages", con=connection)

    df.replace(2, 1, inplace=True)

    X = df.loc[:, "message"]
    y = df.iloc[:, 4:]

    category_names = y.columns.astype("str")

    return X, y, category_names


def tokenize(text):
    """Tokenize input text

    Args:
        text (str): Incomming message text.

    Returns:
        list: A list of cleaned up word tokens.
    """
    url_regex = (
        "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    )
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    words = word_tokenize(text.lower().strip())
    words = [
        WordNetLemmatizer().lemmatize(w)
        for w in words
        if w not in stopwords.words("english")
    ]

    return words


def build_model():
    """Build a model pipeline

    Returns:
        sklearn.pipeline.Pipeline: NLP ML pipeline
    """
    pipeline = Pipeline(
        [
            ("vect", CountVectorizer(tokenizer=tokenize)),
            ("tfidf", TfidfTransformer()),
            (
                "clf",
                MultiOutputClassifier(LinearSVC()),
            ),
        ]
    )

    parameters = {
        "vect__tokenizer": [None, tokenize],
        "clf__estimator__multi_class": ["ovr", "crammer_singer"],
        "clf__estimator__max_iter": [2000, 8000, 15000],
        "clf__estimator__dual": [False, True],
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=10)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluate the model

    Print a classification report based on data predicted by the model and
    a test dataset.

    Args:
        model (sklearn.pipeline.Pipeline): A trained model that can call predict()
        X_test (pandas.DataFrame): A test set of messages.
        Y_test (pandas.DataFrame): A test set of categories.
        category_names (list): A list of category names.
    """
    y_pred = model.predict(X_test)
    y_pred_df = pd.DataFrame(y_pred)
    y_pred_df.columns = category_names

    print(classification_report(Y_test, y_pred_df, target_names=category_names))


def save_model(model, model_filepath):
    """Save the model to a file

    Args:
        model (sklearn.pipeline.Pipeline): The model to save.
        model_filepath (str): A path to the Pickle formatted binary file.
    """
    with open(model_filepath, "wb") as message_classifier_file:
        pickle.dump(model, message_classifier_file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]

        print("Downloading NLTK data\n")
        nltk.download(["punkt", "wordnet", "stopwords"])

        print("Loading data...\n    DATABASE: {}".format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5)

        print("Building model...")
        model = build_model()

        print("Training model...")
        model.fit(X_train, Y_train)

        print("Evaluating model...")
        evaluate_model(model, X_test, Y_test, category_names)

        print("Saving model...\n    MODEL: {}".format(model_filepath))
        save_model(model, model_filepath)

        print("Trained model saved!")

    else:
        print(
            "Please provide the filepath of the disaster messages database "
            "as the first argument and the filepath of the pickle file to "
            "save the model to as the second argument. \n\nExample: python "
            "train_classifier.py ../data/DisasterResponse.db classifier.pkl"
        )


if __name__ == "__main__":
    main()