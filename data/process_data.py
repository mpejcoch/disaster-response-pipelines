import sys
import pandas as pd

from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Loads data from provided CSV files with messages and headers.
    A data model specific to this application is expected.

    Args:
        messages_filepath (str): A path to the CSV file containing messages.
        categories_filepath (str): A path to the CSV file containing categories of messages.

    Returns:
        pandas.DataFrame: A dataframe with combined information from both of the CSV files.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on="id")
    return df


def clean_data(df):
    """Clean duplicates and create dummy variables for categorical data

    Args:
        df (pandas.DataFrame): A dataframe with merged data from loaded CSV files

    Returns:
        pandas.DataFrame: Cleaned up data, that can be used for ML pipeline training.
    """
    categories = df["categories"].str.split(";", expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda s: s[:-2])
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    df.drop("categories", axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    print("Dropping {} duplicates.".format(df.duplicated().sum()))
    df.drop_duplicates(inplace=True)
    # Drop the column with no "1" values
    df.drop(columns="child_alone", inplace=True)

    return df


def save_data(df, database_filename):
    """Store data in an SQLite database

    Args:
        df (pandas.DataFrame): A dataframe.
        database_filename (str): Target SQLite database file name.
    """
    engine = create_engine("sqlite:///{}".format(database_filename))
    df.to_sql("messages", engine, index=False, if_exists="replace")


def main():
    """Execute all the data processing steps in the correct order."""
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print(
            "Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}".format(
                messages_filepath, categories_filepath
            )
        )
        df = load_data(messages_filepath, categories_filepath)

        print("Cleaning data...")
        df = clean_data(df)

        print("Saving data...\n    DATABASE: {}".format(database_filepath))
        save_data(df, database_filepath)

        print("Cleaned data saved to database!")

    else:
        print(
            "Please provide the filepaths of the messages and categories "
            "datasets as the first and second argument respectively, as "
            "well as the filepath of the database to save the cleaned data "
            "to as the third argument. \n\nExample: python process_data.py "
            "disaster_messages.csv disaster_categories.csv "
            "DisasterResponse.db"
        )


if __name__ == "__main__":
    main()