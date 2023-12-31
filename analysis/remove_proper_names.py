"""
Created on 12/30/2023

It removes proper names from the processed text.
"""
import os
import spacy
import re

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Data')
FILE = "processed_comments_122923.txt"
NEWFILE = FILE.replace(".txt", "_nonames.txt")

nlp = spacy.load('en_core_web_trf') # change to en_core_web_sm if you have memory issues

def remove_proper_names(text):
    """
    Removes proper names from the text.

    Args:
        text (str): The text to be processed.

    Returns:
        str: The processed text.
    """
    doc = nlp(text)
    names = [e.text for e in doc.ents if e.label_.lower() == 'person']
    return [token for token in doc if token.text not in names]

def load_text(file_name = FILE):
    """
    Loads the processed text.

    Args:
        file_name (str): The name of the file to be loaded.

    Returns:
        str: The processed text.
    """
    with open(os.path.join(DATA_PATH, file_name), 'r', encoding="utf-8") as file:
        return file.readlines()
    
def save_text(text, file_name=NEWFILE):
    """
    Saves the processed text.

    Args:
        text (str): The processed text.
        file_name (str): The name of the file to be saved.
    """
    with open(os.path.join(DATA_PATH, file_name), 'w', encoding="utf-8") as file:
        for doc in text:
            file.write(','.join(filter(lambda x: x not in ['',' ','[]','[ ]'],doc))+'\n')
    print(f"File {file_name} saved!")

def format_text(text):
    docs = [re.sub("\d+", "", x.strip()).split(',') for x in text]
    return [" ".join(x) for x in docs]

def main():
    """
    Main function.
    """
    print("Loading text...")
    text = load_text()
    print("Formatting text...")
    text = format_text(text)
    print("Removing proper names...")
    new_text = [remove_proper_names(doc) for doc in text]
    save_text(new_text)
    print("Done!")

if __name__ == "__main__":
    main()

       