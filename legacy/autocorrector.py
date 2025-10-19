from textblob import TextBlob

def correct_word(word):
    blob = TextBlob(word)
    return str(blob.correct())
