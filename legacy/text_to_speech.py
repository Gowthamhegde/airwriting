import pyttsx3

engine = pyttsx3.init()

def speak_word(word):
    engine.say(word)
    engine.runAndWait()
