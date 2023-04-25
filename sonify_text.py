from transformers import pipeline
from music21 import key, stream, graph
from nltk import sent_tokenize
import sonify_sentence


"""should we choose mode at the text level or the sentence level? pros of the former would be the enforcement on non-jarring key changes... except, 
the mode changes wouldn't be jarring in the latter either, provided that the sentiment transitions between sentences weren't *themselves* jarring — in 
which case the key changes should(!) be jarring! So, training will occur once, but classification will occur fresh each sentence"""

def sonify_text(text, tonal_center, plot=True, audio_on=True):

    output = stream.Stream()

    classifier = train_classifier() # time consuming; find out how to load trained model from local directory later (should not be too bad)
    sentences = sent_tokenize(text)
    for sentence in sentences:
        mode = classify_mode(sentence, classifier, tonal_center)
        output.append(sonify_sentence.sonify_sentence(sentence, mode, audio_on))

    if plot:
        graph.plot.HorizontalBarPitchClassOffset(output).run()
        graph.plot.HistogramPitchSpace(output).run()

    return output


def classify_mode(sentence, classifier, tonal_center):
    # modes = ["Lydian", "Ionian", "Mixolydian", "Dorian", "Aeolian", "Phrygian"]
    print(classify_sentence_sentiment(sentence, classifier))
    # labels are returned (*i think* in order from highest to lowest, so labels[0] is who we want to go with
    emotions = {
                "ecstasy": "Lydian", "serenity": "Lydian", "love": "Lydian",
                "joy": "Ionian", "trust": "Ionian", "acceptance": "Ionian", "optimism": "Ionian",
                "amazement": "Mixolydian", "surprise": "Mixolydian", "distraction": "Mixolydian", "anticipation": "Mixolydian",
                "neutrality": "Dorian", "boredom": "Dorian", "submission": "Dorian", "apprehension": "Dorian",
                "grief": "Aeolian", "sadness": "Aeolian", "pensiveness": "Aeolian", "remorse": "Aeolian",
                "anger": "Phrygian", "loathing": "Phrygian", "aggressiveness": "Phrygian", "disgust": "Phrygian"
               }
    mode = emotions[classify_sentence_sentiment(sentence, classifier)['labels'][0]]
    print("Mode: " + mode)
    return key.Key(tonal_center, mode)



def train_classifier():
    """only needs to be called when changes to the pipeline have been made"""
    model = pipeline("zero-shot-classification")
    # model.save_pretrained("/Users/jacobhume/PycharmProjects/EarReaderSSD/sentiment_classifier")
    # torch.save(model.state_dict("/Users/jacobhume/PycharmProjects/EarReaderSSD/sentiment_classifier"))
    return model

def classify_sentence_sentiment(sentence, classifier):

    # tokenizer = AutoTokenizer.from_pretrained("/Users/jacobhume/PycharmProjects/EarReaderSSD/sentiment_classifier")
     #classifier = AutoModelForSequenceClassification.from_pretrained("/Users/jacobhume/PycharmProjects/EarReaderSSD/sentiment_classifier")

    # labels to which we want to assign data (NOT FINAL). Based on Plutchtik emotion model
    candidate_labels = ["ecstasy", "joy", "serenity", "love",                     # lydian
                        "admiration", "trust", "acceptance", "optimism",          # ionian
                        "amazement", "surprise", "distraction", "anticipation",   # mixolydian
                        "neutral", "boredom", "submission", "apprehension",       # dorian
                        "grief", "sadness", "pensiveness", "remorse",             # aeolian (consider harmonic minor?)
                        "anger", "loathing", "aggressiveness", "disgust"]         # phrygian
                                                                                  # exclude locrian for now...



    prediction = classifier(sentence, candidate_labels)
    return prediction



if __name__ == "__main__":
    # print(classify_sentence_sentiment("this made my day!"))
    # train_classifier()
    # classify_mode("this made my day!")
    #text = "There is a cat on the hill! Are you afraid?"
    text = "Hello World!"
    output = sonify_text(text, 'C', plot=True, audio_on=False)
    print(output)
    output.write("midi", text + ".mid")


