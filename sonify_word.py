from nltk import tokenize
import sonify_character
from music21 import chord, roman, key, stream, interval
import playback
from figures import spectrogram


def sonify_word(harmonic_function, mode, word, audio_on=True):  # harmonic function is an abstract roman numeral
    segments = segmentize(word)
    c = make_chord(harmonic_function, mode)
    c = add_extensions(c, mode, word)
    print("sonify_word's chord for the word ", word, " is ", harmonic_function, mode, "i.e.,", c)

    output = stream.Stream()

    for segment in segments:
        # print(len(segment))
        if len(segment) % 2 == 0:
            pan_division_width = 1.0 / (len(segment) - 1)
        else:
            pan_division_width = 1.0 / len(segment)
        for index, character in enumerate(segment):
            pan_ratio = index*pan_division_width
            character_stream = sonify_character.sonify_character(character, c, pan_ratio, audio_on=audio_on)
            output.append(character_stream)
    return output


def add_extensions(c, m, word):
    """given chord c and mode m, adds extensions to c based on length, so that listeners
    are able to  hear immediately how long a word is in the same one can while reading"""
    # roughly corresponds to an additional extension per syllable beyond 1

    num_extensions = len(syllable_count(word)) - 1
    offset = False  # to ensure that adding a #11 doesn't also sharpen further extensions
    for i in range(num_extensions):
        third = interval.GenericInterval(3)
        extension = third.transposePitchKeyAware(c[-1].pitch, m)

        # check to see if we need a #11 instead of nat 11
        if i == 2 and (c.commonName == 'major-ninth chord'):

            extension = extension.transpose(1)  # up a half step
            c.add(extension)
            offset = True  # need to ensure that adding a #11 doesn't also sharpen further extensions
        else:
            if offset:
                extension = extension.transpose(-1)
            c.add(extension)
    return c


def make_chord(harmonic_function, mode):
    """given harmonic function and mode, returns the corresponding chord"""
    harmonic_function.key = mode
    notes = [p.transpose('-P8') for p in harmonic_function.pitches]
    print(notes)
    return chord.Chord(notes)


def segmentize(word):
    # divides word into segments
    if len(word) <= 3:
        segments = [word]
    else:
        segments = syllable_count(word)

    return [segment.upper() for segment in segments]


def syllable_count(word):
    tk = tokenize.SyllableTokenizer()
    return tk.tokenize(word.lower())


if __name__ == "__main__":
    # print(syllable_count("Antidisestablishmentarianism"))
    harm_func = roman.RomanNumeral('I')
    mode = key.Key('C')
    word = 'Hi'
    output = sonify_word(harm_func, mode, word)
    output.write("midi", word + ".mid")
    # output.show()
    # make_chord(harm_func, mode)
    # print(add_extensions(chord.Chord('C4 E4 G4'), key.Key('F'), 'ambitious'))

