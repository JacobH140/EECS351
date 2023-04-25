import pandas as pd
import pyfiglet
from music21 import stream, chord
import numpy as np
from music21 import key, stream, graph, analysis
import copy
from mido import MidiFile
from pyo import *  # cursed but the docs and S.O. use this convention
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from horedgedetect import enhance_stagnant_contours
import librosa

import playback


def sonify_character(character=None, c=None, pan_ratio=0.5, num_timesteps=-1, num_freq_bands=-1, audio_on=True, output_filename="test.wav", img=None, to_enhance_stagnant_contours=True):
    """Given a character and chord, RETURNS: a music21 stream object of music21
    chords corresponding to the character's sonification. current implementation (pyfiglet)
    doesn't use num_timesteps or num_freq_bands, but later versions might use different rendering
    strategies which would"""

    output = stream.Stream()

    if img is not None:
        df = pd.DataFrame(img)
        df[df != 0.0] = 1
        df = clean_leading_0_rows(df)
        df = clean_trailing_0_rows(df)
        df = df.loc[(df != 0).any(1)].reset_index(drop=True)  # remove rows of zeros
    else:
        df = make_character_array(character, font="univers")
        df[df != '0'] = 1  # make df all ones and zeros (binarized)

    df = flip_character_array(df)  # flip the character aray to make indexing more intuitive
    #print([df != '0'])

    print("df:", df)


    possible_voices = enumerate_possible_voices(df, c)


    for timestep in range(len(df.iloc[0])):
        #print(timestep)
        voicing = choose_voices(df, timestep, possible_voices)
       # print(voicing.pitches)
        output.append(voicing)


    # output.show()
    # init velocities to 0.4
    for c in output:
        for n in c:
            n.volume.velocityScalar = 0.1

    if to_enhance_stagnant_contours:
        print("de-flipped df", flip_character_array(df))
        #print("horz edge detector output", enhance_stagnant_contours(df.to_numpy(dtype=float)))
        gain_weights = df.to_numpy(dtype=float).T + (enhance_stagnant_contours(flip_character_array(df).to_numpy(dtype=float))).T
        print("df as numpy array", flip_character_array(df).to_numpy(dtype=float))
        # test the selective gain application below
        #gain_weights = df.to_numpy(dtype=float).T + np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        #                                                        [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        #                                                        [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        #                                                        [0,0,0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,0,0,0,0],
        #                                                      [0,0,0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,0,0,0,0],
        #                                                        [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        #                                                        [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        #                                                        [0,0,0,0,0,0,0,0,0,0,0,0,0,0]]).T

        print("gain weights shape", gain_weights.shape)
        print("gain weights, transposed: ", gain_weights.T)
        #print(df.to_numpy(dtype=float).shape)

        # the shape above does not work with the sparse structure of output (different timesteps have different number of notes), address this below
        gain_weights_sparse = []
        for i, _ in enumerate(gain_weights):
            gain_weights_sparse.append([])
            for j, _ in enumerate(gain_weights[i]):
                #print(i)
                #print(j)
                if gain_weights[i][j] != 0:
                    gain_weights_sparse[i].append(gain_weights[i][j])

        # print(gain_weights_sparse)


        #print("gain weights, ", gain_weights)

        for i, chd in enumerate(output):
            for j, n in enumerate(chd):
                #print("first loop ", i, j, id(n))
                print(0.1, gain_weights_sparse[i][j], end=" ")
                print("->", end=" ")
                #n.volume.velocity = int(127*0.4*gain_weights_sparse[i][j])
                n.volume.velocityScalar = n.volume.velocityScalar * gain_weights_sparse[i][j]
                print(n.volume.velocityScalar)
            print("\n")


        # test if the above worked
        for i, chd in enumerate(output):
            print(chd)
            for j, n in enumerate(chd):
                #print("second loop ", i, j, id(n))
                print(n.volume.velocityScalar, end=' ')
            print("\n")


    if audio_on:
        s = playback.init()
        adjusted_output = playback.scale_stream_tempo(output, 0.25)
        fp = playback.stream_to_midi_file(adjusted_output)
        playback.synthesize_midi_test_1(s, fp, pan_ratio, output_filename=output_filename)



    return output


def choose_voices(df, timestep, possible_voices):
    """given a character dataframe, timestep, and possible notes,
    decides with respect to the freq band content of dataframe @ t-step
    which voices to write to output. RETURNS: a new chord object — this
    one being the one we'd like to sonify"""

    # NOTE: notes and frequency_bands are coindexed

    output_chord = chord.Chord()
    frequency_bands = df.iloc[:, timestep]

    for band_index in range(len(frequency_bands)):
        #print(band_index)
        if frequency_bands[band_index] != '0' and frequency_bands[band_index] != 0.0:
            output_chord.add(copy.deepcopy(possible_voices[band_index]))

    return output_chord


def enumerate_possible_voices(df, c):
    """for the given timestep, returns ascending list of notes which are based off of spelling out the
    input chord c with respect to the number of freq bands"""
    # print(df)

    frequency_bands = df.iloc[:, 0]  # the timestep shouldn't matter, chose 0 arbitrarily
    chord_notes = [n for n in c]
    upper_voices = [octavate(chord_notes[(i + len(chord_notes)) % len(chord_notes)], (i + len(chord_notes)) // len(chord_notes)) for i in range(len(frequency_bands) - len(chord_notes))]
    chord_notes_enumerated = [*chord_notes, *upper_voices]

    return chord_notes_enumerated


def octavate(n, x):
    """transposes input up x octaves"""
    i = 0
    while i < x:
        n = n.transpose('P8')
        i += 1
    return n


def make_character_array(character, font="univers"):
    # helper function that uses pyfiglet to make a numpy array out of the input character
    figlet_string = pyfiglet.figlet_format(character, font=font)
    # print(figlet_string)

    output = [list(item.replace(' ', '0')) for item in figlet_string.splitlines()]
    df = pd.DataFrame(output)

    # now clean up the leading and trailing rows of all zeros in dataframe
    df = clean_leading_0_rows(df)
    df = clean_trailing_0_rows(df)

    return df


def clean_trailing_0_rows(df):
    index = 0
    num_rows = len(df[0])  # number of entries in first col is number of rows
    for ind, row in df[::-1].iterrows():
        if all([v == '0' for v in row.values]):  # i.e., if row has at least one nonzero element
            index += 1
        else:
            break
    return df.iloc[:(num_rows-index)].reset_index(drop=True)


def clean_leading_0_rows(df):
    index = 0
    for ind, row in df.iterrows():
        if all([v == '0' for v in row.values]):  # i.e., if row has at least one nonzero element
            index += 1
        else:
            break
    return df.iloc[index:].reset_index(drop=True)


def flip_character_array(df):
    print("before flip: ")
    print(df)
    for col_index in range(len(df.iloc[0])):
        df[col_index] = df[col_index].values[::-1]
    print("after flip: ")
    print(df)
    return df


def DFTandGAIN(sonify_character_output):
    started_server = playback.init()

    mid = Notein()
    fp = playback.stream_to_midi_file(sonify_character_output)
    #amp = MidiAdsr(mid["velocity"])
    pit = MToF(mid["pitch"])
    amp = MidiAdsr(mid["velocity"])

    osc = Sine(freq=pit, mul=amp)

    rec = Record(osc, filename="dftAndGain.wav", fileformat=0, sampletype=4)
    osc.out()

    # Opening the MIDI file...
    mid = MidiFile(fp)

    # ... and reading its content.
    for message in mid.play():
        started_server.addMidiEvent(*message.bytes())

    rec.stop()

    
    # read in test sine wave and plot its DFT magnitude
    sampling_rate, data = wav.read('dftAndGain.wav')

    data = data[1]

    # plot it
    plt.plot(data)
    plt.title('Test sine wave')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.show()

    # compute the magnitude of the DFT
    dft = np.fft.fft(data)
    dft_magnitude = np.abs(dft)

    # plot the magnitude
    plt.plot(dft_magnitude)
    plt.title('DFT Magnitude')
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    

def spectrogram(audio_filename):
    fig, ax = plt.subplots()
    y, sr = librosa.load(audio_filename)
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    s_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(s_dB, x_axis='time',
                                   y_axis='mel',
                                   fmax=8000, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-frequency spectrogram')

    #librosa.feature.tonnetz(y=y, sr=sr)

    fig, ax = plt.subplots(nrows=2, sharex=True)
    chroma = librosa.feature.chroma_stft(S=S, sr=sr)
    img = librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                                   y_axis='log', x_axis='time', ax=ax[0])
    fig.colorbar(img, ax=[ax[0]])
    ax[0].label_outer()
    img = librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=ax[1])
    fig.colorbar(img, ax=[ax[1]])

    y = librosa.effects.harmonic(y)
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    fig, ax = plt.subplots(nrows=2, sharex=True)
    img1 = librosa.display.specshow(tonnetz,
                                    y_axis='tonnetz', x_axis='time', ax=ax[0])
    ax[0].set(title='Tonal Centroids (Tonnetz)')
    ax[0].label_outer()
    img2 = librosa.display.specshow(librosa.feature.chroma_cqt(y=y, sr=sr),
                                    y_axis='chroma', x_axis='time', ax=ax[1])
    ax[1].set(title='Chroma')
    fig.colorbar(img1, ax=[ax[0]])
    fig.colorbar(img2, ax=[ax[1]])








if __name__ == "__main__":

    #DFTandGAIN()

    # sonify_character('1', chord.Chord(['C4', 'E4', 'G4']), 5, 5)
    output = sonify_character('H', chord.Chord(['C4', 'E4', 'G4']), 5, 5, audio_on=True, to_enhance_stagnant_contours=True)
    #DFTandGAIN(output)
    p = graph.plot.HorizontalBarPitchSpaceOffset(output)
    p.run()

    spectrogram("test.wav") # recall that sonify_character outputs a wav file called "test.wav" by default
    plt.show()
    # sonify_character('C', chord.Chord(['C4', 'E4', 'G4']), 5, 5).show()
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for letter in alphabet:
        print(letter)
        sonify_character(letter, chord.Chord(['C4', 'E4', 'G4']), 5, 5, audio_on=True, output_filename="letters/"+letter.lower()+".wav")

