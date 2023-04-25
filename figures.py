import sonify_text
import playback
import librosa
import matplotlib.pyplot as plt
import numpy as np
from music21 import key, stream, graph, chord
import sonify_character

def plot_midi(stream_obj):
    p = graph.plot.HorizontalBarPitchSpaceOffset(stream_obj)
    p.run()



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


    text = "Hello"
    output = sonify_text.sonify_text(text, 'C', plot=True, audio_on=True)
    #output = sonify_character.sonify_character('A', chord.Chord(['C4', 'E4', 'G4']), 5, 5, audio_on=True, enhance_stagnant_contours=False)
    output.write("midi", text + ".mid")

    s = playback.init()
    adjusted_output = playback.scale_stream_tempo(output, 0.25)
    fp = playback.stream_to_midi_file(adjusted_output)
    filename = text+".wav"
    playback.synthesize_midi_test_1(s, fp, pan_ratio=0.5, output_filename=filename)

    print(type(output))
    plot_midi(output)
    spectrogram(filename)
    plt.show()

