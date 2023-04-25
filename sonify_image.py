import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy
import cv2
import sonify_character
from music21 import stream, chord, key, stream, graph
import playback
import librosa

def d(x, y):
    """helper function for gaussian kernel"""
    return np.abs(x - y)

def gaussian_helper(sigma, x, y):
  coeff = 1 / (2 * np.pi * sigma**2)
  exponent = -(x**2 + y**2) / (2 * sigma**2)
  return coeff * np.exp(exponent)

def gaussian_kernel(filt_shape, sigma):
    gaussian_filter = np.zeros(filt_shape)
    # fill in the gaussian kernel: (looping bc short on time)
    for r in range(len(gaussian_filter)):
      for c in range(len(gaussian_filter[r])):
        center_x = np.floor(filt_shape[0]/2)
        center_y = np.floor(filt_shape[1]/2)
        x = d(r, center_x)
        y = d(c, center_y)
        gaussian_filter[r, c] = gaussian_helper(sigma, x, y)
    return gaussian_filter

def discard_every_other_pixel(gray_img):
    return gray_img[0::2, 0::2]

def downsample(img, kernel_size):
    """downsamples by performing convolution with a Gaussian kernel and then discarding every other pixel"""
    g_kernel = gaussian_kernel(filt_shape=kernel_size, sigma=1)
    img = scipy.ndimage.convolve(img, g_kernel)
    img = discard_every_other_pixel(img)
    return img


def sonify_image(img, output_filename="img_test.wav"):
    """a bit of a detour, but it is worth noting that sonify_character() can work on any BW image, not just characters... hence this function"""
    # start by downsampling
    sz = (5, 5)
    # (arbitrarily for now) call downsample 6 times
    for i in range(4):
        img = downsample(img, sz)
    thresh = 10
    _, img = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
    plt.imshow(img, cmap="gray", vmin=0, vmax=1)
    output = sonify_character.sonify_character(character=None, c=chord.Chord(['C2', 'e2-', 'G2']), num_timesteps=5, num_freq_bands=5, audio_on=True, output_filename=output_filename, img=img)

    plt.show()

    return output

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


if __name__ == '__main__':
    # load in the image, grayscale it, and downsample it
    im_gray = cv2.bitwise_not(cv2.imread('glasses.jpg', cv2.IMREAD_GRAYSCALE))


    thresh = 250
    # threshold the image
    _, im_bw = cv2.threshold(im_gray, thresh, 255, cv2.THRESH_BINARY)

    plt.imshow(im_gray)
    plt.show()

    # convert range to (0,1) instead of (0,255)
    #im_bw = (im_bw / 255).astype(int)


    # reverse color scheme
    #im_bw = np.where(im_bw==1, 0, 1)


    output = sonify_image(im_bw, "glasses.wav")
    p = graph.plot.HorizontalBarPitchSpaceOffset(output)
    p.run()

    # try to make spectrogram
    s = playback.init()
    adjusted_output = playback.scale_stream_tempo(output, 0.25)
    fp = playback.stream_to_midi_file(adjusted_output)
    filename = "image.wav"
    playback.synthesize_midi_test_1(s, fp, pan_ratio=0.5, output_filename=filename)

    spectrogram("image.wav")
    plt.show()


    #cv2.imwrite('blackwhite.png', im_bw)

