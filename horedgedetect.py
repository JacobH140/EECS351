import numpy
from scipy.signal import convolve2d


def convolve(im, h):
  h = numpy.array(h)
  img = numpy.array(im)
  f = numpy.zeros(img.shape) # initialize the filtered result
  pad_width = max(h.shape)
  img = numpy.pad(img, pad_width=pad_width, mode='constant', constant_values=0) # Zero-pad to avoid going out of bounds while convolving
  for im_row_idx in range(len(img))[pad_width:-pad_width]:
    for im_col_idx in range(len(img[im_row_idx]))[pad_width:-pad_width]: #img[im_row_idx][pad_width:-pad_width]:
      for k_row_idx in range(len(h)):
        for k_col_idx in range(len(h[k_row_idx])):
          f[im_row_idx  - pad_width, im_col_idx  - pad_width] += h[k_row_idx, k_col_idx]*img[im_row_idx - k_row_idx, im_col_idx - k_col_idx]

  return f




def enhance_stagnant_contours(img):

    print("edge detector input shape", img.shape)
    print("edge detector input", img)

    #This is the global maximum gain
    gain_limit = 5
    edgethresh = 6
    #edgethresh is the threshold edge detection value. Tuned based on ASCII characters
    offset = 1
    #offset is the vertical offset of the gain matrix

    #Sobel Matrix
    Sy = numpy.array([[-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]])

    

    dy = numpy.array([[1], [-1]])

    #This is the array to be updated
    #letter = numpy.array([[1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
    #                [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
    #                [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
    #                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    #                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    #                [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
    #                [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
    #                [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]])

    img = numpy.pad(img, ((3, 3), (3, 3)), mode='constant', constant_values=0)

    #Gx = convolve2d(img, Sy, mode='valid')
    Gx = convolve(img, dy)

    m, n = Gx.shape
    conv2output = Gx.copy()
    #print(conv2output)
    largerOutput = numpy.zeros((m, n))
    for i in range(1, m):
        for j in range(n):
        #if Gx[i, j] != 0 and (Gx[i, j] == -1 * Gx[i-1, j] or Gx[i, j] == -1 * Gx[i+1, j]):
    #    if Gx[i, j] != 0 and abs(Gx[i,j]-Gx[i-1,j])>edgethresh:
    #        largerOutput[i, j] = 1
    #    else:
    #        largerOutput[i, j] = 0
            largerOutput[i,j]=abs(conv2output[i,j])

    output = largerOutput[2+offset:m-4+offset, 3:n-3]

    a, b = output.shape
    #Scale it, but not normalized
    for i in range(a):
        for j in range(1, b):
            if(output[i, j-1]!=0 and output[i, j]!=0):
                output[i, j] = output[i, j-1]+1

    #Find highest value present in matrix
    max_val = max(max(row) for row in output)

    #Normalize each row based on the global maximum gain_limit
    for i in range(a):
        for j in range(b):
            if(output[i, j]==1):
                output[i, j]=0
            output[i, j] = output[i, j]/max_val*gain_limit

    print("edge detector output shape", output.shape)
    print("edge detector output", output)
    return output