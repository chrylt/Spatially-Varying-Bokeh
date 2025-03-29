import sys
from PIL import Image
import numpy as np
import glob

for fileName in glob.glob('*.png'):

    if fileName.endswith(".mag.png"):
        continue

    print(fileName)

    outFileName = fileName[:-4] + ".mag.png"

    # load the image
    im = np.array(Image.open(fileName), dtype=float) / 255.0
    #im = im * 2.0 - 1.0
    # print(im.shape)

    # get the magnitude, zero out DC and shift it so DC is in the middle
    dft = abs(np.fft.fft2(im))
    dft[0,0] = 0.0
    dft = np.fft.fftshift(dft)

    # log and normalize
    imOut = np.log(1+dft)
    themin = np.min(imOut)
    themax = np.max(imOut)
    if themin != themax:
        imOut = (imOut - themin) / (themax - themin)
    else:
        imOut = imOut

    # save
    Image.fromarray((imOut*255.0).astype(np.uint8), mode="L").save(outFileName)
