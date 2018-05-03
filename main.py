import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import utils

video_path = sys.argv[1]

signals, fps = utils.get_signals(video_path)
smooth_signals = utils.smooth_signal(signals, 15)

minimum_idxs = []
for i in range(1,smooth_signals.shape[0]-1):
    if smooth_signals[i]<smooth_signals[i-1] and smooth_signals[i] <smooth_signals[i+1]:
        minimum_idxs.append(i)

f_img = (max(minimum_idxs) - min(minimum_idxs))*1.0/(len(minimum_idxs)-1)
f =f_img/fps
print ("Frequency: {:f} Hz".format(f))
print ("Heart rate: {:f}/s".format(f*60))



