import os
import cv2
import glob
from img_utils import *





# Step 1: Split video into images.

IMGS_DIR = "./images"

capture = cv2.VideoCapture("./project.mp4")
fps = capture.get(cv2.CAP_PROP_FPS)

images = stream_to_imgs(capture)
filename_pat = os.path.join(IMGS_DIR, "image{}.jpg")

for _id, image in enumerate(images, start=1):
    filename = filename_pat.format(_id)
    cv2.imwrite(filename, image)

capture.release()
