import os
import cv2
import glob
from math import log10
from img_utils import *


IMGS_DIR = os.path.join(".", "images")
VID_IN_FILENAME = "project.mp4"
VID_OUT_FILENAME = "new_project.mp4"




# Step 1: Split video into images.

capture = cv2.VideoCapture(VID_IN_FILENAME)

# Extract video metadata.
frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
fps = capture.get(cv2.CAP_PROP_FPS)

images = stream_to_imgs(capture)

FILENAME_PAT = os.path.join(IMGS_DIR, "image_{:0{}d}.jpg")
id_length = max(int(log10(frame_count)) + 1, 1)

for _id, image in enumerate(images, start=1):
    filename = FILENAME_PAT.format(_id, id_length)
    cv2.imwrite(filename, image)

    # Keep track of the video dimensions.
    height, width, _ = image.shape
    dims = (width, height)

capture.release()





# Step 3: Tie the frames back into a video.

images = folder_to_imgs(IMGS_DIR, ".jpg")

imgs_to_vid(VID_OUT_FILENAME, dims, fps, images)
