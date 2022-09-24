import cv2
import os
import glob


def stream_to_imgs(capture):
    """
    Return an iterator to the images in a given OpenCV
    VideoCapture stream. Involves IO.

    Parameter:
    capture : OpenCV VideoCapture stream, must stay opened.
    """
    # Frames can only be extracted if the video capture
    # stream is opened.
    assert capture.isOpened(), "Error opening VideoCapture stream."

    # Extract frames until the stream is exhausted.
    while True:
        success, image = capture.read()
        if not success:
            break
        yield image


def folder_to_imgs(folder_path, img_extension):
    """
    Return an iterator to the images in the specified folder
    of the specified type. Involves IO.

    Parameters:
    folder_path   : The path to the folder containing the images.
    img_extension : Image format extension, e.g. ".jpg".
    """
    # Create a path that matches all images in the folder of
    # the desired type.
    path = os.path.join(folder_path, f"*{img_extension}")

    filenames = sorted(glob.glob(path))

    # Return an iterator to the images read.
    # TODO Check whether the images yielded are in correct order.
    return map(cv2.imread, filenames)


def imgs_to_vid(filename, dims, fps, images):
    """
    Parameters:
    filename : The name of the video output, must end with ".mp4".
    dims     : A tuple (width, height) specifying the video dimensions.
    fps      : Frames per second.
    images   : An iterator to OpenCV Mat image objects.
    """
    # Function limitation: can only output videos in mp4 codec.
    assert filename.endswith(".mp4"), "Attempted to output non mp4-codec video."

    # Create a OpenCV VideoWriter object as IO wrapper.
    writer = cv2.VideoWriter(
        filename,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        dims
    )

    # Write the image into the video container.
    for image in images:
        writer.write(image)

    # Clean up to prevent memory leaks.
    writer.release()
