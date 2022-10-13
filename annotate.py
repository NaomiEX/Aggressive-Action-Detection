import cv2
from PIL import Image
from pathlib import Path
import json

__all__ = ["annotate"]


BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)

THICKNESS = 2
import cv2


__all__ = ("annotated",)


BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)

THICKNESS = 2

FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1

def annotated(frame, preds):
    """
    Given a video frame and HOI model predictions, return the
    annotated video frame.

    Parameters:
    frame : A video frame, stored in OpenCV image format.
    preds : The HOI model predictions for a video frame.
    """
    # If there is no detected aggressive HOI in the frame,
    # return the frame.
    if preds is None:
        return frame

    has_interaction = lambda pred: pred["verb"] != "NO_INTERACTION"

    # Annotate the video frame with model predictions.
    for pred in filter(has_interaction, preds):
        sub_box = tuple(map(round, pred["sub_box"]))
        obj_box = tuple(map(round, pred["obj_box"]))
        frame = _draw_bbox(frame, sub_box, RED, label="HUMAN")
        frame = _draw_bbox(frame, obj_box, GREEN, label=pred["obj_cat"])
        frame = _draw_vector(frame, sub_box, obj_box, BLUE, label=pred["verb"])

    return frame


def _draw_bbox(frame, box, color, label=None):
    """
    Draw the bounding box on the OpenCV image, return
    the annotated image.

    Parameters:
    image : An image in OpenCV BGR format.
    box1  : A box.
    box2  : Another box.
    color : Color of the interaction vector, in (B, G, R).
    label : The text placed besides the vector, if any.
    """
    # Extract the coordinates of the bounding boxes.
    # (x1, y1): Coordinates of the top-left corner.
    # (x2, y2): Coordinates of the bottom-right corner.
    x1, y1, x2, y2 = box

    # Draw bounding box.
    frame = cv2.rectangle(
        img=frame, pt1=(x1, y1), pt2=(x2, y2), color=color, thickness=THICKNESS
    )

    # Put label above the bounding box.
    if label is not None:
        frame = cv2.putText(
            img=frame,
            org=(x1, y1 - 10),
            text=label,
            fontFace=FONT_FACE,
            fontScale=FONT_SCALE,
            color=color,
            thickness=THICKNESS,
        )

    return frame


def _draw_vector(frame, box1, box2, color, label=None):
    """
    Draw the interaction vector on the OpenCV image, return
    the annotated image.

    Parameters:
    image : An image in OpenCV BGR format.
    box1  : A box.
    box2  : Another box.
    color : Color of the interaction vector, in (B, G, R).
    label : The text placed besides the vector, if any.
    """
    # Find the center point of the boxes.
    c1, c2 = _center(box1), _center(box2)

    # Draw the interaction vector (as a line).
    frame = cv2.line(img=frame, pt1=c1, pt2=c2, color=color, thickness=THICKNESS)

    # Put the label beside the line.
    if label is not None:
        mx, my = _midpoint(c1, c2)
        frame = cv2.putText(
            img=frame,
            org=(mx, my - 5),
            text=label,
            fontFace=FONT_FACE,
            fontScale=FONT_SCALE,
            color=color,
            thickness=THICKNESS,
        )

    return frame


def _center(box):
    """
    Return the center point of the box.

    Parameter:
    box : Return the center point of the box.
    """
    x1, y1, x2, y2 = box
    pt1 = (x1, y1)
    pt2 = (x2, y2)
    return _midpoint(pt1, pt2)


def _midpoint(pt1, pt2):
    """
    Return the midpoint of two points.

    Parameters:
    pt1 : A point represented as (x1, y1).
    pt2 : A point represented as (x2, y2).
    """
    x1, y1 = pt1
    x2, y2 = pt2
    return (x1 + x2) // 2, (y1 + y2) // 2


if __name__ == "__main__":
    # image = cv2.imread("./frame_1.jpg")
    filename = 6385
    image = cv2.imread(f"model/data/surveillance/indoors/{filename}.jpg")
    with open(Path(f"predictions/{filename}.json"), "r") as rf:
        preds = json.load(rf)
    image = annotated(image,preds)
    # image = annotated(image)
    cv2.imwrite(f"./temp_{filename}_output.jpg", image)
