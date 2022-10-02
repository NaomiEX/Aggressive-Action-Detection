import cv2
from PIL import Image


__all__ = ["annotate"]


BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)

THICKNESS = 2

FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 10


def annotated(image):
    """
    Given an OpenCV image, return the annotated image, in
    OpenCV format.
    """
    # Retrieve model predictions.
    height, width = image.shape
    pil_image = _cv2_to_pil(image)
    preds = predict(model, "cuda:0", pil_image, width, height)

    has_interaction = lambda pred: pred["verb"] != "NO_INTERACTION"

    # Annotate the image with model predictions.
    for pred in filter(has_interaction, preds):
        sub_box = pred["sub_box"]
        obj_box = pred["obj_box"]
        image = _draw_bbox(image, sub_box, RED, label="HUMAN")
        image = _draw_bbox(image, obj_box, GREEN, label=pred["obj_cat"])
        image = _draw_vector(image, sub_box, obj_box, BLUE, label=pred["verb"])

    return image


def _draw_bbox(image, box, color, label=None):
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
    image = cv2.rectangle(
        img=image, pt1=(x1, y1), pt2=(x2, y2), color=color, thickness=THICKNESS
    )

    # Put label above the bounding box.
    if label is not None:
        image = cv2.putText(
            img=image,
            org=(x1, y1 - 10),
            text=label,
            fontFace=FONT_FACE,
            fontScale=FONT_SCALE,
            color=color,
            thickness=THICKNESS,
        )

    return image


def _draw_vector(image, box1, box2, color, label=None):
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
    image = cv2.line(img=image, pt1=c1, pt2=c2, color=color, thickness=THICKNESS)

    # Put the label beside the line.
    if label is not None:
        image = cv2.putText(
            img=image,
            org=_midpoint(c1, c2),
            text=label,
            fontFace=FONT_FACE,
            fontScale=FONT_SCALE,
            color=color,
            thickness=THICKNESS,
        )

    return image


def _center(box):
    """
    Return the center point of the box.

    Parameters:
    box : Return the center point of the box.
    """
    x1, y1, x2, y2 = box
    pt1 = (x1, y1)
    pt2 = (x2, y2)
    return _midpoint(pt1, pt2)


def _midpoint(pt1, pt2):
    """
    Return the midpoint of two boxes.

    Parameters:
    pt1 : A point represented as (x1, y1).
    pt2 : A point represented as (x2, y2).
    """
    x1, y1 = pt1
    x2, y2 = pt2
    return (x1 + x2) / 2, (y1 + y2) / 2


def _cv2_to_pil(image):
    """
    Convert an OpenCV image to a PIL image.

    Parameter:
    image : OpenCV image, in BGR format.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image)
