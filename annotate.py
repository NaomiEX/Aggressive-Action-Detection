import cv2


BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)

BBOX_THICKNESS = 5

HUMAN_COLOR = RED
WEAPON_COLOR = BLUE

FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 10
FONT_COLOR = BLUE
FONT_THICKNESS = 2


image = cv2.imread("./images/image_100.jpg")


# Draw bounding box.
x1, y1 = 100, 100  # Top-left point.
x2, y2 = 200, 200  # Bottom-right point.
image = cv2.rectangle(
    img=image, pt1=(x1, y1), pt2=(x2, y2), color=BLUE, thickness=BBOX_THICKNESS
)

# Draw label
text = "LABEL"
image = cv2.putText(
    img=image,
    org=(x1, y1 - 10),
    text=text,
    fontFace=FONT_FACE,
    fontScale=FONT_SCALE,
    color=FONT_COLOR,
    thickness=FONT_THICKNESS,
)

# Draw interaction vector.
image = cv2.line(image, (150, 150), (300, 300), GREEN, thickness=5)

cv2.imwrite("./WINNAME.jpg", image)


def _midpoint(pt1, pt2):
    x1, y1 = pt1
    x2, y2 = pt2
    return (x1+x2)//2, (y1+y2)//2