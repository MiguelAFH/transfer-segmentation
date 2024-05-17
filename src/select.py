import cv2
import numpy as np

# Initialize global variables
drawing = False  # True if mouse is pressed
ix, iy = -1, -1  # Initial coordinates
rect = None

# Mouse callback function
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, rect

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # Draw a rectangle from the starting point to the current mouse position
            img_copy = img.copy()
            cv2.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow('image', img_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        rect = (ix, iy, x, y)
        cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)
        cv2.imshow('image', img)

# Load an image
img = cv2.imread('path/to/your/image.jpg')
img = cv2.resize(img, (800, 600))  # Resize for better visibility

# Create a window and bind the function to window
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_rectangle)

while True:
    cv2.imshow('image', img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:  # Escape key to exit
        break

cv2.destroyAllWindows()

if rect:
    print("Top-left corner: ({}, {})".format(rect[0], rect[1]))
    print("Bottom-right corner: ({}, {})".format(rect[2], rect[3]))
