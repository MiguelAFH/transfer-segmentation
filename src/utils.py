import cv2
import numpy as np
import torchvision.transforms as transforms
import torch

from PIL import Image

# Draw a prediction box with confidence and title
def draw_prediction(frame, classes, classId, conf, left, top, right, bottom):

    # Draw a bounding box.
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0))

    # Assign confidence to label
    label = '%.2f' % conf

    # Print a label of class.
    if classes:
        assert(classId < len(classes))
        label = '%s: %s' % (classes[classId], label)

    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv2.rectangle(frame, (left, top - labelSize[1]), (left + labelSize[0], top + baseLine), (255, 255, 255), cv2.FILLED)
    cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

# Process frame, eliminating boxes with low confidence scores and applying non-max suppression
def process_frame(frame, outs, classes, confThreshold, nmsThreshold):
    # Get the width and height of the image
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    # Network produces output blob with a shape NxC where N is a number of
    # detected objects and C is a number of classes + 4 where the first 4
    # numbers are [center_x, center_y, width, height]
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                # Scale the detected coordinates back to the frame's original width and height
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                # Save the classId, confidence and bounding box for later use
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Apply non-max suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        draw_prediction(frame, classes, classIds[i], confidences[i], left, top, left + width, top + height)
        
def log(*msg):
    print("="*50)
    print(*msg)
    
    
class ImageLoader:
    
    def __init__(self, device, imsize=512):
        self.device = device
        self.loader = transforms.Compose([
            transforms.Resize(imsize),  # scale imported image
            transforms.ToTensor()])  # transform it into a torch tensor
    
    def load2(self, image_path: str) -> torch.Tensor:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.loader(image).unsqueeze(0)
        return image

    def load(self, image_path: str) -> torch.Tensor:
        image = Image.open(image_path)
        image = self.loader(image).unsqueeze(0)
        return image.to(self.device, torch.float)