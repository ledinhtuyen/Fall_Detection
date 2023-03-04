import cv2
import mediapipe as mp
import torch
from train import model
from utils import softmax

mp_pose = mp.solutions.pose

mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(static_image_mode=True,
                    smooth_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                    )


mp_pose = mp.solutions.pose



def predict(img,model):
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_RGB)
    pos = []
    for i in range(33):
        cx = results.pose_landmarks.landmark[i].x
        cy = results.pose_landmarks.landmark[i].y
        cz = results.pose_landmarks.landmark[i].z
        pos.append(cx)
        pos.append(cy)
        pos.append(cz)
    pos_input = torch.tensor(pos)
    pos_input = pos_input.unsqueeze(0)
    output_result = model(pos_input)
    output_result = output_result.detach()
    softmax(output_result)

if __name__ == '__main__':
    model = torch.load("fall_vs_up.pth")
    img = cv2.imread('fall3.jpg')
    predict(img,model=model)
