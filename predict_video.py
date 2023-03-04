import time

import mediapipe as mp
import torch
from predict_img import softmax
from train import model
import cv2


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=True,
                    smooth_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)

model=torch.load("checkpoint/fall_vs_up.pth")

def predict(key_point_list):
    input_tensor=torch.tensor(key_point_list)
    input_tensor=input_tensor.unsqueeze(0)
    output_tensor=model(input_tensor)
    output_tensor=output_tensor.detach()
    return softmax(output_tensor)

def process_frame(img):
    text = ""
    start_time = time.time()

    h, w = img.shape[0], img.shape[1]

    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_RGB)

    if results.pose_landmarks:
        pos = []
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        for i in range(33):
            x = results.pose_landmarks.landmark[i].x
            y = results.pose_landmarks.landmark[i].y
            z = results.pose_landmarks.landmark[i].z
            cx = int(x * w)
            cy = int(y * h)
            # cz = z

            pos.append(x)
            pos.append(y)
            pos.append(z)

            radius = 10

            if i == 0:
                img = cv2.circle(img, (cx, cy), radius, (0, 0, 255), -1)
            elif i in [11, 12]:
                img = cv2.circle(img, (cx, cy), radius, (223, 155, 6), -1)
            elif i in [23, 24]:
                img = cv2.circle(img, (cx, cy), radius, (1, 240, 255), -1)
            elif i in [13, 14]:
                img = cv2.circle(img, (cx, cy), radius, (140, 47, 240), -1)
            elif i in [25, 26]:
                img = cv2.circle(img, (cx, cy), radius, (0, 0, 255), -1)
            elif i in [15, 16, 27, 28]:
                img = cv2.circle(img, (cx, cy), radius, (223, 155, 60), -1)

            #             elif i in [17,19,21]:
            #                 img = cv2.circle(img,(cx,cy), radius, (94,218,121), -1)
            #             elif i in [18,20,22]:
            #                 img = cv2.circle(img,(cx,cy), radius, (16,144,247), -1)

            elif i == 27:
                img = cv2.circle(img, (cx, cy), radius, (29, 123, 243), -1)
            elif i == 28:
                img = cv2.circle(img, (cx, cy), radius, (193, 182, 255), -1)
            elif i in [7, 8]:
                img = cv2.circle(img, (cx, cy), radius, (94, 218, 121), -1)
            else:
                #                 img = cv2.circle(img,(cx,cy), radius, (0,255,0), -1)
                pass
        posibiliy = predict(pos)
        #         if posibiliy>0.5:
        text = str(posibiliy)

    else:
        scaler = 1
        failure_str = 'No Person'
        img = cv2.putText(img, failure_str, (25 * scaler, 100 * scaler), cv2.FONT_HERSHEY_SIMPLEX, 1.25 * scaler,
                          (255, 0, 255), 2 * scaler)
        text = "safe"

    end_time = time.time()
    FPS = 1 / (end_time - start_time)
    scaler = 1
    #     img = cv2.putText(img, 'FPS  '+str(int(FPS)), (25 * scaler, 50 * scaler), cv2.FONT_HERSHEY_SIMPLEX, 1.25 * scaler, (255, 0, 255), 2 * scaler)
    img = cv2.putText(img, 'FPS  ' + str(int(FPS)) + '       fall_probability      ' + text, (25 * scaler, 50 * scaler),
                      cv2.FONT_HERSHEY_SIMPLEX, 1.25 * scaler, (255, 0, 255), 2 * scaler)
    return img


def predict_with_webcam(input_path):
    cap = cv2.VideoCapture(input_path)
    cv2.namedWindow('Webcam', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Webcam",cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        if not cap.isOpened():
            print("Không thể mở webcam")
            exit()
        try:
            success, frame = cap.read()
            if not success:
                break

            try:
                frame = cv2.resize(frame, (1280, 720))
                frame = process_frame(frame)
            except:
                print('error')
                pass

            if success == True:
                cv2.imshow('Webcam', frame)

            if cv2.waitKey(1) == ord('q'):
                break
        except:
            pass

    cv2.destroyAllWindows()
    cap.release()

if __name__ == '__main__':
    predict_with_webcam(input_path=0)