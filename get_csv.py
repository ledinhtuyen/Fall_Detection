import csv
import os
import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=True,
                    smooth_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                    )


def predict(img):
    # transfer BGR to RGB
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
    return pos


def save_array_to_csv(data, output_folder="data", name='ceshi'):
    with open(output_folder + "/" + name + ".csv", 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        for row in data:
            writer.writerow(row)


def get_pos_list(path):
    '''
    :param path: the folder of imgs
    :return: A 2 dimensional array [N,99] of key points. N is the size of imgs and the 99 is all the x,y,z coordinates of 33 key points in mediapipe.
    '''
    array_of_pos = []
    for filename in os.listdir(path):
        file_dir = path + "\\" + filename
        img = cv2.imread(file_dir)
        array_of_pos.append(predict(img))  # predict(img)
    return array_of_pos


def get_csv_data(path, name):
    """
    :param path: the imgs folder contains a certain position of "up" or "down"
    :param name: the class name
    :return: a csv of [N,99]
    """
    save_array_to_csv(data=get_pos_list(path), name=name)


if __name__ == '__main__':
    get_csv_data("F:\\programs\\data\\IMG\\falling_ustb\\fall", name="fall")
