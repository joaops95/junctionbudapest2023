from facenet_pytorch import MTCNN
from PIL import Image
from matplotlib import pyplot  as plt
import numpy as np
import math
import requests
import argparse
import torch
import cv2
import time
import sys
parser = argparse.ArgumentParser("Face pose detection for one face")

parser.add_argument("--path", help="To use video path.", type=str)
parser.add_argument("--record_path", help="To use video path.", type=str)
args = parser.parse_args()

path = args.path
record_path = args.record_path
left_offset = 20
fontScale = 2
fontThickness = 3
text_color = (0,0,255)
lineColor = (255, 255, 0)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Running on device: {device}')

mtcnn = MTCNN(image_size=160,
              margin=0,
              min_face_size=20,
              thresholds=[0.6, 0.7, 0.7], # MTCNN thresholds
              factor=0.709,
              post_process=True,
              device=device # If you don't have GPU
        )


def visualize(image, landmarks_, angle_R_, angle_L_, pred_):
    fig , ax = plt.subplots(1, 1, figsize= (8,8))
    
    leftCount = len([i for i in pred_ if i == 'Left Profile'])
    rightCount = len([i for i in pred_ if i == 'Right Profile'])
    frontalCount = len([i for i in pred_ if i == 'Frontal'])
    facesCount = len(pred_) # Number of detected faces (above the threshold)
    ax.set_title(f"Number of detected faces = {facesCount} \n frontal = {frontalCount}, left = {leftCount}, right = {rightCount}")
    for landmarks, angle_R, angle_L, pred in zip(landmarks_, angle_R_, angle_L_, pred_):
        
        if pred == 'Frontal':
            color = 'white'
        elif pred == 'Right Profile':
            color = 'blue'
        else:
            color = 'red'
            
        point1 = [landmarks[0][0], landmarks[1][0]]
        point2 = [landmarks[0][1], landmarks[1][1]]

        point3 = [landmarks[2][0], landmarks[0][0]]
        point4 = [landmarks[2][1], landmarks[0][1]]

        point5 = [landmarks[2][0], landmarks[1][0]]
        point6 = [landmarks[2][1], landmarks[1][1]]
        for land in landmarks:
            ax.scatter(land[0], land[1])
        plt.plot(point1, point2, 'y', linewidth=3)
        plt.plot(point3, point4, 'y', linewidth=3)
        plt.plot(point5, point6, 'y', linewidth=3)
        plt.text(point1[0], point2[0], f"{pred} \n {math.floor(angle_L)}, {math.floor(angle_R)}", 
                size=20, ha="center", va="center", color=color)
        ax.imshow(image)
        fig.savefig('Output_detection.jpg')
    return print('Done detect')

def visualizeCV2(frame, landmarks_, angle_R_, angle_L_, pred_):
    
    for landmarks, angle_R, angle_L, pred in zip(landmarks_, angle_R_, angle_L_, pred_):
                
        if pred == 'Frontal':
            color = (50, 50, 50)  # Dark gray
        elif pred == 'Right Profile':
            color = (200, 0, 0)  # Dark red
        elif pred == 'Left Profile':
            color = (0, 200, 0)  # Dark green
        elif pred == 'Diagonal Right':
            color = (200, 200, 0)  # Olive
        elif pred == 'Diagonal Left':
            color = (200, 0, 200)  # Dark magenta
        else:
            color = (0, 0, 200)  # Navy blue


        deepfake_color = (0, 0, 255)
        deepfake_text = "Alert !!! Deep fake detected"
        if pred == 'Frontal':
            deepfake_color = (0, 255, 0)  # Green for No deep fake detected
            deepfake_text = "Looking good..."


        point1 = [int(landmarks[0][0]), int(landmarks[1][0])]
        point2 = [int(landmarks[0][1]), int(landmarks[1][1])]

        point3 = [int(landmarks[2][0]), int(landmarks[0][0])]
        point4 = [int(landmarks[2][1]), int(landmarks[0][1])]

        point5 = [int(landmarks[2][0]), int(landmarks[1][0])]
        point6 = [int(landmarks[2][1]), int(landmarks[1][1])]

        for land in landmarks:
            cv2.circle(frame, (int(land[0]), int(land[1])), radius=5, color=(0, 255, 255), thickness=-1)
        cv2.line(frame, (int(landmarks[0][0]), int(landmarks[0][1])), (int(landmarks[1][0]), int(landmarks[1][1])), lineColor, 3)
        cv2.line(frame, (int(landmarks[0][0]), int(landmarks[0][1])), (int(landmarks[2][0]), int(landmarks[2][1])), lineColor, 3)
        cv2.line(frame, (int(landmarks[1][0]), int(landmarks[1][1])), (int(landmarks[2][0]), int(landmarks[2][1])), lineColor, 3)
        
        text_sizeR, _ = cv2.getTextSize(pred, cv2.FONT_HERSHEY_PLAIN, fontScale, 4)
        text_wR, text_hR = text_sizeR
        
        cv2.putText(frame, pred, (point1[0] + 10, point2[0]), cv2.FONT_HERSHEY_PLAIN, fontScale, color, fontThickness, cv2.LINE_AA)


        text_position = (10, 30)  # Example position, adjust as needed

        # Draw the text on the frame
        cv2.putText(frame, deepfake_text, text_position, cv2.FONT_HERSHEY_PLAIN, fontScale, deepfake_color, fontThickness, cv2.LINE_AA)

# Landmarks: [Left Eye], [Right eye], [nose], [left mouth], [right mouth]

def npAngle(a, b, c):
    ba = a - b
    bc = c - b 

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    
    return np.degrees(angle)

def corssedNpAngle(a, b, c):
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    # Calculate the cross product (in 2D)
    cross_product = ba[0]*bc[1] - ba[1]*bc[0]

    # Determine the sign of the angle
    angle = np.degrees(angle) * np.sign(cross_product)
    return angle

def npAngle2(a, b):
    cosine_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    angle = np.arccos(cosine_angle)

    # Convert to degrees
    angle = np.degrees(angle)
    return angle


def predFacePose(frame):
    bbox_, prob_, landmarks_ = mtcnn.detect(frame, landmarks=True)

    angle_R_List = []
    angle_L_List = []
    predLabelList = []
    
    try:

        for bbox, landmarks, prob in zip(bbox_, landmarks_, prob_):
            if bbox is not None:
                if prob > 0.9:
                    # Angles for frontal and profile detection
                    angR = npAngle(landmarks[0], landmarks[1], landmarks[2])
                    angL = npAngle(landmarks[1], landmarks[0], landmarks[2])

                    eye_vector = landmarks[1] - landmarks[0]  # Right eye - Left eye
                    midpoint_of_eyes = (landmarks[0] + landmarks[1]) / 2
                    nose_vector = landmarks[2] - midpoint_of_eyes  # Nose - Midpoint of eyes

                    # Vertical axis vector
                    vertical_axis = np.array([0, -1])

                    # Angle relative to the vertical axis
                    ang = npAngle2(midpoint_of_eyes, vertical_axis)

                    # Calculate the cross product for direction (right or left tilt)
                    cross_product = np.cross(midpoint_of_eyes, eye_vector)
                    if cross_product > 0:
                        ang = -ang  # Left tilt

                    # print(f"Angle: {ang}")

                    angle_R_List.append(angR)
                    angle_L_List.append(angL)

                    # Pose determination logic
                    # Pose determination logic
                    if ((int(angR) in range(35, 57)) and (int(angL) in range(35, 58)) and (int(ang) in range(115, 130))):
                        predLabel = 'Frontal'
                    elif ((int(angR) in range(35, 57)) and (int(angL) in range(35, 58)) and int(ang) in range(100, 119)):
                        predLabel = 'Diagonal Left'
                    elif ((int(angR) in range(35, 57)) and (int(angL) in range(35, 58)) and int(ang) in range(130, 170)):
                        predLabel = 'Diagonal Right'
                    elif angR < angL:
                        predLabel = 'Left Profile'
                    elif angL < angR:
                        predLabel = 'Right Profile'
                    else:
                        predLabel = 'Uncertain'

                    print(f"PREDICTED LABEL: {predLabel}")

                    predLabelList.append(predLabel)
                else:
                    print('The detected face is less than the detection threshold')
            else:
                print('No face detected in the image')

        return landmarks_, angle_R_List, angle_L_List, predLabelList
    except Exception as e:
        print(e)
        print('No face detected in the image')
        return [], [], [], []


def main(video_source=0):
    # Use command-line argument as video source if provided

    # Initialize video capture with the provided source
    video = cv2.VideoCapture(video_source)

    if not video.isOpened():
        print("Could not open video source:", video_source)
        return
    
    if record_path:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(record_path, fourcc, 20.0, (640, 480))

    while True:
        ret, frame = video.read()
        if ret:
            landmarks_, angle_R_List, angle_L_List, predLabelList = predFacePose(frame)
            visualizeCV2(frame, landmarks_, angle_R_List, angle_L_List, predLabelList)
            if record_path:
                out.write(frame)

            cv2.imshow("Output", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    video.release()
    if record_path:
        out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    if path:
        main(path)
    else:
        main()