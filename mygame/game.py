import cv2
import mediapipe as mp
import time
import numpy as np
import math
from bomb import Bomb
import random as rd

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

device = 1 # camera device number

def getFrameNumber(start:float, fps:int):
    now = time.perf_counter() - start
    frame_now = int(now * 1000 / fps)

    return frame_now

def centerOfGravity(landmark):
    n = len(landmark)
    x = 0
    y = 0
    for index, landmark in enumerate(landmark):
        x += landmark.x
        y += landmark.y
    return (x/n, y/n)

def drawPoint(image, point, index=''):
    # point is normalized(x, y)
    image_width = image.shape[1]
    image_height = image.shape[0]
    x = min(int(point[0] * image_width), image_width - 1)
    y = min(int(point[1] * image_height), image_height - 1)
    cv2.circle(image, (x, y), 3, (255, 0, 0), 1)
    # cv2.putText(image, '{}({}, {})'.format(index, x, y), (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0),1)
    cv2.putText(image, '{}'.format(index), (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0),1)

def calcAngle(p1, midpoint, p2, axis='x'):
    if axis == 'y':
        v1 = (p1[0]-midpoint[0], p1[1]-midpoint[1], 0.02)
        v2 = (p2[0]-midpoint[0], p2[1]-midpoint[1], 0)
    else:
        v1 = (p1[0]-midpoint[0], p1[1]-midpoint[1], 0)
        v2 = (p2[0]-midpoint[0], p2[1]-midpoint[1], 0)
    v1_n = np.linalg.norm(v1)
    v2_n = np.linalg.norm(v2)
    cos_theta = np.inner(v1, v2) / (v1_n * v2_n)
    return np.arccos(cos_theta)

def drawFace(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []

    for index, landmark in enumerate(landmarks.landmark):
        if landmark.visibility < 0 or landmark.presence < 0:
            continue
        
        # Convert the obtained landmark values x and y to the coordinates on the image
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y, landmark_z])

    if len(landmark_point) != 0:
        for i in range(0, len(landmark_point)):
            cv2.circle(image, (int(landmark_point[i][0]),int(landmark_point[i][1])), 1, (0, 255, 0), 1)

def drawLine(image, point1, point2, color=(255,0,0)):
    # point is normalized(x, y)
    image_width = image.shape[1]
    image_height = image.shape[0]
    x1 = min(int(point1[0] * image_width), image_width - 1)
    y1 = min(int(point1[1] * image_height), image_height - 1)
    x2 = min(int(point2[0] * image_width), image_width - 1)
    y2 = min(int(point2[1] * image_height), image_height - 1)
    cv2.line(image, (x1, y1), (x2, y2), color, 2)

def pos(key, landmark):
    dict = {
        'eye_left': [33, 159, 145, 133],
        'eye_right': [382, 386, 263, 374],
        'beye_left': [205, 206],
        'beye_right': [425, 426],
        'nose': [4],
        'forehead': [9],
        'chin': [152]
    }
    if len(dict[key]) > 1:
        temp = []
        temp = [landmark[index] for index in dict[key]]
        return centerOfGravity(temp)
    else:
        k = dict[key][0]
        return (landmark[k].x, landmark[k].y)

def normalize2pixel(point, image_width, image_height):
    x = min(int(point[0] * image_width), image_width - 1)
    y = min(int(point[1] * image_height), image_height - 1)
    if x < 0: x = 0
    if y < 0: y = 0
    return (int(x), int(y))

def formatAngleWithQ(angle, Q, axis='x'):
    format_angle = angle
    if axis == 'x':
        if Q == 2 or Q == 3:
            format_angle = np.pi - angle 
    if axis == 'y':
        if Q == 1 or Q == 2:
            format_angle = angle - np.pi/2
        elif Q == 3 or Q == 4:
            format_angle = np.pi/2 - angle
    return format_angle
    
def calculate_xy(angle, beam_size):
    # calculate the extend lenght which will be added to ray to project a beam
    x = np.cos(angle) * beam_size
    y = np.sin(angle) * beam_size
    return x, y

def findQuandrant(angle_x, angle_y):
    pi = np.pi
    if angle_x < pi/2 and angle_y < pi/2: return 4
    elif angle_x < pi/2 and angle_y > pi/2: return 1
    elif angle_x > pi/2 and angle_y < pi/2: return 3
    elif angle_x > pi/2 and angle_y > pi/2: return 2
    else: return 4

def find_endbeam(end_ray, xx, yx, xy, yy, Q):
    if Q == 1:
        yx = yx * -1; yy = yy * -1
    elif Q == 2:
        xx = xx * -1; xy = xy * -1
        yx = yx * -1; yy = yy * -1
    elif Q == 3:
        xx = xx * -1; xy = xy * -1

    end_beam = (end_ray[0]+int(xx)+int(xy), end_ray[1]+int(yx)+int(yy))
    return end_beam

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

def main():
    # For webcam input:
    bomb_init = False
    gameover = False
    global device
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    cap = cv2.VideoCapture(device)
    with mp_face_mesh.FaceMesh(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
        while cap.isOpened():
            ret, frame = cap.read()
            cv2.namedWindow('MediaPipe FaceMesh')
            if not ret:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue
            frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            frame.flags.writeable = False
            results = face_mesh.process(frame)
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if gameover:
                break

            if not bomb_init:
                bomb = Bomb(frame_width=frame.shape[1], frame_height=frame.shape[0])
                print('Bomb init...')
                for _ in range(4):
                    bomb.randomPosition()
                bomb_init = True

            if len(bomb.getPosition()) == 0:
                start = time.time()
                while True:
                    ret, frame = cap.read()
                    frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    now = time.time()-start
                        
                    cv2.putText(frame, 'CONGRATULATION!!!', (frame.shape[1]//2-420, frame.shape[0]//2-200), cv2.FONT_HERSHEY_COMPLEX, 3, (0,255,0), 3)

                    if now > 3:
                        gameover = True
                        break

                    cv2.imshow('Bomb Game', frame)
                    if cv2.waitKey(5) & 0xFF == 27:
                        break

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    
                    eye_left = pos('eye_left', face_landmarks.landmark)
                    eye_right = pos('eye_right', face_landmarks.landmark) 
                    center_eyes = ((eye_left[0]+eye_right[0])/2, (eye_left[1]+eye_right[1])/2)
                    
                    y1 = (center_eyes[0], center_eyes[1]-0.1)
                    y0 = (center_eyes[0], center_eyes[1]+0.1)

                    nose = pos('nose', face_landmarks.landmark)
                    forehead = pos('forehead', face_landmarks.landmark)
                    end_ray = ((nose[0]+forehead[0])/2, (nose[1]+forehead[1])/2)
                    
                    angle_x = calcAngle(end_ray, center_eyes, eye_right, axis='x')
                    angle_y = calcAngle(end_ray, center_eyes, y0, axis='y')

                    q = findQuandrant(angle_x, angle_y)

                    angle_x = formatAngleWithQ(angle_x, q, axis='x')
                    angle_y = formatAngleWithQ(angle_y, q, axis='y')

                    beam_size = 250
                    end_ray = normalize2pixel(end_ray, frame.shape[1], frame.shape[0])
                    # xx, yx, xy, yy -> Distance to needed to extend the ray
                    xx, yx = calculate_xy(angle_x, beam_size)
                    # print('X-value ({}, {})'.format(xx, yx))
                    xy, yy = calculate_xy(angle_y, beam_size)
                    # print('Y-value ({}, {})'.format(xy, yy))

                    end_beam = find_endbeam(end_ray, xx, yx, xy, yy, q)
                    center = normalize2pixel(center_eyes, frame.shape[1], frame.shape[0])
                    eye_left = normalize2pixel(eye_left, frame.shape[1], frame.shape[0])
                    eye_right = normalize2pixel(eye_right, frame.shape[1], frame.shape[0])

                    cv2.line(frame, eye_left, end_beam, (100,100,100), 1)
                    cv2.line(frame, eye_right, end_beam, (100,100,100), 1)

                    hit, center_bomb, bomb_index = bomb.isHit(eye_left, eye_right, end_beam, q)
                    if hit:
                        color1 = (rd.randint(0, 255), rd.randint(0, 255), rd.randint(0, 255))
                        color2 = (rd.randint(0, 255), rd.randint(0, 255), rd.randint(0, 255))
                        cv2.line(frame, eye_left, center_bomb, color1, 5)
                        cv2.line(frame, eye_right, center_bomb, color2, 5)
                        position = bomb.getPosition()
                        countdown = bomb.getCountdown()
                        countdown[bomb_index] -= 17
                        if countdown[bomb_index] <= 0:
                            del position[bomb_index]
                            del countdown[bomb_index]
                        bomb.setPositon(position)
                        bomb.setCountdown(countdown)

                    bomb.displayBomb(frame, bomb.getPosition())
                
            cv2.imshow('Bomb Game', frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break
        cap.release()

if __name__ == '__main__':
    main()
