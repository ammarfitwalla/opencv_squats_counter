import requests
import cv2
import numpy as np
import urllib3

urllib3.disable_warnings()
print("Warning: Certificates not verified!")

process_start = False
filename = 'faces.jpg'
video_file_name = ''
curr_mod = 'demo'
video_file = 0
start_counter = 10
process_stand = False
countdown_started = False
process_finish = False
raise_hand = True
total_time_remaining = 0
total_time = 120
total_squat = 0
start_ymin = 0
end_ymin = 0
list_miny = []
start_miny_list = []
xMid = 10
yMid = 10
CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
IMG_WIDTH = 416
IMG_HEIGHT = 416

COLOR_BLUE = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_YELLOW = (0, 255, 255)



def do_something(image):
    fontface = cv2.FONT_HERSHEY_SIMPLEX
    global process_start, fmin_X, fmin_Y, fmax_X, fmax_Y, hmaxX, hmaxY, hmin_X, hmin_Y
    global countdown_started
    global start_counter
    global total_time_remaining
    global xMid, yMid, raise_hand
    from random import randint
    # file_name = "Test" + str(randint(0, 1)) + ".jpg"
    # cv2.imwrite(file_name, image)
    if raise_hand:
        cv2.putText(image, "Raise your hand above your head", (20, 50), fontface, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # api_url_hands = "https://10.150.20.61/powerai-vision/api/dlapis/a322454a-330a-4190-b922-2f4551b5357d"

    fontface = cv2.FONT_HERSHEY_SIMPLEX
    hmin_X = 0
    hmin_Y = 0
    minX = 0
    minY = 0
    rc1 = 0
    rc11 = 0

    if not process_start:
        blob = cv2.dnn.blobFromImage(image, 1 / 255, (IMG_WIDTH, IMG_HEIGHT), [0, 0, 0], 1, crop=False)

        # Sets the input to the network
        net_hand.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = net_hand.forward(get_outputs_names(net_hand))

        # Remove the bounding boxes with low confidence
        hand = post_process(image, outs, CONF_THRESHOLD, NMS_THRESHOLD)
        # print('[i] ==> # detected faces: {}'.format(len(faces)))
        if len(hand) == 1:
            print("hand ", hand)
            hmin_X = hand[0][0]
            print('hmin_X', hmin_X)
            hmin_Y = hand[0][1]
            print('hmin_Y', hmin_Y)
            hmax_X = hand[0][2]
            print('hmax_X', hmax_X)
            hmax_Y = hand[0][3]
            print('hmax_Y', hmax_Y)

        blob = cv2.dnn.blobFromImage(image, 1 / 255, (IMG_WIDTH, IMG_HEIGHT), [0, 0, 0], 1, crop=False)

        # Sets the input to the network
        net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = net.forward(get_outputs_names(net))

        # Remove the bounding boxes with low confidence
        faces = post_process(image, outs, CONF_THRESHOLD, NMS_THRESHOLD)
        # print('[i] ==> # detected faces: {}'.format(len(faces)))

        if len(faces) == 1:
            print("faces ", faces)
            fmin_X = faces[0][0]
            print('fmin_X', fmin_X)
            fmin_Y = faces[0][1]
            print('fmin_Y', fmin_Y)
            fmax_X = faces[0][2]
            print('fmax_X', fmax_X)
            fmax_Y = faces[0][3]
            print('fmax_Y', fmax_Y)

            if fmin_X > hmin_X and fmin_Y > hmin_Y and fmin_X > 0 and fmin_Y > 0 and hmin_X > 0 and hmin_Y > 0:
                raise_hand = False
                # cv2.putText(image, "Hand is above head", (minX, minY), fontface, 1, (0, 255, 255), 1, cv2.LINE_AA)
                # cv2.rectangle(image, (minX, minY), (fmax_X, fmax_Y), (0, 255, 0), 2)
                # cv2.rectangle(image, (hminX, hminY), (hmaxX, hmaxY), (0, 255, 0), 2)
                process_start = True

    else:
        print('inside else')
        cv2.putText(image, "countdown begin in", (xMid - 270, yMid), fontface, 1.8, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.putText(image, str(start_counter), (xMid - 30, yMid + 50), fontface, 2, (0, 0, 255), 3, cv2.LINE_AA)
        if start_counter > 0:
            import time
            time.sleep(1)
            start_counter = start_counter - 1
            print("count down will start in " + str(start_counter))
        if start_counter == 0:
            countdown_started = True
            counter = 0
    return image


def draw_predict(frame, left, top, right, bottom):
    # Draw a bounding box.
    cv2.rectangle(frame, (left, top), (right, bottom), COLOR_YELLOW, 2)


def get_outputs_names(net):
    layers_names = net.getLayerNames()
    return [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]


def post_process(frame, outs, conf_threshold, nms_threshold):
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    confidences = []
    boxes = []
    final_boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                width = int(detection[2] * frame_width)
                height = int(detection[3] * frame_height)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        final_boxes.append(box)
        # print('boxes ', final_boxes)
        # left, top, right, bottom = refined_box(left, top, width, height)
        draw_predict(frame, left, top, left + width, top + height)
        # draw_predict(frame, confidences[i], left, top, right, bottom)
    return final_boxes


def check_squats(image):
    fontface = cv2.FONT_HERSHEY_SIMPLEX
    global countdown_started, start_counter, total_time_remaining, total_squat, start_ymin, end_ymin, process_start, total_time, list_miny, process_finish, xMid, yMid, process_stand, raise_hand, start_miny_list
    requests.packages.urllib3.disable_warnings()
    print(total_time_remaining)
    if total_squat >= 5 and total_time_remaining <= total_time:
        cv2.putText(image, "Congratulations", (xMid - 250, yMid), fontface, 2, (0, 0, 255), 2, cv2.LINE_AA)
        # print("congrats")
        raise_hand = True
        process_finish = True
        countdown_started = False
        process_start = False
        total_squat = 0
        start_counter = 5
        total_time_remaining = 0
        list_miny = []
        start_miny_list = []
    elif total_time_remaining > total_time:
        cv2.putText(image, "Keep trying you are the best", (xMid - 270, yMid), fontface, 1.2, (0, 0, 255), 3,
                    cv2.LINE_AA)
        process_finish = True
        raise_hand = True
        countdown_started = False
        total_squat = 0
        start_counter = 5
        process_start = False
        list_miny = []
        start_miny_list = []
        total_time_remaining = 0
    else:
        from random import randint
        # file_name = "Test" + str(randint(0, 1)) + ".jpg"
        # cv2.imwrite(file_name, image)
        # print('faces in check squats ', faces)
        # Create a 4D blob from a frame.
        blob = cv2.dnn.blobFromImage(image, 1 / 255, (IMG_WIDTH, IMG_HEIGHT),
                                     [0, 0, 0], 1, crop=False)

        # Sets the input to the network
        net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = net.forward(get_outputs_names(net))

        # Remove the bounding boxes with low confidence
        faces = post_process(image, outs, CONF_THRESHOLD, NMS_THRESHOLD)
        print('[i] ==> # detected faces: {}'.format(len(faces)))
        print('faces in check squats', faces)

        if len(faces) == 0:
            pass
            # check_squats(image)
        else:
            list_miny.append(faces[0][1])

        # print('list of min y ', list_miny)
        fontface = cv2.FONT_HERSHEY_SIMPLEX
        # minY = 0
        cv2.putText(image, "Total Squats: " + str(total_squat), (xMid - 310, 100), fontface, 1.5, (0, 0, 255), 2,
                    cv2.LINE_AA)

        if len(faces) > 0:
            minY = faces[0][1]
            if len(list_miny) > 0:
                start_ymin = list_miny[0]
                # print('start_ymin ', start_ymin)
                # print(minY)
                if minY < start_ymin + 100:
                    if process_stand:
                        if list_miny[0] - 50 < minY < list_miny[0] + 50:
                            total_squat += 1
                            end_ymin = 0
                            process_stand = False
                            start_ymin = 0
                            list_miny = []
                    else:
                        print("pass")
                else:
                    if minY > end_ymin + 1:
                        end_ymin = minY
                    else:
                        # print('process stand true')
                        process_stand = True

            # End Of Squats counting Logic

            total_time_remaining += 1
        # time.sleep(1)
        else:
            pass
    counter = 0
    return image


def show(image):
    global countdown_started, xMid, yMid
    global process_finish
    height, width, channel = image.shape
    xMid = width // 2
    yMid = height // 2
    if not countdown_started:
        image = do_something(image)
    else:
        if process_start:
            process_finish = False
            image = check_squats(image)
    image = cv2.resize(image, (800, 600))
    cv2.imshow('image', image)
    ret, jpeg = cv2.imencode('.jpg', image)
    return jpeg.tobytes()


camera_port = 0
# camera_port = 'rtsp://admin:maaz@123@192.168.0.115'
# camera_port = 'http://192.168.0.102:6677/videofeed?username=&password='
# camera_port = 'http://83.110.154.74:8060/videofeed?username=admin&password=9999'
# Loading the config and weights file
yolo_face_cfg = 'yolov3-face.cfg'
yolo_face_weights = 'yolov3-wider_16000.weights'

yolo_hand_cfg = 'yolov3-tiny.cfg'
yolo_hand_weights = 'yolov3-tiny_8000.weights'


net_hand = cv2.dnn.readNetFromDarknet(yolo_hand_cfg, yolo_hand_weights)
net_hand.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net_hand.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

net = cv2.dnn.readNetFromDarknet(yolo_face_cfg, yolo_face_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

video = cv2.VideoCapture(camera_port)

while True:
    ret, image = video.read()
    show(image)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
