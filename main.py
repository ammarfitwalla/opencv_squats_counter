import os
import time

import cv2
import numpy as np


class YoloModel:
    def __init__(self, cfg_file_path, weights_file_path, confidence_threshold=0.5, nms_threshold=0.4):
        self.cfg_file_path = cfg_file_path
        self.weights_file_path = weights_file_path
        self.net = self.load_model()
        self.height = 416
        self.width = 416
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold

    def detect_objects(self, image):
        blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255,
                                     mean=[0, 0, 0], swapRB=True,
                                     crop=False, size=(self.height, self.width))
        self.net.setInput(blob)
        outs = self.net.forward(self.get_layers_name())
        objects, image = self.post_process(image, outs)
        return objects, image

    def post_process(self, frame, outs):
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]

        boxes, confidences = [], []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > self.confidence_threshold:
                    center_x = int(detection[0] * frame_width)
                    center_y = int(detection[1] * frame_height)
                    width = int(detection[2] * frame_width)
                    height = int(detection[3] * frame_height)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)

        final_boxes = []
        for i in indices:
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            right = left + width
            bottom = top + height
            final_boxes.append(box)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 2)

        return final_boxes, frame

    def get_layers_name(self):
        layers_name = self.net.getLayerNames()
        return [layers_name[i - 1] for i in self.net.getUnconnectedOutLayers()]

    def load_model(self):
        net = cv2.dnn.readNetFromDarknet(self.cfg_file_path, self.weights_file_path)
        return net


class Squats:
    def __init__(self):
        self.camera_port = 1
        # Face model
        self.face_yolo_model_dir = r'yolo_models/face_model'
        self.face_model_cfg_path = os.path.join(self.face_yolo_model_dir, 'yolov3-face.cfg')
        self.face_model_weights_path = os.path.join(self.face_yolo_model_dir, 'yolov3-wider_16000.weights')
        self.face_yolo_model = YoloModel(self.face_model_cfg_path, self.face_model_weights_path)

        # Hand model
        self.hand_yolo_model_dir = r'yolo_models/hand_model'
        self.hand_model_cfg_path = os.path.join(self.hand_yolo_model_dir, 'yolov3-tiny.cfg')
        self.hand_model_weights_path = os.path.join(self.hand_yolo_model_dir, 'yolov3-tiny_8000.weights')
        self.hand_yolo_model = YoloModel(self.hand_model_cfg_path, self.hand_model_weights_path)

        self.countdown = False
        self.countdown_time = 10
        self.start_time = time.time()

        self.image_height = 416
        self.image_width = 416
        self.mid_x = self.image_width // 2
        self.mid_y = self.image_height // 2

        self.squat_counting_process = False
        self.total_squats_to_be_done = 5
        self.total_squats_completed = 0

        self.miny_list = []
        self.min_y = 0
        self.start_min_y = 0
        self.end_min_y = 0
        self.process_stand = False

        self.exercise_start_time = time.time()
        self.total_time = 60
        self.time_remaining = self.total_time

        self.display_image_height = 750
        self.display_image_width = 1000

    def capture_and_show(self):
        video = cv2.VideoCapture(self.camera_port)

        try:
            if not video.isOpened():
                raise Exception("Error opening the camera!")

            while True:
                ret, image = video.read()

                if not ret:
                    break

                if self.squat_counting_process:
                    image = self.check_squats(image)
                if not self.countdown and not self.squat_counting_process:
                    face_coordinates, image = self.get_faces(image)
                    hand_coordinates, image = self.get_hands(image)
                    cv2.putText(image, "Raise hand above head: ", (30, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    if face_coordinates and hand_coordinates and not self.countdown:
                        raised_hands = self.check_hand_above_head(face_coordinates, hand_coordinates)
                        if raised_hands:
                            self.countdown = True
                            self.start_time = time.time()

                if self.countdown:
                    elapsed_time = time.time() - self.start_time
                    if elapsed_time >= 1:
                        self.countdown_time -= 1
                        self.start_time = time.time()

                        if self.countdown_time < 1:
                            self.countdown = False
                            self.countdown_time = 10
                            self.squat_counting_process = True
                            self.exercise_start_time = time.time()
                            continue

                    cv2.putText(image, "Countdown: " + str(self.countdown_time), (self.mid_x - 100, self.mid_y,),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                image = cv2.resize(image, (self.display_image_width, self.display_image_height))
                cv2.imshow("image", image)
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
        finally:
            video.release()
            cv2.destroyAllWindows()

    def check_squats(self, image):
        self.update_time()
        if self.total_squats_completed < self.total_squats_to_be_done and self.time_remaining > 0:
            image = self.count_squat(image)
        if self.time_remaining == 0:
            image = self.handle_try_again(image)
        if self.total_squats_completed == self.total_squats_to_be_done and self.time_remaining > 0:
            image = self.handle_squats_completed(image)
        return image

    def reset_counters(self):
        self.countdown = False
        self.total_squats_completed = 0
        self.countdown_time = 10
        self.squat_counting_process = False
        self.miny_list = []
        self.time_remaining = 0

    def update_time(self):
        current_time = time.time()
        elapsed_time = current_time - self.exercise_start_time
        self.time_remaining = int(max(self.total_time - elapsed_time, 0))

    def handle_try_again(self, image):
        cv2.putText(image, "Try Again!", (self.mid_x - 100, self.mid_y), cv2.FONT_HERSHEY_SIMPLEX, 2,
                    (0, 0, 255), 2)
        image = cv2.resize(image, (self.display_image_width, self.display_image_height))
        cv2.imshow("image", image)
        cv2.waitKey(3000)
        self.reset_counters()
        return image

    def handle_squats_completed(self, image):
        cv2.putText(image, "Congratulations!", (self.mid_x - 100, self.mid_y), cv2.FONT_HERSHEY_SIMPLEX, 2,
                    (0, 0, 255), 2)
        image = cv2.resize(image, (self.display_image_width, self.display_image_height))
        cv2.imshow("image", image)
        cv2.waitKey(3000)
        self.reset_counters()
        return image

    def count_squat(self, image):
        face_coordinates, image = self.get_faces(image)
        if face_coordinates:
            self.update_miny_value(face_coordinates)
            self.draw_squat_count(image)
            self.update_squat_count()

        return image

    def update_squat_count(self):
        if len(self.miny_list) > 0:
            self.start_min_y = self.miny_list[0]
            if self.min_y < self.start_min_y + 100:
                if self.process_stand:
                    if self.start_min_y - 50 < self.min_y < self.start_min_y + 50:
                        self.total_squats_completed += 1
                        self.end_min_y = 0
                        self.process_stand = False
                        self.start_min_y = 0
                        self.miny_list = []
                else:
                    pass
            else:
                if self.min_y > self.end_min_y + 1:
                    self.end_min_y = self.min_y
                else:
                    self.process_stand = True

    def update_miny_value(self, face_coordinates):
        self.miny_list.append(face_coordinates[1])
        self.min_y = face_coordinates[1]

    def draw_squat_count(self, image):
        cv2.putText(image, "Total Count: " + str(self.total_squats_completed), (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.5, (0, 0, 255), 2)
        cv2.putText(image, "Time Left: " + str(self.time_remaining), (30, 120), cv2.FONT_HERSHEY_SIMPLEX,
                    1.5, (0, 0, 255), 2)

    @staticmethod
    def check_hand_above_head(face_coordinates, hand_coordinates):
        (face_min_x, face_min_y, face_max_x, face_max_y) = face_coordinates
        (hand_min_x, hand_min_y, hand_max_x, hand_max_y) = hand_coordinates

        if face_min_y > hand_min_y:
            return True

        return False

    def get_faces(self, image):
        faces, image = self.face_yolo_model.detect_objects(image)
        if len(faces) == 1:
            face_min_x = faces[0][0]
            face_min_y = faces[0][1]
            face_max_x = faces[0][2]
            face_max_y = faces[0][3]
            return (face_min_x, face_min_y, face_max_x, face_max_y), image

        return None, image

    def get_hands(self, image):
        hands, image = self.hand_yolo_model.detect_objects(image)
        if len(hands) == 1:
            hand_min_x = hands[0][0]
            hand_min_y = hands[0][1]
            hand_max_x = hands[0][2]
            hand_max_y = hands[0][3]
            return (hand_min_x, hand_min_y, hand_max_x, hand_max_y), image

        return None, image


if __name__ == '__main__':
    obj_squats = Squats()
    obj_squats.capture_and_show()
