from calendar import c
import cv2
import mediapipe as mp
import joblib
import numpy as np
from flask import Flask, render_template, Response
import json
import random
import time
import io
import datetime

# Hàm ghi điểm và mốc thời gian vào tệp scores.txt


def write_score_to_file(score):
    # Định dạng ngày giờ hiện tại
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Dòng dữ liệu để ghi vào tệp
    line = f"{current_time}, {score}\n"

    # Đường dẫn tệp scores.txt
    scores_file_path = 'scores.txt'

    # Mở tệp scores.txt để ghi vào cuối tệp
    with open(scores_file_path, 'a') as file:
        file.write(line)


app = Flask(__name__)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


def data_clean(landmark):

    data = landmark[0]

    try:
        data = str(data)

        data = data.strip().split('\n')

        garbage = ['landmark {', '  visibility: 0.0', '  presence: 0.0', '}']

        without_garbage = []

        for i in data:
            if i not in garbage:
                without_garbage.append(i)

        clean = []

        for i in without_garbage:
            i = i.strip()
            clean.append(i[2:])

        for i in range(0, len(clean)):
            clean[i] = float(clean[i])

        return ([clean])

    except:
        return (np.zeros([1, 63], dtype=int)[0])


# Function to calculate the Euclidean distance between two points
def calculate_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

# Function to check hand stability based on landmark movement


def check_hand_stability(landmarks_history):
    # Check if the Euclidean distances between consecutive landmark positions are below a threshold
    threshold = 10.0
    for i in range(len(landmarks_history[0]) - 1):
        distance = calculate_distance(
            landmarks_history[0][i], landmarks_history[0][i + 1])
        if distance > threshold:
            return False
    return True


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/model')
def model():
    return render_template('model.html')


def recognize_sign_language():
    hands = mp_hands.Hands(min_detection_confidence=0.7,
                           min_tracking_confidence=0.5)
    clf = joblib.load('E:\CourseHKII_Grade3\seminar\model_svm.pkl')
    cap = cv2.VideoCapture(0)
    previous_prediction = None
    predicted_string = ''
    landmarks_history = []

    while cap.isOpened():
        success, image = cap.read()
        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if not success:
            break

        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                if results.multi_handedness:
                    handedness = results.multi_handedness[idx].classification[0].label
                    if handedness == 'Right':
                        mp_drawing.draw_landmarks(
                            image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                        cleaned_landmark = data_clean(
                            results.multi_hand_landmarks)

                        if cleaned_landmark:
                            landmarks_history.append(cleaned_landmark)
                            if len(landmarks_history) > 15:
                                y_pred = clf.predict(cleaned_landmark)
                                landmarks_history = []
                                if y_pred[0] == 'del':
                                    predicted_string = ''

                                elif y_pred[0] == 'space':
                                    predicted_string += ' '
                                else:
                                    predicted_string += y_pred[0]

                            (text_width, text_height), _ = cv2.getTextSize(
                                predicted_string[-11:], cv2.FONT_HERSHEY_SIMPLEX, 1, 2)

                            x1 = 50
                            y1 = 50
                            x2 = x1 + text_width + 20
                            y2 = y1 + text_height + 20

                            if len(predicted_string) > 0:
                                cv2.rectangle(image, (x1, y1),
                                              (x2, y2), (255, 255, 255), 2)

                            cv2.putText(
                                image, predicted_string[-11:], (x1 +
                                                                10, y1 + text_height + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (
                                    0, 0, 255), 2, cv2.LINE_AA
                            )

        ret, jpeg = cv2.imencode('.jpg', image)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


# Mãng chứa các ký tự có thể xuất hiện trong trò chơi
characters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
              'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Hàm nhận dạng ngôn ngữ ký hiệu trong trò chơi


def recognize_sign_language_game():
    hands = mp_hands.Hands(min_detection_confidence=0.7,
                           min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(0)
    clf = joblib.load('E:\CourseHKII_Grade3\seminar\model_svm.pkl')

    # Chọn ngẫu nhiên một ký tự từ danh sách
    current_character = random.choice(characters)
    score = 0
    start_time = time.time()

    while cap.isOpened():

        success, image = cap.read()
        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
       # Vẽ ký tự yêu cầu và thời gian còn lại lên hình ảnh
        cv2.putText(image, f"Character: {current_character}", (
            20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        time_remaining = max(0, 10 - (time.time() - start_time))
        cv2.putText(image, f"Time Remaining: {int(time_remaining)} seconds", (
            20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Vẽ điểm số lên hình ảnh
        cv2.putText(image, f"Score: {score}", (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        if not success:
            break
        if time_remaining <= 0:  # Kiểm tra thời gian
            # Thay image bằng nền màu tím
            image[:, :] = [255, 127, 127]
            write_score_to_file(score)
            cv2.putText(image, "Game Over! Your final score is " + str(
                score), (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cap.release()
            cv2.destroyAllWindows()

        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Xử lý landmarks và kiểm tra ký hiệu
                cleaned_landmark = data_clean([hand_landmarks])
                if cleaned_landmark:
                    y_pred = clf.predict(cleaned_landmark)
                    predicted_character = y_pred[0]
                    if time_remaining <= 0:  # Kiểm tra thời gian
                        # Thay image bằng nền màu tím
                        image[:, :] = [255, 127, 127]
                        write_score_to_file(score)
                        cv2.putText(image, "Game Over! Your final score is " + str(
                            score), (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                        cap.release()
                        cv2.destroyAllWindows()
                    if predicted_character == current_character:
                        score += 1
                        current_character = random.choice(
                            characters)  # Chọn ký tự mới
                        start_time = time.time()  # Reset thời gian

        ret, jpeg = cv2.imencode('.jpg', image)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    cap.release()
    cv2.destroyAllWindows()


@app.route('/game')
def game():
    # Read scores from the text file
    scores_file_path = 'scores.txt'

    scores = []

    # Đọc từng dòng từ tệp scores.txt và lưu vào danh sách scores
    with open(scores_file_path, 'r') as file:
        for line in file:
            if line.strip():  # Kiểm tra xem dòng có dữ liệu không trống
                timestamp, score = line.strip().split(',')
                score = int(score)
                scores.append((timestamp, score))

    # Sắp xếp điểm số theo thứ tự giảm dần và chọn 5 điểm cao nhất
    top_scores = sorted(scores, key=lambda x: x[1], reverse=True)[:5]

    # Render the game.html template and pass the scores as a variable
    return render_template('game.html', scores=top_scores)


@app.route('/game_feed')
def game_feed():
    return Response(recognize_sign_language_game(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed')
def video_feed():
    return Response(recognize_sign_language(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
