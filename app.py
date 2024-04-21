import cv2
import mediapipe as mp
import joblib
import numpy as np
from flask import Flask, render_template, Response

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


@app.route('/video_feed')
def video_feed():
    return Response(recognize_sign_language(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
