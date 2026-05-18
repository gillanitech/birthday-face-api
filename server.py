from flask import Flask, request, jsonify
import face_recognition
import pickle
import numpy as np

app = Flask(__name__)

with open("encodings.pkl", "rb") as f:
    known_encodings = pickle.load(f)

if len(known_encodings) == 0:
    print("❌ No known encodings found! Run training again.")

@app.route("/")
def home():
    return "Birthday Face API Running 🎉"

@app.route("/check_face", methods=["POST"])
def check_face():
    try:
        file = request.files["image"]
        img = face_recognition.load_image_file(file)
        encodings = face_recognition.face_encodings(img)

        if len(encodings) == 0:
            return jsonify({"match": False, "reason": "No face detected in input image"})

        face = encodings[0]

        if len(known_encodings) == 0:
            return jsonify({"match": False, "reason": "No trained encodings found"})

        distances = face_recognition.face_distance(known_encodings, face)
        best_distance = np.min(distances)
        print("Best distance:", best_distance)

        if best_distance < 0.45:
            return jsonify({"match": True})
        else:
            return jsonify({"match": False})

    except Exception as e:
        return jsonify({"match": False, "error": str(e)})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
