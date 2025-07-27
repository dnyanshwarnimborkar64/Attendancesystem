import sqlite3
import cv2
import numpy as np
from facenet_pytorch import InceptionResnetV1
import torch
from scipy.spatial.distance import cosine
from PIL import Image
from datetime import datetime
from flask import Flask, render_template, Response, request, jsonify, redirect, url_for
import time
import threading
from pushbullet import Pushbullet
from ultralytics import YOLO

app = Flask(__name__)

# Database Setup
conn = sqlite3.connect('attendance.db', check_same_thread=False)
c = conn.cursor()

# Initialize Database
def create_tables():
    global class_schedule
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            embedding BLOB NOT NULL
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            class_name TEXT NOT NULL,
            entry_time DATETIME,
            exit_time DATETIME,
            duration_minutes INTEGER,
            attended_classes INTEGER
        )
    ''')
    conn.commit()

    # Store class schedule in the database instead of a global variable
    c.execute('''
        CREATE TABLE IF NOT EXISTS class_schedule (
            class_name TEXT PRIMARY KEY,
            duration INTEGER
        )
    ''')
    conn.commit()

    # Load class schedule from the database
    c.execute('SELECT * FROM class_schedule')
    schedule_data = c.fetchall()
    
    class_schedule = {row[0]: row[1] for row in schedule_data}
    print("Loaded class schedule from DB:", class_schedule)

create_tables()

# Load FaceNet model for face recognition
model = InceptionResnetV1(pretrained='vggface2').eval()

# Load YOLO model for mobile detection
yolo_model = YOLO("yolov8n.pt")

# Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Pushbullet API setup
PUSHBULLET_API_KEY = "o.GacO5UC3X2yyBaHTKl7EeWwerZnDYi64"
pb = Pushbullet(PUSHBULLET_API_KEY)

# Global variables
class_schedule = {}
last_recognition_times = {}
last_alert_time = 0
ALERT_COOLDOWN = 10

@app.route('/')
def home():
    return redirect(url_for('register_page'))

@app.route('/register')
def register_page():
    return render_template('register.html')

@app.route('/configure')
def configure_page():
    return render_template('configure.html')

@app.route('/attendance')
def attendance_page():
    return render_template('attendance.html')

@app.route('/register', methods=['POST'])
def register():
    try:
        print("Received registration request:", request.form)
        name = request.form['name']
        uploaded_file = request.files['photo']
        
        if not uploaded_file:
            return jsonify({'error': "No file uploaded."})

        image = Image.open(uploaded_file).convert('RGB')
        image_np = np.array(image)
        resized_face = cv2.resize(image_np, (160, 160))
        embedding = get_face_embedding(resized_face)
        store_face_embedding(name, embedding)

        return jsonify({'message': f"Registered {name} successfully!", 'redirect': url_for('configure_page')})

    except Exception as e:
        return jsonify({'error': f"Registration failed. Error: {str(e)}"})

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def detect_faces(img):
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
    return faces

def get_face_embedding(face_img):
    face_img = np.array(face_img) / 255.0
    face_tensor = torch.tensor(face_img).permute(2, 0, 1).unsqueeze(0).float()

    with torch.no_grad():
        embedding = model(face_tensor)

    print("Generated embedding:", embedding.numpy().shape)
    return embedding.detach().numpy()

def store_face_embedding(name, embedding):
    embedding = embedding.flatten()
    embedding_blob = embedding.tobytes()
    c.execute('INSERT INTO users (name, embedding) VALUES (?, ?)', (name, embedding_blob))
    conn.commit()

def retrieve_face_embeddings():
    c.execute('SELECT * FROM users')
    users = c.fetchall()
    return [(user[1], np.frombuffer(user[2], dtype=np.float32).flatten()) for user in users]

def match_face(embedding, threshold=0.6):
    users = retrieve_face_embeddings()

    min_distance = float('inf')
    match_name = None

    print("🔍 Registered Users:", [user[0] for user in users])

    for name, stored_embedding in users:
        distance = cosine(embedding.flatten(), stored_embedding)
        print(f"🔎 Comparing with {name}: Distance = {distance}")

        if distance < min_distance:
            min_distance = distance
            match_name = name

    print(f"🟢 Best match: {match_name}, Distance: {min_distance}")

    if min_distance >= threshold:
        print("🚫 No valid match found! Face is unknown.")
        return None, min_distance

    return match_name, min_distance

def mark_entry_exit_with_auto_classes(name):
    global class_schedule
    conn = sqlite3.connect('attendance.db', check_same_thread=False)
    c = conn.cursor()

    c.execute('SELECT * FROM class_schedule ORDER BY duration ASC')
    schedule_data = c.fetchall()
    class_schedule = {row[0]: row[1] for row in schedule_data}

    print(f"Checking attendance for {name}...")
    print(f"Current class schedule: {class_schedule}")

    if not class_schedule:
        print("Error: Class schedule is empty!")
        conn.close()
        return "Class schedule is empty!"

    c.execute('SELECT * FROM attendance WHERE name=? AND exit_time IS NULL', (name,))
    record = c.fetchone()
    current_time = datetime.now()

    if record:
        entry_time_str = record[3]
        entry_time = datetime.strptime(entry_time_str, '%Y-%m-%d %H:%M:%S.%f')
        total_duration = (current_time - entry_time).total_seconds() / 60

        attended_classes = 0
        time_spent = 0

        for class_name, class_duration in class_schedule.items():
            if time_spent + class_duration <= total_duration:
                attended_classes += 1
                time_spent += class_duration
            else:
                break

        print(f"Updating exit for {name}: Duration: {int(total_duration)} minutes, Attended Classes: {attended_classes}")

        c.execute('''
            UPDATE attendance 
            SET exit_time=?, duration_minutes=?, attended_classes=? 
            WHERE id=?
        ''', (current_time, int(total_duration), attended_classes, record[0]))

        conn.commit()
        conn.close()
        return f"Marked exit for {name}. Duration: {int(total_duration)} minutes. Classes attended: {attended_classes}."

    else:
        print(f"{name} is not in attendance list, marking entry...")

        for class_name in class_schedule.keys():
            c.execute('INSERT INTO attendance (name, class_name, entry_time) VALUES (?, ?, ?)', (name, class_name, current_time))

        conn.commit()
        conn.close()
        return f"Marked entry for {name}."

@app.route('/configure_classes', methods=['POST'])
def configure_classes():
    try:
        global class_schedule
        class_count = request.form.get('class_count')

        if not class_count or not class_count.isdigit():
            return jsonify({'error': 'Invalid class count'}), 400

        class_count = int(class_count)
        class_schedule = {}

        c.execute('DELETE FROM class_schedule')

        for i in range(class_count):
            class_name = request.form.get(f'class_name_{i}')
            class_duration = request.form.get(f'class_duration_{i}')

            if not class_name or not class_duration or not class_duration.isdigit():
                return jsonify({'error': f'Missing or invalid data for class {i}'}), 400
            
            class_schedule[class_name] = int(class_duration)
            c.execute('INSERT INTO class_schedule (class_name, duration) VALUES (?, ?)', (class_name, int(class_duration)))
        
        conn.commit()

        print("Updated class schedule (stored in DB):", class_schedule)

        return jsonify({'message': 'Class schedule updated successfully!', 'redirect': url_for('attendance_page')})

    except Exception as e:
        return jsonify({'error': f"Class configuration failed. Error: {str(e)}"}), 500

def send_alert():
    global last_alert_time
    if time.time() - last_alert_time > ALERT_COOLDOWN:
        pb.push_note("Alert!", "Mobile phone detected in the classroom!")
        print("Alert sent to principal.")
        last_alert_time = time.time()

@app.route('/get_attendance')
def get_attendance():
    conn = sqlite3.connect('attendance.db', check_same_thread=False)
    c = conn.cursor()
    
    c.execute('SELECT name, class_name, entry_time, exit_time FROM attendance ORDER BY entry_time DESC')
    records = c.fetchall()
    conn.close()

    if not records:
        return jsonify([])

    attendance_data = []
    for row in records:
        attendance_data.append({
            "name": row[0],
            "class_name": row[1],
            "entry_time": row[2],
            "exit_time": row[3] if row[3] else "Still in Class"
        })

    return jsonify(attendance_data)

def detect_mobile(frame):
    results = yolo_model(frame)
    for result in results:
        for box in result.boxes.data:
            x1, y1, x2, y2, conf, cls = box
            if int(cls) == 67 and conf > 0.7:
                return True
    return False

def gen_frames():
    global last_recognition_time
    camera = cv2.VideoCapture(0)

    while True:
        success, frame = camera.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(gray)

        face_images = []
        face_positions = []

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            resized_face = cv2.resize(face, (160, 160))
            face_images.append(resized_face)
            face_positions.append((x, y, w, h))
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        if face_images:
            threading.Thread(target=process_multiple_faces, args=(face_images, face_positions, frame)).start()

        if detect_mobile(frame):
            send_alert()

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

def process_multiple_faces(face_images, face_positions, frame):
    global last_recognition_times
    recognized_students = []

    for face_img, (x, y, w, h) in zip(face_images, face_positions):
        try:
            embedding = get_face_embedding(face_img)
            match, distance = match_face(embedding)

            if match:
                current_time = time.time()
                if match in last_recognition_times and (current_time - last_recognition_times[match]) < 10:
                    print(f"⏳ Skipping {match} - Last recognized {round(current_time - last_recognition_times[match], 2)} sec ago")
                    continue
                
                last_recognition_times[match] = current_time
                recognized_students.append((match, (x, y, w, h)))
                print(f"✅ Recognized: {match} at {x}, {y}")
                mark_entry_exit_with_auto_classes(match)

            else:
                print(f"🚫 Unknown face at {x}, {y}")

        except Exception as e:
            print(f"⚠️ Error processing face at {x}, {y}: {str(e)}")

    if not recognized_students:
        print("🚫 No valid matches found for detected faces.")

if __name__ == '__main__':
    app.run(debug=True)