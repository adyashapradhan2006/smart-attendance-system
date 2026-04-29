import cv2
import numpy as np
import face_recognition
import pandas as pd
import time
import base64
from datetime import datetime, date
from pathlib import Path
from flask import Flask, render_template, jsonify, request, Response
from flask_socketio import SocketIO, emit

BASE_DIR        = Path(__file__).parent
KNOWN_FACES_DIR = BASE_DIR / "known_faces"
ATTENDANCE_DIR  = BASE_DIR / "attendance_logs"
RECOGNITION_TOL = 0.65
FRAME_SKIP      = 3
CAM_INDEX       = 0
SCALE           = 0.5

KNOWN_FACES_DIR.mkdir(exist_ok=True)
ATTENDANCE_DIR.mkdir(exist_ok=True)

app = Flask(__name__, template_folder=str(BASE_DIR))
app.config["SECRET_KEY"] = "attendance-secret-2024"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")


class AttendanceEngine:

    def __init__(self):
        self.known_encodings  = []
        self.known_names      = []
        self.attendance_today = {}
        self.running          = False
        self.cap              = None
        self.frame_count      = 0
        self.stats = {
            "total_known":   0,
            "present_today": 0,
            "last_detected": "---",
            "fps":           0,
        }
        self._load_known_faces()

    def _load_known_faces(self):
        self.known_encodings.clear()
        self.known_names.clear()
        print("\n--- Loading known faces ---")
        images = (
            list(KNOWN_FACES_DIR.glob("*.jpg"))
            + list(KNOWN_FACES_DIR.glob("*.jpeg"))
            + list(KNOWN_FACES_DIR.glob("*.png"))
        )
        if not images:
            print("WARNING: No images in known_faces/ - register someone first.")
        for p in images:
            try:
                img  = face_recognition.load_image_file(str(p))
                encs = face_recognition.face_encodings(img, model="large")
                if encs:
                    self.known_encodings.append(encs[0])
                    self.known_names.append(p.stem.replace("_", " ").title())
                    print("  OK: " + p.stem)
                else:
                    print("  SKIP " + p.name + ": no face found - use a clearer photo!")
            except Exception as ex:
                print("  ERROR " + p.name + ": " + str(ex))
        self.stats["total_known"] = len(self.known_names)
        print("--- " + str(len(self.known_names)) + " person(s) loaded ---\n")

    def register_face(self, name, frame):
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        encs = face_recognition.face_encodings(rgb, model="large")
        if not encs:
            return {
                "success": False,
                "message": "No face found! Use better lighting and look straight at camera."
            }
        safe = name.strip().replace(" ", "_").lower()
        cv2.imwrite(str(KNOWN_FACES_DIR / (safe + ".jpg")), frame)
        self.known_encodings.append(encs[0])
        self.known_names.append(name.strip().title())
        self.stats["total_known"] = len(self.known_names)
        print("Registered: " + name.title())
        return {"success": True, "message": name.title() + " registered successfully!"}

    def _today_csv(self):
        return ATTENDANCE_DIR / ("attendance_" + date.today().isoformat() + ".csv")

    def _mark_attendance(self, name):
        if name in self.attendance_today:
            return False
        now = datetime.now()
        self.attendance_today[name] = now
        self.stats["present_today"] = len(self.attendance_today)
        self.stats["last_detected"] = name
        csv_path = self._today_csv()
        row = pd.DataFrame([{
            "Name":   name,
            "Date":   now.date(),
            "Time":   now.strftime("%H:%M:%S"),
            "Status": "Present"
        }])
        row.to_csv(str(csv_path), mode="a", header=not csv_path.exists(), index=False)
        print("MARKED PRESENT: " + name + " at " + now.strftime("%H:%M:%S"))
        socketio.emit("attendance_marked", {
            "name": name,
            "time": now.strftime("%H:%M:%S"),
            "date": str(now.date())
        })
        return True

    def _detect_and_recognize(self, frame):
        small      = cv2.resize(frame, (0, 0), fx=SCALE, fy=SCALE)
        rgb_small  = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        locs_small = face_recognition.face_locations(rgb_small, model="hog")

        if not locs_small:
            cv2.putText(
                frame,
                "No face detected - move closer or improve lighting",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 120, 255), 2
            )
            return frame

        inv  = 1.0 / SCALE
        locs = [
            (int(t * inv), int(r * inv), int(b * inv), int(l * inv))
            for (t, r, b, l) in locs_small
        ]
        rgb_full  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb_full, locs, model="large")

        for enc, (top, right, bottom, left) in zip(encodings, locs):
            name      = "Unknown"
            best_dist = 1.0

            if self.known_encodings:
                dists     = face_recognition.face_distance(self.known_encodings, enc)
                idx       = int(np.argmin(dists))
                best_dist = float(dists[idx])
                print("  " + self.known_names[idx] + " dist=" + str(round(best_dist, 3)) + " limit=" + str(RECOGNITION_TOL))
                if best_dist <= RECOGNITION_TOL:
                    name = self.known_names[idx]
                    self._mark_attendance(name)

            color = (0, 220, 100) if name != "Unknown" else (30, 30, 220)
            lbl_y = top - 35 if top > 35 else bottom + 5

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, lbl_y), (right, lbl_y + 30), color, -1)
            cv2.putText(
                frame, name,
                (left + 6, lbl_y + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2
            )
            if name == "Unknown" and self.known_encodings:
                cv2.putText(
                    frame, "dist:" + str(round(best_dist, 2)),
                    (left + 6, lbl_y + 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (200, 200, 0), 1
                )
        return frame

    def generate_frames(self):
        self.cap = cv2.VideoCapture(CAM_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.running     = True
        self.frame_count = 0
        t_prev           = time.time()
        print("Camera started.")

        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                break
            self.frame_count += 1

            if self.frame_count % FRAME_SKIP == 0:
                frame  = self._detect_and_recognize(frame)
                t_now  = time.time()
                self.stats["fps"] = round(FRAME_SKIP / max(t_now - t_prev, 0.001), 1)
                t_prev = t_now
                socketio.emit("stats_update", self.stats)

            cv2.putText(
                frame,
                "Known:" + str(len(self.known_names)) + "  Present:" + str(self.stats["present_today"]) + "  FPS:" + str(self.stats["fps"]),
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 180), 2
            )
            cv2.putText(
                frame,
                datetime.now().strftime("%Y-%m-%d  %H:%M:%S"),
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (160, 160, 160), 1
            )
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + buf.tobytes()
                + b"\r\n"
            )

        self.cap.release()
        self.running = False
        print("Camera stopped.")

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()

    def get_today_records(self):
        p = self._today_csv()
        return pd.read_csv(str(p)).to_dict("records") if p.exists() else []

    def get_all_dates(self):
        return sorted(
            [p.stem.replace("attendance_", "")
             for p in ATTENDANCE_DIR.glob("attendance_*.csv")],
            reverse=True
        )

    def get_records_for_date(self, d):
        p = ATTENDANCE_DIR / ("attendance_" + d + ".csv")
        return pd.read_csv(str(p)).to_dict("records") if p.exists() else []


engine = AttendanceEngine()


@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        engine.generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/api/stop_camera", methods=["POST"])
def stop_camera():
    engine.stop()
    return jsonify({"success": True})


@app.route("/api/stats")
def stats():
    return jsonify(engine.stats)


@app.route("/api/today")
def today():
    return jsonify(engine.get_today_records())


@app.route("/api/dates")
def dates():
    return jsonify(engine.get_all_dates())


@app.route("/api/records/<date_str>")
def records(date_str):
    return jsonify(engine.get_records_for_date(date_str))


@app.route("/api/register", methods=["POST"])
def register():
    data      = request.json
    name      = data.get("name", "").strip()
    frame_b64 = data.get("frame", "")
    if not name or not frame_b64:
        return jsonify({"success": False, "message": "Name and frame required."})
    img_bytes = base64.b64decode(frame_b64.split(",")[-1])
    frame     = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    return jsonify(engine.register_face(name, frame))


@app.route("/api/reload_faces", methods=["POST"])
def reload_faces():
    engine._load_known_faces()
    return jsonify({"success": True, "count": len(engine.known_names)})


@app.route("/api/known_people")
def known_people():
    return jsonify(engine.known_names)


if __name__ == "__main__":
    print("\nSmart Attendance System starting...")
    print("Open http://localhost:5000 in your browser\n")
    socketio.run(app, host="0.0.0.0", port=5000, debug=False)
 