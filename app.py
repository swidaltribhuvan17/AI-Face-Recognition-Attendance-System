import os
import base64
import json
import sqlite3

from datetime import datetime, timedelta
from functools import wraps

import cv2
import numpy as np
import face_recognition

from flask import (
    Flask,
    request,
    redirect,
    url_for,
    session,
    jsonify,
    render_template_string
)

from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)


app = Flask(__name__)

app.secret_key = "ai_face_attendance_secret_key"

DB_NAME = "attendance.db"
def frame_from_data_url(data_url):

    if "," in data_url:
        data_url = data_url.split(",", 1)[1]

    image_bytes = base64.b64decode(data_url)

    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8
    )

    frame = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if frame is None:
        raise ValueError("Invalid camera image.")

    return frame

def get_db():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():

    conn = get_db()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS admins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            roll_no TEXT UNIQUE NOT NULL,
            email TEXT NOT NULL,
            mobile TEXT NOT NULL,
            face_encoding TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            slot TEXT NOT NULL,
            verification_count INTEGER DEFAULT 0,
            verification_cycles TEXT DEFAULT '[]',
            recognition_time TEXT,
            status TEXT NOT NULL,
            UNIQUE(user_id, date, slot)
        )
    """)

    admin = conn.execute(
        "SELECT * FROM admins WHERE username=?",
        ("admin",)
    ).fetchone()

    if admin is None:

        conn.execute(
            """
            INSERT INTO admins
            (username, password_hash)
            VALUES (?, ?)
            """,
            (
                "admin",
                generate_password_hash("admin123")
            )
        )

    conn.commit()
    conn.close()


init_db()


MATCH_TOLERANCE = 0.48

REQUIRED_VERIFICATIONS = 2

TOTAL_VERIFICATION_CYCLES = 3


SLOTS = [
    ("09:00-10:00", 9, 10),
    ("10:00-11:00", 10, 11),
    ("11:00-12:00", 11, 12),
    ("12:00-13:00", 12, 13),
    ("14:00-15:00", 14, 15),
    ("15:00-16:00", 15, 16)
]


DISPLAY_SLOTS = [
    "09:00-10:00",
    "10:00-11:00",
    "11:00-12:00",
    "12:00-13:00",
    "14:00-15:00",
    "15:00-16:00"
]
def login_required(func):

    @wraps(func)
    def wrapper(*args, **kwargs):

        if "admin_id" not in session:
            return redirect(url_for("login"))

        return func(*args, **kwargs)

    return wrapper


def is_sunday(date_value):

    return date_value.weekday() == 6


def current_slot(now):

    for slot_name, start_hour, end_hour in SLOTS:

        start = now.replace(
            hour=start_hour,
            minute=0,
            second=0,
            microsecond=0
        )

        end = now.replace(
            hour=end_hour,
            minute=0,
            second=0,
            microsecond=0
        )

        if start <= now < end:

            return (
                slot_name,
                start_hour,
                end_hour
            )

    return None
def verification_cycle(now):
    minute = now.minute

    if minute < 20:
        return 1
    elif minute < 40:
        return 2
    else:
        return 3
def find_duplicate_face(new_encoding):

    conn = get_db()

    users = conn.execute(
        """
        SELECT id, name, roll_no, face_encoding
        FROM users
        """
    ).fetchall()

    conn.close()

    for user in users:

        try:
            old_encoding = np.array(
                json.loads(user["face_encoding"])
            )

            distance = face_recognition.face_distance(
                [old_encoding],
                new_encoding
            )[0]

            if distance <= MATCH_TOLERANCE:
                return user

        except Exception:
            continue

    return None
def get_one_face_encoding(frame):

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    locations = face_recognition.face_locations(
        rgb
    )

    if len(locations) == 0:
        return None, "No face detected."

    if len(locations) > 1:
        return None, "Please keep only one face in camera."

    encodings = face_recognition.face_encodings(
        rgb,
        locations
    )

    if not encodings:
        return None, "Face encoding failed."

    return encodings[0], None
@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        name = request.form.get("name", "").strip()
        roll_no = request.form.get("roll_no", "").strip()
        email = request.form.get("email", "").strip()
        mobile = request.form.get("mobile", "").strip()
        image = request.form.get("image", "")

        if not name or not roll_no or not email or not mobile:
            return jsonify({
                "success": False,
                "message": "All fields are required."
            })

        if not image:
            return jsonify({
                "success": False,
                "message": "Face image is required."
            })

        conn = get_db()

        existing = conn.execute(
            "SELECT id FROM users WHERE roll_no=?",
            (roll_no,)
        ).fetchone()

        if existing:
            conn.close()

            return jsonify({
                "success": False,
                "message": "Roll number already registered."
            })

        conn.close()

        try:
            frame = frame_from_data_url(image)

            encoding, error = get_one_face_encoding(frame)

            if error:
                return jsonify({
                    "success": False,
                    "message": error
                })

            duplicate = find_duplicate_face(
                encoding
            )

            if duplicate:
                return jsonify({
                    "success": False,
                    "message": "This face is already registered."
                })

            conn = get_db()

            conn.execute(
                """
                INSERT INTO users
                (name, roll_no, email, mobile,
                 face_encoding, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    name,
                    roll_no,
                    email,
                    mobile,
                    json.dumps(encoding.tolist()),
                    datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                )
            )

            conn.commit()
            conn.close()

            return jsonify({
                "success": True,
                "message": "Student registered successfully."
            })

        except Exception as e:

            return jsonify({
                "success": False,
                "message": str(e)
            })

    return render_template_string(
        REGISTER_HTML
    )
@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        username = request.form.get(
            "username", ""
        ).strip()

        password = request.form.get(
            "password", ""
        )

        conn = get_db()

        admin = conn.execute(
            """
            SELECT *
            FROM admins
            WHERE username=?
            """,
            (username,)
        ).fetchone()

        conn.close()

        if admin and check_password_hash(
            admin["password_hash"],
            password
        ):

            session["admin_id"] = admin["id"]
            session["admin_username"] = admin["username"]

            return redirect(
                url_for("dashboard")
            )

        return render_template_string(
            LOGIN_HTML,
            error="Invalid username or password."
        )

    return render_template_string(
        LOGIN_HTML,
        error=""
    )
@app.route("/logout")
def logout():

    session.clear()

    return redirect(
        url_for("login")
    )
@app.route("/dashboard")
@login_required
def dashboard():

    conn = get_db()

    total_students = conn.execute(
        "SELECT COUNT(*) AS total FROM users"
    ).fetchone()["total"]

    today = datetime.now().date().isoformat()

    present_today = conn.execute(
        """
        SELECT COUNT(*) AS total
        FROM attendance
        WHERE date=? AND status='Present'
        """,
        (today,)
    ).fetchone()["total"]

    absent_today = conn.execute(
        """
        SELECT COUNT(*) AS total
        FROM attendance
        WHERE date=? AND status='Absent'
        """,
        (today,)
    ).fetchone()["total"]

    conn.close()

    return render_template_string(
        DASHBOARD_HTML,
        total_students=total_students,
        present_today=present_today,
        absent_today=absent_today,
        today=today
    )
@app.route("/attendance")
@login_required
def attendance():

    selected_date = request.args.get(
        "date",
        datetime.now().strftime("%Y-%m-%d")
    )

    selected_month = request.args.get(
        "month",
        datetime.now().strftime("%Y-%m")
    )

    selected_student = request.args.get(
        "student_id",
        ""
    )

    view = request.args.get(
        "view",
        "daily"
    )

    today = datetime.now().strftime("%Y-%m-%d")

    conn = get_db()

    users = conn.execute(
        """
        SELECT *
        FROM users
        ORDER BY roll_no
        """
    ).fetchall()

    # -----------------------------
    # DAILY ATTENDANCE
    # -----------------------------

    daily_records = conn.execute(
        """
        SELECT *
        FROM attendance
        WHERE date=?
        """,
        (selected_date,)
    ).fetchall()

    daily_data = []

    for user in users:

        slots = {}

        for slot in DISPLAY_SLOTS:
            slots[slot] = "-"

        for record in daily_records:

            if record["user_id"] == user["id"]:

                if record["slot"] in slots:
                    slots[record["slot"]] = record["status"]

        daily_data.append({
            "id": user["id"],
            "name": user["name"],
            "roll_no": user["roll_no"],
            "slots": slots
        })

   # -----------------------------
    # MONTHLY ATTENDANCE
    # -----------------------------

    monthly_records = conn.execute(
        """
        SELECT *
        FROM attendance
        WHERE date LIKE ?
        ORDER BY date
        """,
        (selected_month + "%",)
    ).fetchall()

    # Count attendance for each student on each date
    monthly_map = {}

    for record in monthly_records:
        key = (record["user_id"], record["date"])

        if key not in monthly_map:
            monthly_map[key] = {
                "present": 0,
                "absent": 0,
                "pending": 0
            }

        if record["status"] == "Present":
            monthly_map[key]["present"] += 1
        elif record["status"] == "Absent":
            monthly_map[key]["absent"] += 1
        elif record["status"] == "Pending":
            monthly_map[key]["pending"] += 1

    monthly_data = []

    try:
        month_start = datetime.strptime(
            selected_month + "-01",
            "%Y-%m-%d"
        )

        if month_start.month == 12:
            next_month = month_start.replace(
                year=month_start.year + 1,
                month=1,
                day=1
            )
        else:
            next_month = month_start.replace(
                month=month_start.month + 1,
                day=1
            )

        total_days = (next_month - month_start).days

    except Exception:
        month_start = datetime.now().replace(
            day=1,
            hour=0,
            minute=0,
            second=0,
            microsecond=0
        )
        total_days = 30

    # Generate date columns for the selected month
    monthly_dates = []

    for day_number in range(1, total_days + 1):
        current_date = month_start.replace(day=day_number)

        monthly_dates.append({
            "date": current_date.strftime("%Y-%m-%d"),
            "label": current_date.strftime("%d %b"),
            "holiday": current_date.weekday() == 6
        })

    # One row per student
    for user in users:

        attendance_by_date = {}

        for day in monthly_dates:

            date_string = day["date"]

            # Sunday is a holiday
            if day["holiday"]:
                attendance_by_date[date_string] = "H"
                continue

            counts = monthly_map.get(
                (user["id"], date_string),
                {
                    "present": 0,
                    "absent": 0,
                    "pending": 0
                }
            )

            present_count = counts["present"]
            absent_count = counts["absent"]
            completed_slots = present_count + absent_count

            if present_count >= 2:
                attendance_by_date[date_string] = "P"

            elif completed_slots >= len(DISPLAY_SLOTS):
                attendance_by_date[date_string] = "A"

            else:
                attendance_by_date[date_string] = "-"

        monthly_data.append({
            "id": user["id"],
            "name": user["name"],
            "roll_no": user["roll_no"],
            "attendance": attendance_by_date
        })
    # -----------------------------
    # STUDENT-WISE MONTHLY SUMMARY
    # -----------------------------

    student_summary = []

    for user in users:

        present = 0
        absent = 0
        pending = 0

        for record in monthly_records:

            if record["user_id"] != user["id"]:
                continue

            record_date = datetime.strptime(
                record["date"],
                "%Y-%m-%d"
            )

            if record_date.weekday() == 6:
                continue

            if record["status"] == "Present":
                present += 1

            elif record["status"] == "Absent":
                absent += 1

            elif record["status"] == "Pending":
                pending += 1

        completed = present + absent

        percentage = 0

        if completed > 0:

            percentage = round(
                (present / completed) * 100,
                2
            )

        student_summary.append({

            "id": user["id"],

            "name": user["name"],

            "roll_no": user["roll_no"],

            "present": present,

            "absent": absent,

            "pending": pending,

            "percentage": percentage

        })

    conn.close()

    # -----------------------------
    # SELECTED STUDENT
    # -----------------------------

    selected_student_data = None

    if selected_student:

        try:

            student_id = int(
                selected_student
            )

            for item in student_summary:

                if item["id"] == student_id:

                    selected_student_data = item
                    break

        except Exception:
            pass

    # -----------------------------
    # SUNDAY CHECK
    # -----------------------------

    holiday = False

    try:

        holiday = (
            datetime.strptime(
                selected_date,
                "%Y-%m-%d"
            ).weekday() == 6
        )

    except Exception:
        holiday = False

    return render_template_string(

        ATTENDANCE_HTML,

        users=users,

        daily_data=daily_data,

        monthly_data=monthly_data,

        monthly_dates=monthly_dates,

        student_summary=student_summary,

        selected_student_data=selected_student_data,

        selected_date=selected_date,

        selected_month=selected_month,

        selected_student=selected_student,

        today=today,

        view=view,

        holiday=holiday,

        display_slots=DISPLAY_SLOTS

    )
    return render_template_string(
        ATTENDANCE_HTML,
        attendance_data=attendance_data,
        selected_date=selected_date,
        holiday=holiday,
        display_slots=DISPLAY_SLOTS
    )
@app.route("/recognize", methods=["POST"])
@login_required
def recognize():

    data = request.get_json()

    if not data or "image" not in data:
        return jsonify({
            "success": False,
            "message": "Camera image not received."
        })

    try:

        frame = frame_from_data_url(
            data["image"]
        )

        result = recognize_faces(frame)

        return jsonify(result)

    except Exception as e:

        return jsonify({
            "success": False,
            "message": str(e)
        })
def recognize_faces(frame):

    now = datetime.now()

    if is_sunday(now.date()):
        return {
            "success": False,
            "message": "Sunday Holiday - Attendance closed.",
            "recognized": [],
            "face_count": 0
        }

    slot = current_slot(now)

    if slot is None:
        return {
            "success": False,
            "message": "Attendance is not active.",
            "recognized": [],
            "face_count": 0
        }

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    locations = face_recognition.face_locations(rgb)

    if not locations:
        return {
            "success": True,
            "message": "No face detected.",
            "recognized": [],
            "face_count": 0
        }

    encodings = face_recognition.face_encodings(
        rgb,
        locations
    )

    conn = get_db()

    users = conn.execute(
        """
        SELECT id, name, roll_no, face_encoding
        FROM users
        """
    ).fetchall()

    recognized = []

    cycle = verification_cycle(now)

    for encoding in encodings:

        best_user = None
        best_distance = 1.0

        for user in users:

            try:
                old_encoding = np.array(
                    json.loads(user["face_encoding"])
                )

                distance = face_recognition.face_distance(
                    [old_encoding],
                    encoding
                )[0]

                if distance < best_distance:
                    best_distance = distance
                    best_user = user

            except:
                continue

        if best_user is None:
            continue

        if best_distance > MATCH_TOLERANCE:
            continue

        today = now.date().isoformat()

        row = conn.execute(
            """
            SELECT *
            FROM attendance
            WHERE user_id=? AND date=? AND slot=?
            """,
            (
                best_user["id"],
                today,
                slot[0]
            )
        ).fetchone()

        if row is None:

            conn.execute(
                """
                INSERT INTO attendance
                (
                    user_id,
                    date,
                    slot,
                    verification_count,
                    verification_cycles,
                    recognition_time,
                    status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    best_user["id"],
                    today,
                    slot[0],
                    1,
                    json.dumps([cycle]),
                    now.strftime("%Y-%m-%d %H:%M:%S"),
                    "Pending"
                )
            )

            status = "Pending"

        else:

            cycles = json.loads(
                row["verification_cycles"] or "[]"
            )

            if cycle not in cycles:
                cycles.append(cycle)

            count = len(cycles)

            status = (
                "Present"
                if count >= REQUIRED_VERIFICATIONS
                else "Pending"
            )

            conn.execute(
                """
                UPDATE attendance
                SET verification_count=?,
                    verification_cycles=?,
                    recognition_time=?,
                    status=?
                WHERE id=?
                """,
                (
                    count,
                    json.dumps(cycles),
                    now.strftime("%Y-%m-%d %H:%M:%S"),
                    status,
                    row["id"]
                )
            )

        recognized.append({
            "name": best_user["name"],
            "roll_no": best_user["roll_no"],
            "status": status,
            "cycle": cycle
        })

    conn.commit()
    conn.close()

    return {
        "success": True,
        "message": "Face recognition completed.",
        "recognized": recognized,
        "face_count": len(locations),
        "slot": slot[0],
        "cycle": cycle
    }
def finalize_completed_slots():

    now = datetime.now()

    if is_sunday(now.date()):
        return

    conn = get_db()

    users = conn.execute(
        "SELECT id FROM users"
    ).fetchall()

    for slot_name, start_hour, end_hour in SLOTS:

        if now.hour < end_hour:
            continue

        for user in users:

            row = conn.execute(
                """
                SELECT *
                FROM attendance
                WHERE user_id=? AND date=? AND slot=?
                """,
                (
                    user["id"],
                    now.date().isoformat(),
                    slot_name
                )
            ).fetchone()

            if row is None:

                conn.execute(
                    """
                    INSERT INTO attendance
                    (
                        user_id,
                        date,
                        slot,
                        verification_count,
                        verification_cycles,
                        recognition_time,
                        status
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        user["id"],
                        now.date().isoformat(),
                        slot_name,
                        0,
                        "[]",
                        "",
                        "Absent"
                    )
                )

            elif row["status"] == "Pending":

                status = (
                    "Present"
                    if row["verification_count"]
                    >= REQUIRED_VERIFICATIONS
                    else "Absent"
                )

                conn.execute(
                    """
                    UPDATE attendance
                    SET status=?
                    WHERE id=?
                    """,
                    (status, row["id"])
                )

    conn.commit()
    conn.close()
@app.route("/api/status")
@login_required
def api_status():

    now = datetime.now()

    if is_sunday(now.date()):
        return jsonify({
            "active": False,
            "message": "Sunday Holiday"
        })

    slot = current_slot(now)

    if slot is None:
        return jsonify({
            "active": False,
            "message": "Attendance is not active"
        })

    return jsonify({
        "active": True,
        "slot": slot[0],
        "cycle": verification_cycle(now),
        "message": "Attendance is active"
    })
@app.route("/report")
@login_required
def report():

    from_date = request.args.get(
        "from_date",
        datetime.now().strftime("%Y-%m-%d")
    )

    to_date = request.args.get(
        "to_date",
        datetime.now().strftime("%Y-%m-%d")
    )

    conn = get_db()

    users = conn.execute(
        "SELECT * FROM users ORDER BY roll_no"
    ).fetchall()

    records = conn.execute(
        """
        SELECT *
        FROM attendance
        WHERE date BETWEEN ? AND ?
        """,
        (from_date, to_date)
    ).fetchall()

    conn.close()

    report_data = []

    for user in users:

        present = 0
        absent = 0
        pending = 0

        for record in records:

            if record["user_id"] != user["id"]:
                continue

            day = datetime.strptime(
                record["date"],
                "%Y-%m-%d"
            ).weekday()

            if day == 6:
                continue

            if record["status"] == "Present":
                present += 1

            elif record["status"] == "Absent":
                absent += 1

            elif record["status"] == "Pending":
                pending += 1

        total = present + absent

        percentage = 0

        if total > 0:
            percentage = round(
                (present / total) * 100,
                2
            )

        report_data.append({
            "name": user["name"],
            "roll_no": user["roll_no"],
            "present": present,
            "absent": absent,
            "pending": pending,
            "percentage": percentage
        })

    return render_template_string(
        REPORT_HTML,
        report_data=report_data,
        from_date=from_date,
        to_date=to_date
    )
@app.route(
    "/delete-student/<int:user_id>",
    methods=["POST"]
)
@login_required
def delete_student(user_id):

    conn = get_db()

    conn.execute(
        "DELETE FROM attendance WHERE user_id=?",
        (user_id,)
    )

    conn.execute(
        "DELETE FROM users WHERE id=?",
        (user_id,)
    )

    conn.commit()
    conn.close()

    return redirect(
        url_for("admin")
    )
@app.route("/admin")
@login_required
def admin():


    conn = get_db()

    users = conn.execute(
        "SELECT * FROM users ORDER BY roll_no"
    ).fetchall()

    conn.close()

    return render_template_string(
        ADMIN_HTML,
        users=users
    )


@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():

    if request.method == "POST":

        username = request.form.get(
            "username", ""
        ).strip()

        password = request.form.get(
            "password", ""
        )

        conn = get_db()

        admin_user = conn.execute(
            """
            SELECT *
            FROM admins
            WHERE username=?
            """,
            (username,)
        ).fetchone()

        conn.close()

        if admin_user and check_password_hash(
            admin_user["password_hash"],
            password
        ):

            session["admin_id"] = admin_user["id"]
            session["admin_username"] = admin_user["username"]

            return redirect(
                url_for("admin")
            )

        return render_template_string(
            ADMIN_LOGIN_HTML,
            error="Invalid username or password."
        )

    return render_template_string(
        ADMIN_LOGIN_HTML,
        error=""
    )


@app.route("/admin/logout")
def admin_logout():

    session.clear()

    return redirect(
        url_for("login")
    )


@app.route("/")
def home():

    if "admin_id" in session:
        return redirect(
            url_for("dashboard")
        )

    return redirect(
        url_for("login")
    )
LOGIN_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Admin Login</title>

    <style>
        * {
            box-sizing: border-box;
            font-family: Arial, sans-serif;
        }

        body {
            margin: 0;
            background: #eef3f8;
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }

        .login-box {
            width: 380px;
            background: white;
            padding: 35px;
            border-radius: 18px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.12);
        }

        h1 {
            text-align: center;
            color: #0b2d4d;
            margin-bottom: 8px;
        }

        .subtitle {
            text-align: center;
            color: #777;
            margin-bottom: 25px;
        }

        label {
            font-weight: bold;
            color: #333;
        }

        input {
            width: 100%;
            padding: 13px;
            margin: 8px 0 18px;
            border: 1px solid #ccd5df;
            border-radius: 9px;
        }

        button {
            width: 100%;
            padding: 13px;
            border: none;
            border-radius: 9px;
            background: #0b5ed7;
            color: white;
            font-size: 16px;
            cursor: pointer;
        }

        button:hover {
            background: #084da8;
        }

        .error {
            background: #ffe5e5;
            color: #c62828;
            padding: 10px;
            border-radius: 8px;
            margin-bottom: 15px;
            text-align: center;
        }
    </style>
</head>

<body>

<div class="login-box">

    <h1>AI Attendance</h1>

    <div class="subtitle">
        Face Recognition Attendance System
    </div>

    {% if error %}
        <div class="error">{{ error }}</div>
    {% endif %}

    <form method="POST">

        <label>Username</label>

        <input
            type="text"
            name="username"
            placeholder="Enter username"
            required
        >

        <label>Password</label>

        <input
            type="password"
            name="password"
            placeholder="Enter password"
            required
        >

        <button type="submit">
            Login
        </button>

    </form>

</div>

</body>
</html>
"""


DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>

    <title>Dashboard</title>

    <style>

        * {
            box-sizing: border-box;
            font-family: Arial, sans-serif;
        }

        body {
            margin: 0;
            background: #f3f6fa;
            color: #222;
        }

        .navbar {
            background: #092c4c;
            color: white;
            padding: 18px 30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .navbar h2 {
            margin: 0;
        }

        .logout {
            color: white;
            text-decoration: none;
            background: #d62828;
            padding: 9px 16px;
            border-radius: 8px;
        }

        .container {
            padding: 30px;
        }

        .cards {
            display: grid;
            grid-template-columns:
                repeat(auto-fit, minmax(220px, 1fr));
            gap: 20px;
        }

        .card {
            background: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.08);
        }

        .card h3 {
            color: #666;
            margin-top: 0;
        }

        .number {
            font-size: 35px;
            font-weight: bold;
            color: #0b5ed7;
        }

        .menu {
            margin-top: 30px;
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
        }

        .menu a {
            text-decoration: none;
            background: #0b5ed7;
            color: white;
            padding: 13px 20px;
            border-radius: 9px;
        }

        .menu a:hover {
            background: #084da8;
        }

    </style>

</head>

<body>

<div class="navbar">

    <h2>AI Face Recognition Attendance</h2>

    <a class="logout" href="/logout">
        Logout
    </a>

</div>

<div class="container">

    <h1>Dashboard</h1>

    <p>
        Today's Date: {{ today }}
    </p>

    <div class="cards">

        <div class="card">
            <h3>Total Students</h3>
            <div class="number">
                {{ total_students }}
            </div>
        </div>

        <div class="card">
            <h3>Present Today</h3>
            <div class="number">
                {{ present_today }}
            </div>
        </div>

        <div class="card">
            <h3>Absent Today</h3>
            <div class="number">
                {{ absent_today }}
            </div>
        </div>

    </div>

    <div class="menu">

        <a href="/register">
            Register Student
        </a>

        <a href="/attendance">
            Attendance
        </a>

        <a href="/report">
            Reports
        </a>

        <a href="/admin">
            Admin Panel
        </a>

    </div>

</div>

</body>
</html>
"""


REGISTER_HTML = """
<!DOCTYPE html>
<html>

<head>

    <title>Register Student</title>

    <style>

        * {
            box-sizing: border-box;
            font-family: Arial, sans-serif;
        }

        body {
            margin: 0;
            background: #eef3f8;
        }

        .navbar {
            background: #092c4c;
            color: white;
            padding: 18px 30px;
        }

        .container {
            max-width: 900px;
            margin: 30px auto;
            background: white;
            padding: 30px;
            border-radius: 18px;
            box-shadow: 0 5px 25px rgba(0,0,0,0.1);
        }

        h1 {
            color: #092c4c;
        }

        .form-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 18px;
        }

        input {
            width: 100%;
            padding: 12px;
            border: 1px solid #ccd5df;
            border-radius: 8px;
        }

        label {
            font-weight: bold;
        }

        .field {
            margin-bottom: 5px;
        }

        .camera {
            margin-top: 20px;
            text-align: center;
        }

        video {
            width: 100%;
            max-width: 500px;
            border-radius: 12px;
            background: black;
        }

        button {
            margin-top: 15px;
            padding: 13px 25px;
            border: none;
            border-radius: 8px;
            background: #0b5ed7;
            color: white;
            cursor: pointer;
        }

        #message {
            margin-top: 15px;
            font-weight: bold;
        }

        @media(max-width:700px) {

            .form-grid {
                grid-template-columns: 1fr;
            }

        }

    </style>

</head>

<body>

<div class="navbar">
    <h2>Student Registration</h2>
</div>

<div class="container">

    <h1>Register New Student</h1>

    <form id="registerForm">

        <div class="form-grid">

            <div class="field">
                <label>Name</label>
                <input
                    type="text"
                    name="name"
                    required
                >
            </div>

            <div class="field">
                <label>Roll Number</label>
                <input
                    type="text"
                    name="roll_no"
                    required
                >
            </div>

            <div class="field">
                <label>Email</label>
                <input
                    type="email"
                    name="email"
                    required
                >
            </div>

            <div class="field">
                <label>Mobile</label>
                <input
                    type="text"
                    name="mobile"
                    required
                >
            </div>

        </div>

        <div class="camera">

            <h3>Face Capture</h3>

            <video
                id="video"
                autoplay
                playsinline
            ></video>

            <br>

            <button
                type="button"
                onclick="captureFace()"
            >
                Capture Face
            </button>

            <canvas
                id="canvas"
                style="display:none;"
            ></canvas>

            <input
                type="hidden"
                id="image"
                name="image"
            >

            <br>

            <button type="submit">
                Register Student
            </button>

            <div id="message"></div>

        </div>

    </form>

</div>

<script>

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const image = document.getElementById("image");
const message = document.getElementById("message");

navigator.mediaDevices.getUserMedia({
    video: true
})
.then(stream => {
    video.srcObject = stream;
})
.catch(() => {
    message.innerText =
        "Camera permission required.";
});


function captureFace() {

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    canvas.getContext("2d").drawImage(
        video,
        0,
        0
    );

    image.value =
        canvas.toDataURL("image/jpeg");

    message.innerText =
        "Face captured successfully.";

}


document.getElementById(
    "registerForm"
).addEventListener(
    "submit",
    async function(e) {

        e.preventDefault();

        const formData =
            new FormData(this);

        const response =
            await fetch("/register", {
                method: "POST",
                body: formData
            });

        const result =
            await response.json();

        message.innerText =
            result.message;

        if (result.success) {
            this.reset();
            image.value = "";
        }

    }
);

</script>

</body>
</html>
"""
ATTENDANCE_HTML = """
<!DOCTYPE html>
<html>

<head>

<meta charset="UTF-8">

<meta name="viewport"
content="width=device-width, initial-scale=1.0">

<title>Attendance Management</title>

<style>

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    font-family: Arial, Helvetica, sans-serif;
    background: #f4f7fb;
    color: #172033;
}

.navbar {
    background: linear-gradient(
        135deg,
        #0f172a,
        #1e3a8a
    );

    color: white;
    padding: 16px 28px;

    display: flex;
    justify-content: space-between;
    align-items: center;

    flex-wrap: wrap;
    gap: 12px;
}

.nav-title {
    font-size: 22px;
    font-weight: bold;
}

.nav-links {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
}

.nav-links a {
    text-decoration: none;
    color: white;

    padding: 9px 14px;

    border-radius: 8px;

    background: rgba(
        255,
        255,
        255,
        0.10
    );

    font-size: 14px;
}

.nav-links a:hover {
    background: rgba(
        255,
        255,
        255,
        0.20
    );
}

.container {
    width: 95%;
    max-width: 1500px;
    margin: 25px auto;
}

.page-header {
    display: flex;

    justify-content: space-between;
    align-items: center;

    flex-wrap: wrap;

    gap: 15px;

    margin-bottom: 20px;
}

.page-header h1 {
    margin: 0;
    font-size: 28px;
}

.card {
    background: white;

    border-radius: 16px;

    padding: 20px;

    margin-bottom: 20px;

    box-shadow:
    0 5px 20px
    rgba(
        15,
        23,
        42,
        0.08
    );
}

.filters {
    display: flex;

    gap: 12px;

    align-items: end;

    flex-wrap: wrap;
}

.filter-group {
    display: flex;

    flex-direction: column;

    gap: 6px;
}

.filter-group label {
    font-size: 13px;

    font-weight: bold;

    color: #475569;
}

input,
select {
    padding: 11px 13px;

    border: 1px solid #d6dce5;

    border-radius: 9px;

    font-size: 14px;

    background: white;
}

button,
.btn {
    border: none;

    padding: 11px 17px;

    border-radius: 9px;

    cursor: pointer;

    font-weight: bold;

    text-decoration: none;

    display: inline-block;
}

.btn-blue {
    background: #2563eb;
    color: white;
}

.btn-green {
    background: #16a34a;
    color: white;
}

.btn-orange {
    background: #f59e0b;
    color: white;
}

.btn-dark {
    background: #0f172a;
    color: white;
}

.btn-light {
    background: #e2e8f0;
    color: #0f172a;
}

.tabs {
    display: flex;

    gap: 8px;

    flex-wrap: wrap;

    margin-bottom: 20px;
}

.tab {
    padding: 11px 18px;

    border-radius: 9px;

    text-decoration: none;

    background: #e2e8f0;

    color: #334155;

    font-weight: bold;
}

.tab.active {
    background: #2563eb;
    color: white;
}

.stats {
    display: grid;

    grid-template-columns:
    repeat(4, 1fr);

    gap: 15px;

    margin-bottom: 20px;
}

.stat {
    background: white;

    padding: 20px;

    border-radius: 14px;

    box-shadow:
    0 5px 15px
    rgba(
        15,
        23,
        42,
        0.06
    );
}

.stat-title {
    color: #64748b;
    font-size: 13px;
}

.stat-value {
    font-size: 27px;

    font-weight: bold;

    margin-top: 6px;
}

.green {
    color: #15803d;
}

.red {
    color: #dc2626;
}

.orange {
    color: #d97706;
}

.blue {
    color: #2563eb;
}

.table-wrapper {
    overflow-x: auto;
}

table {
    width: 100%;

    border-collapse: collapse;

    min-width: 950px;
}

th {
    background: #0f172a;

    color: white;

    padding: 12px;

    text-align: center;

    font-size: 13px;
}

td {
    padding: 11px;

    border-bottom:
    1px solid #e5e7eb;

    text-align: center;

    font-size: 13px;
}

td:first-child,
td:nth-child(2) {
    text-align: left;
}

.badge {
    display: inline-block;

    padding: 5px 9px;

    border-radius: 20px;

    font-size: 11px;

    font-weight: bold;
}

.badge-present {
    background: #dcfce7;
    color: #15803d;
}

.badge-absent {
    background: #fee2e2;
    color: #dc2626;
}

.badge-pending {
    background: #fef3c7;
    color: #b45309;
}

.badge-none {
    background: #e2e8f0;
    color: #64748b;
}

.holiday-row {
    background: #fff7ed;
}

.holiday-box {
    padding: 16px;

    border-radius: 12px;

    background: #fff7ed;

    color: #c2410c;

    border:
    1px solid #fed7aa;

    margin-bottom: 18px;

    font-weight: bold;
}

.camera-box {
    background: #0f172a;

    border-radius: 16px;

    padding: 18px;

    color: white;

    margin-bottom: 20px;
}

.camera-header {
    display: flex;

    justify-content: space-between;

    align-items: center;

    flex-wrap: wrap;

    gap: 10px;
}

.camera-status {
    padding: 7px 12px;

    border-radius: 20px;

    background: #334155;

    font-size: 12px;
}

 {
    margin-top: 15px;

    display: flex;

    justify-content: center;
}

video {
    width: 100%;
.camera-area {
    width: 100%;
    max-width: 700px;
    height: 420px;
    margin: 0 auto;
    border-radius: 14px;
    background: black;
    overflow: hidden;
}

.camera-area video {
    width: 100%;
    height: 100%;
    display: block;
    object-fit: cover;
    background: black;
}

.camera-buttons {
    margin-top: 15px;

    display: flex;

    gap: 10px;

    justify-content: center;

    flex-wrap: wrap;
}

.result-box {
    margin-top: 15px;

    padding: 15px;

    background: #1e293b;

    border-radius: 12px;

    min-height: 55px;
}

.student-card {
    display: grid;

    grid-template-columns:
    repeat(4, 1fr);

    gap: 15px;
}

.student-summary {
    background: #f8fafc;

    padding: 18px;

    border-radius: 12px;

    border:
    1px solid #e2e8f0;
}

.student-summary h3 {
    margin-top: 0;
}

.percent {
    font-size: 24px;

    font-weight: bold;
}

.print-only {
    display: none;
}

@media(max-width: 900px) {

    .stats {
        grid-template-columns:
        repeat(2, 1fr);
    }

    .student-card {
        grid-template-columns:
        repeat(2, 1fr);
    }

}

@media(max-width: 600px) {

    .stats {
        grid-template-columns: 1fr;
    }

    .student-card {
        grid-template-columns: 1fr;
    }

    .container {
        width: 92%;
    }

    .page-header h1 {
        font-size: 22px;
    }

}

@media print {

    body {
        background: white;
    }

    .navbar,
    .filters,
    .tabs,
    .camera-box,
    .no-print {
        display: none !important;
    }

    .card {
        box-shadow: none;
        border: none;
    }

    .print-only {
        display: block;
    }

    table {
        min-width: 0;
    }

}

</style>

</head>

<body>

<div class="navbar">

    <div class="nav-title">
        🤖 AI Face Attendance
    </div>

    <div class="nav-links">

        <a href="{{ url_for('dashboard') }}">
            Dashboard
        </a>

        <a href="{{ url_for('attendance') }}">
            Attendance
        </a>

        <a href="{{ url_for('report') }}">
            Reports
        </a>

        <a href="{{ url_for('admin') }}">
            Students
        </a>

        <a href="{{ url_for('logout') }}">
            Logout
        </a>

    </div>

</div>

<div class="container">

<div class="page-header">

    <div>

        <h1>
            Attendance Management
        </h1>

        <p>
            Daily, Monthly & Student-wise Attendance
        </p>

    </div>

    <button
        class="btn btn-dark"
        onclick="window.print()">

        🖨 Print

    </button>

</div>

<div class="tabs no-print">

    <a
        class="tab {% if view == 'daily' %}active{% endif %}"
        href="{{ url_for(
            'attendance',
            view='daily',
            date=selected_date
        ) }}">

        📅 Daily Attendance

    </a>

    <a
        class="tab {% if view == 'monthly' %}active{% endif %}"
        href="{{ url_for(
            'attendance',
            view='monthly',
            month=selected_month
        ) }}">

        📆 Monthly Attendance

    </a>

    <a
        class="tab {% if view == 'student' %}active{% endif %}"
        href="{{ url_for(
            'attendance',
            view='student',
            month=selected_month
        ) }}">

        👩‍🎓 Student-wise

    </a>

</div>

{% if view == 'daily' %}

<div class="card no-print">

    <form
        method="GET"
        action="{{ url_for('attendance') }}">

        <input
            type="hidden"
            name="view"
            value="daily">

        <div class="filters">

            <div class="filter-group">

                <label>
                    Select Date
                </label>

                <input
                    type="date"
                    name="date"
                    value="{{ selected_date }}">

            </div>

            <button
                class="btn btn-blue"
                type="submit">

                View Attendance

            </button>

        </div>

    </form>

</div>

{% if holiday %}

<div class="holiday-box">

    ☀️ Sunday Holiday —
    Attendance is not conducted on Sundays.

</div>

{% elif selected_date == today %}

<div class="camera-box no-print">

    <div class="camera-header">

        <div>

            <h2>
                📷 Live Classroom Camera
            </h2>

            <div id="slotText">
                Checking attendance schedule...
            </div>

        </div>

        <div
            class="camera-status"
            id="cameraStatus">

            Camera OFF

        </div>

    </div>

    <div class="camera-area">

        <video
            id="camera"
            autoplay
            playsinline>
        </video>

    </div>

    <div class="camera-buttons">

        <button
            class="btn btn-green"
            onclick="startCamera()">

            ▶ Start Camera

        </button>

        <button
            class="btn btn-orange"
            onclick="stopCamera()">

            ■ Stop Camera

        </button>

    </div>

    <div
        class="result-box"
        id="resultBox">

        Camera is ready.

    </div>

</div>

{% else %}

<div class="card">

    <strong>
        📖 View-only mode
    </strong>

    <p>
        Camera attendance is available only
        for today's date.
    </p>

</div>

{% endif %}

<div class="stats">

    <div class="stat">

        <div class="stat-title">
            Total Students
        </div>

        <div class="stat-value blue">
            {{ daily_data|length }}
        </div>

    </div>

    <div class="stat">

        <div class="stat-title">
            Attendance Date
        </div>

        <div
            class="stat-value"
            style="font-size:18px">

            {{ selected_date }}

        </div>

    </div>

    <div class="stat">

        <div class="stat-title">
            Attendance Slots
        </div>

        <div class="stat-value green">
            6
        </div>

    </div>

    <div class="stat">

        <div class="stat-title">
            Lunch Break
        </div>

        <div
            class="stat-value orange"
            style="font-size:18px">

            1 PM – 2 PM

        </div>

    </div>

</div>

<div class="card">

    <h2>
        Daily Slot-wise Attendance
    </h2>

    <div class="table-wrapper">

        <table>

            <thead>

                <tr>

                    <th>
                        Roll No
                    </th>

                    <th>
                        Student Name
                    </th>

                    {% for slot in display_slots %}

                    <th>
                        {{ slot }}
                    </th>

                    {% endfor %}

                </tr>

            </thead>

            <tbody>

            {% for student in daily_data %}

                <tr>

                    <td>
                        {{ student.roll_no }}
                    </td>

                    <td>
                        {{ student.name }}
                    </td>

                    {% for slot in display_slots %}

                        {% set status =
                        student.slots[slot] %}

                        <td>

                        {% if status == "Present" %}

                            <span
                                class="badge badge-present">
                                P
                            </span>

                        {% elif status == "Absent" %}

                            <span
                                class="badge badge-absent">
                                A
                            </span>

                        {% elif status == "Pending" %}

                            <span
                                class="badge badge-pending">
                                Pending
                            </span>

                        {% else %}

                            <span
                                class="badge badge-none">
                                -
                            </span>

                        {% endif %}

                        </td>

                    {% endfor %}

                </tr>

            {% endfor %}

            </tbody>

        </table>

    </div>

</div>

{% endif %}
{% if view == 'monthly' %}

<div class="card no-print">

    <form
        method="GET"
        action="{{ url_for('attendance') }}">

        <input
            type="hidden"
            name="view"
            value="monthly">

        <div class="filters">

            <div class="filter-group">

                <label>
                    Select Month
                </label>

                <input
                    type="month"
                    name="month"
                    value="{{ selected_month }}">

            </div>

            <button
                class="btn btn-blue"
                type="submit">

                View Month

            </button>

        </div>

    </form>

</div>


<div class="card">

    <h2>📆 Monthly Attendance</h2>

<p>
    Month:
    <strong>{{ selected_month }}</strong>
</p>

<div class="table-wrapper">

    <table>
        <thead>
            <tr>
                <th>Roll No</th>
                <th>Student Name</th>

                {% for day in monthly_dates %}
                    <th {% if day.holiday %}style="background:#fef3c7;color:#92400e"{% endif %}>
                        {{ day.label }}
                    </th>
                {% endfor %}
            </tr>
        </thead>

        <tbody>
            {% for student in monthly_data %}
                <tr>
                    <td>{{ student.roll_no }}</td>
                    <td>{{ student.name }}</td>

                    {% for day in monthly_dates %}
                        {% set status = student.attendance[day.date] %}

                        <td {% if day.holiday %}style="background:#fef3c7"{% endif %}>
                            {% if status == "P" %}
                                <span class="badge badge-present">P</span>
                            {% elif status == "A" %}
                                <span class="badge badge-absent">A</span>
                            {% elif status == "H" %}
                                <strong style="color:#92400e">H</strong>
                            {% else %}
                                <span class="badge badge-none">-</span>
                            {% endif %}
                        </td>
                    {% endfor %}

                </tr>
            {% endfor %}
        </tbody>
    </table>

</div>

</div>

{% endif %}
<!-- ========================= -->
<!-- STUDENT-WISE VIEW -->
<!-- ========================= -->

{% if view == 'student' %}

<div class="card no-print">

    <form
        method="GET"
        action="{{ url_for('attendance') }}">

        <input
            type="hidden"
            name="view"
            value="student">

        <div class="filters">

            <div class="filter-group">

                <label>
                    Select Student
                </label>

                <select name="student_id">

                    <option value="">
                        Select Student
                    </option>

                    {% for user in users %}

                    <option
                        value="{{ user.id }}"
                        {% if selected_student ==
                        user.id|string %}
                        selected
                        {% endif %}>

                        {{ user.roll_no }}
                        -
                        {{ user.name }}

                    </option>

                    {% endfor %}

                </select>

            </div>


            <div class="filter-group">

                <label>
                    Select Month
                </label>

                <input
                    type="month"
                    name="month"
                    value="{{ selected_month }}">

            </div>


            <button
                class="btn btn-blue"
                type="submit">

                View Student

            </button>

        </div>

    </form>

</div>


{% if selected_student_data %}

<div class="student-card">

    <div class="student-summary">

        <h3>
            👩‍🎓 Student
        </h3>

        <strong>
            {{ selected_student_data.name }}
        </strong>

        <p>
            Roll No:
            {{ selected_student_data.roll_no }}
        </p>

    </div>


    <div class="student-summary">

        <h3>
            🟢 Present
        </h3>

        <div class="percent green">

            {{ selected_student_data.present }}

        </div>

        <small>
            Present slot records
        </small>

    </div>


    <div class="student-summary">

        <h3>
            🔴 Absent
        </h3>

        <div class="percent red">

            {{ selected_student_data.absent }}

        </div>

        <small>
            Absent slot records
        </small>

    </div>


    <div class="student-summary">

        <h3>
            📊 Attendance
        </h3>

        <div class="percent blue">

            {{ selected_student_data.percentage }}%

        </div>

        <small>
            Completed slots
        </small>

    </div>

</div>

{% else %}

<div class="card">

    <h3>
        Student-wise Attendance
    </h3>

    <p>
        Select a student and month above to view
        detailed attendance.
    </p>

</div>

{% endif %}


<div class="card">

    <h2>
        📊 All Students Monthly Summary
    </h2>

    <div class="table-wrapper">

    <table>
        <thead>
            <tr>
                <th>Date</th>
                <th>Roll No</th>
                <th>Student</th>
                <th>Attendance</th>
            </tr>
        </thead>

        <tbody>
            {% for day in monthly_data %}

                {% if day.holiday %}

                    <tr class="holiday-row">
                        <td colspan="4">
                            ☀️ {{ day.date }} — SUNDAY HOLIDAY (H)
                        </td>
                    </tr>

                {% else %}

                    {% for student in day.students %}
                        <tr>
                            <td>{{ day.date }}</td>
                            <td>{{ student.roll_no }}</td>
                            <td>{{ student.name }}</td>

                            <td>
                                {% if student.status == "P" %}
                                    <span class="badge badge-present">P</span>

                                {% elif student.status == "A" %}
                                    <span class="badge badge-absent">A</span>

                                {% else %}
                                    <span class="badge badge-none">-</span>
                                {% endif %}
                            </td>
                        </tr>
                    {% endfor %}

                {% endif %}

            {% endfor %}
        </tbody>
    </table>

</div>

</div>

{% endif %}


</div>
<!-- CAMERA JAVASCRIPT -->

<script>

let cameraStream = null;
let scanTimer = null;
let scanning = false;

async function startCamera() {

    const video =
        document.getElementById("camera");

    const status =
        document.getElementById("cameraStatus");

    const result =
        document.getElementById("resultBox");

    try {

        cameraStream =
            await navigator.mediaDevices.getUserMedia({
                video: true,
                audio: false
            });

        video.srcObject = cameraStream;

        video.muted = true;
        video.playsInline = true;

        await video.play();

        status.innerText = "Camera ON";
        status.style.background = "#16a34a";

        result.innerText =
            "Camera started successfully.";
            scanTimer = setInterval(scanFrame,5000);

    } catch (error) {

        console.error(error);

        status.innerText = "Camera Error";
        status.style.background = "#dc2626";

        result.innerText =
            "Camera Error: " + error.message;
    }
}


function stopCamera() {

    if (cameraStream) {

        cameraStream
            .getTracks()
            .forEach(function(track) {
                track.stop();
            });

        cameraStream = null;
    }

    if (scanTimer) {

        clearInterval(scanTimer);
        scanTimer = null;
    }

    const video =
        document.getElementById("camera");

    const status =
        document.getElementById("cameraStatus");

    const result =
        document.getElementById("resultBox");

    video.srcObject = null;

    status.innerText = "Camera OFF";
    status.style.background = "#334155";

    result.innerText =
        "Camera stopped.";
}
async function scanFrame() {

    if (!cameraStream || scanning) {
        return;
    }

    const video = document.getElementById("camera");
    const result = document.getElementById("resultBox");

    if (video.videoWidth === 0 || video.videoHeight === 0) {
        return;
    }

    scanning = true;

    try {

        const canvas = document.createElement("canvas");

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        const context = canvas.getContext("2d");

        context.drawImage(
            video,
            0,
            0,
            canvas.width,
            canvas.height
        );

        const image = canvas.toDataURL("image/jpeg", 0.8);

        const response = await fetch(
            "{{ url_for('recognize') }}",
            {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    image: image
                })
            }
        );

        const data = await response.json();

        console.log("Recognition response:", data);

        if (!data.success) {

            result.innerText =
                data.message || "Attendance is not active.";

            return;
        }

        if (data.recognized && data.recognized.length > 0) {

            const students = data.recognized.map(function(student) {

                return "✓ " +
                    student.name +
                    " (" + student.roll_no + ") - " +
                    student.status;

            });

            result.innerText = [ ...new
            Set(students) ].join("|");

        } else {

            result.innerText =
                data.face_count > 0
                    ? "Face detected, but no registered student recognized."
                    : "No face detected. Please face the camera.";
        }

    } catch (error) {

        console.error("Recognition error:", error);

        result.innerText =
            "Recognition error: " + error.message;

    } finally {

        scanning = false;
    }
}

</script>

               

</body>

</html>
"""

ADMIN_HTML = """
<!DOCTYPE html>
<html>
<head>

    <title>Admin Panel</title>

    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <style>

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: Arial, sans-serif;
        }

        body {
            background: #f4f7fb;
            color: #1e293b;
        }

        .navbar {
            background: #0f172a;
            color: white;
            padding: 16px 30px;

            display: flex;
            justify-content: space-between;
            align-items: center;

            flex-wrap: wrap;
            gap: 15px;
        }

        .logo {
            font-size: 21px;
            font-weight: bold;
        }

        .links {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }

        .links a {
            color: white;
            text-decoration: none;
            padding: 9px 14px;
            background: #1e293b;
            border-radius: 8px;
            font-size: 14px;
        }

        .links a:hover {
            background: #2563eb;
        }

        .container {
            max-width: 1200px;
            margin: 30px auto;
            padding: 0 20px;
        }

        h1 {
            font-size: 28px;
            margin-bottom: 8px;
        }

        .subtitle {
            color: #64748b;
            margin-bottom: 25px;
        }

        .card {
            background: white;
            border-radius: 15px;
            padding: 22px;

            box-shadow: 0 5px 20px rgba(0,0,0,0.06);

            overflow-x: auto;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            min-width: 850px;
        }

        th {
            background: #0f172a;
            color: white;
            padding: 13px;
        }

        td {
            padding: 12px;
            text-align: center;
            border-bottom: 1px solid #e2e8f0;
        }

        tr:hover {
            background: #f8fafc;
        }

        .delete-btn {
            background: #dc2626;
            color: white;
            border: none;

            padding: 8px 13px;
            border-radius: 7px;

            cursor: pointer;
            font-weight: bold;
        }

        .delete-btn:hover {
            background: #b91c1c;
        }

        .student-count {
            background: #eff6ff;
            color: #1d4ed8;

            display: inline-block;

            padding: 8px 14px;

            border-radius: 20px;

            margin-bottom: 20px;

            font-weight: bold;
        }

        .empty {
            padding: 30px;
            color: #64748b;
            text-align: center;
        }

        @media (max-width: 700px) {

            .navbar {
                padding: 15px;
            }

            .container {
                padding: 0 12px;
            }
        }

    </style>

</head>

<body>


<div class="navbar">

    <div class="logo">
        AI Face Attendance - Admin
    </div>

    <div class="links">

        <a href="/dashboard">
            Dashboard
        </a>

        <a href="/register">
            Register Student
        </a>

        <a href="/attendance">
            Attendance
        </a>

        <a href="/report">
            Reports
        </a>

        <a href="/admin/logout">
            Logout
        </a>

    </div>

</div>


<div class="container">

    <h1>
        Admin Panel
    </h1>

    <p class="subtitle">
        Manage registered students and attendance records.
    </p>


    <div class="student-count">

        Total Registered Students:
        {{ users|length }}

    </div>


    <div class="card">

        <table>

            <thead>

                <tr>

                    <th>Sr. No.</th>

                    <th>Roll No.</th>

                    <th>Name</th>

                    <th>Email</th>

                    <th>Mobile</th>

                    <th>Registered On</th>

                    <th>Action</th>

                </tr>

            </thead>


            <tbody>

                {% if users %}

                    {% for user in users %}

                    <tr>

                        <td>
                            {{ loop.index }}
                        </td>

                        <td>
                            {{ user.roll_no }}
                        </td>

                        <td>
                            {{ user.name }}
                        </td>

                        <td>
                            {{ user.email }}
                        </td>

                        <td>
                            {{ user.mobile }}
                        </td>

                        <td>
                            {{ user.created_at }}
                        </td>

                        <td>

                            <form
                                method="POST"
                                action="/delete-student/{{ user.id }}"
                                onsubmit="return confirm(
                                    'Are you sure you want to delete this student?'
                                );"
                            >

                                <button
                                    type="submit"
                                    class="delete-btn"
                                >
                                    Delete
                                </button>

                            </form>

                        </td>

                    </tr>

                    {% endfor %}

                {% else %}

                    <tr>

                        <td
                            colspan="7"
                            class="empty"
                        >
                            No students registered yet.
                        </td>

                    </tr>

                {% endif %}

            </tbody>

        </table>

    </div>

</div>

</body>
</html>
"""


ADMIN_LOGIN_HTML = """
<!DOCTYPE html>
<html>
<head>

    <title>Admin Login</title>

    <meta
        name="viewport"
        content="width=device-width, initial-scale=1.0"
    >

    <style>

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: Arial, sans-serif;
        }

        body {

            min-height: 100vh;

            background:
                linear-gradient(
                    135deg,
                    #0f172a,
                    #1d4ed8
                );

            display: flex;

            justify-content: center;

            align-items: center;

            padding: 20px;
        }

        .login-card {

            width: 100%;
            max-width: 430px;

            background: white;

            padding: 35px;

            border-radius: 18px;

            box-shadow:
                0 20px 50px
                rgba(0,0,0,0.25);
        }

        .icon {

            width: 70px;
            height: 70px;

            background: #eff6ff;

            color: #2563eb;

            border-radius: 50%;

            display: flex;

            justify-content: center;

            align-items: center;

            margin: 0 auto 18px;

            font-size: 32px;
        }

        h1 {

            text-align: center;

            color: #0f172a;

            margin-bottom: 8px;
        }

        .subtitle {

            text-align: center;

            color: #64748b;

            margin-bottom: 28px;

            font-size: 14px;
        }

        .field {

            margin-bottom: 18px;
        }

        label {

            display: block;

            margin-bottom: 7px;

            font-size: 14px;

            font-weight: bold;

            color: #334155;
        }

        input {

            width: 100%;

            padding: 13px;

            border: 1px solid #cbd5e1;

            border-radius: 9px;

            font-size: 15px;

            outline: none;
        }

        input:focus {

            border-color: #2563eb;

            box-shadow:
                0 0 0 3px
                rgba(37,99,235,0.1);
        }

        .login-btn {

            width: 100%;

            border: none;

            padding: 13px;

            border-radius: 9px;

            background: #2563eb;

            color: white;

            font-size: 15px;

            font-weight: bold;

            cursor: pointer;
        }

        .login-btn:hover {

            background: #1d4ed8;
        }

        .error {

            background: #fee2e2;

            color: #b91c1c;

            padding: 11px;

            border-radius: 8px;

            margin-bottom: 18px;

            text-align: center;

            font-size: 14px;
        }

        .back {

            display: block;

            text-align: center;

            margin-top: 20px;

            color: #2563eb;

            text-decoration: none;

            font-size: 14px;
        }

        .back:hover {
            text-decoration: underline;
        }

    </style>

</head>

<body>


<div class="login-card">

    <div class="icon">
        🔐
    </div>


    <h1>
        Admin Login
    </h1>


    <p class="subtitle">
        AI Face Recognition Attendance System
    </p>


    {% if error %}

        <div class="error">
            {{ error }}
        </div>

    {% endif %}


    <form
        method="POST"
        action="/admin/login"
    >

        <div class="field">

            <label>
                Username
            </label>

            <input
                type="text"
                name="username"
                placeholder="Enter admin username"
                required
            >

        </div>


        <div class="field">

            <label>
                Password
            </label>

            <input
                type="password"
                name="password"
                placeholder="Enter admin password"
                required
            >

        </div>


        <button
            type="submit"
            class="login-btn"
        >
            Login
        </button>

    </form>


    <a
        href="/login"
        class="back"
    >
        ← Back to Login
    </a>

</div>


</body>
</html>
"""
# ============================================================
# FINAL ATTENDANCE LOGIC
# ============================================================

def finalize_completed_slots():
    now = datetime.now()

    # Sunday is a holiday
    if is_sunday(now.date()):
        return

    today = now.date().isoformat()

    conn = get_db()

    users = conn.execute(
        """
        SELECT *
        FROM users
        ORDER BY roll_no
        """
    ).fetchall()

    for slot_name, start_hour, end_hour in SLOTS:

        # Slot is not completed yet
        if now.hour < end_hour:
            continue

        for user in users:

            # If student was registered after the slot ended,
            # don't mark that student absent for the old slot.
            try:
                created_date = datetime.strptime(
                    user["created_at"],
                    "%Y-%m-%d %H:%M:%S"
                )

                slot_end = now.replace(
                    hour=end_hour,
                    minute=0,
                    second=0,
                    microsecond=0
                )

                if created_date > slot_end:
                    continue

            except Exception:
                pass

            row = conn.execute(
                """
                SELECT *
                FROM attendance
                WHERE user_id=?
                AND date=?
                AND slot=?
                """,
                (
                    user["id"],
                    today,
                    slot_name
                )
            ).fetchone()

            # No attendance record means student was never recognized
            if row is None:

                conn.execute(
                    """
                    INSERT INTO attendance
                    (
                        user_id,
                        date,
                        slot,
                        verification_count,
                        verification_cycles,
                        recognition_time,
                        status
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        user["id"],
                        today,
                        slot_name,
                        0,
                        "[]",
                        "",
                        "Absent"
                    )
                )

            # Pending record must be finalized
            elif row["status"] == "Pending":

                if row["verification_count"] >= REQUIRED_VERIFICATIONS:
                    final_status = "Present"
                else:
                    final_status = "Absent"

                conn.execute(
                    """
                    UPDATE attendance
                    SET status=?
                    WHERE id=?
                    """,
                    (
                        final_status,
                        row["id"]
                    )
                )

    conn.commit()
    conn.close()


# ============================================================
# IMPROVED FACE RECOGNITION
# ============================================================

def recognize_faces(frame):

    now = datetime.now()

    # --------------------------------------------------------
    # Sunday holiday
    # --------------------------------------------------------

    if is_sunday(now.date()):

        return {
            "success": False,
            "message": "Sunday Holiday - Attendance closed.",
            "recognized": [],
            "face_count": 0
        }


    # --------------------------------------------------------
    # Finalize previously completed slots
    # --------------------------------------------------------

    finalize_completed_slots()


    # --------------------------------------------------------
    # Check current slot
    # --------------------------------------------------------

    slot = current_slot(now)

    if slot is None:

        return {
            "success": False,
            "message": "Attendance is not active.",
            "recognized": [],
            "face_count": 0
        }


    # --------------------------------------------------------
    # Convert camera frame
    # --------------------------------------------------------

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------------------------
    # Detect all faces
    # --------------------------------------------------------

    locations = face_recognition.face_locations(
        rgb
    )


    if not locations:

        return {
            "success": True,
            "message": "No face detected.",
            "recognized": [],
            "face_count": 0,
            "slot": slot[0],
            "cycle": verification_cycle(now)
        }


    # --------------------------------------------------------
    # Generate encodings
    # --------------------------------------------------------

    encodings = face_recognition.face_encodings(
        rgb,
        locations
    )


    conn = get_db()


    users = conn.execute(
        """
        SELECT
            id,
            name,
            roll_no,
            face_encoding
        FROM users
        """
    ).fetchall()


    recognized = []

    cycle = verification_cycle(now)

    today = now.date().isoformat()


    # --------------------------------------------------------
    # Process every detected face
    # --------------------------------------------------------

    for encoding in encodings:

        best_user = None

        best_distance = 1.0


        # Compare with registered students
        for user in users:

            try:

                old_encoding = np.array(
                    json.loads(
                        user["face_encoding"]
                    )
                )


                distance = face_recognition.face_distance(
                    [old_encoding],
                    encoding
                )[0]


                if distance < best_distance:

                    best_distance = distance

                    best_user = user


            except Exception:

                continue


        # ----------------------------------------------------
        # Face not matched
        # ----------------------------------------------------

        if best_user is None:
            continue


        if best_distance > MATCH_TOLERANCE:
            continue


        # ----------------------------------------------------
        # Check existing attendance
        # ----------------------------------------------------

        row = conn.execute(
            """
            SELECT *
            FROM attendance
            WHERE user_id=?
            AND date=?
            AND slot=?
            """,
            (
                best_user["id"],
                today,
                slot[0]
            )
        ).fetchone()


        # ----------------------------------------------------
        # Student already Present
        # ----------------------------------------------------

        if row is not None and row["status"] == "Present":

            recognized.append(
                {
                    "name": best_user["name"],
                    "roll_no": best_user["roll_no"],
                    "status": "Present",
                    "cycle": cycle
                }
            )

            continue


        # ----------------------------------------------------
        # Student already Absent
        # ----------------------------------------------------

        if row is not None and row["status"] == "Absent":

            recognized.append(
                {
                    "name": best_user["name"],
                    "roll_no": best_user["roll_no"],
                    "status": "Absent",
                    "cycle": cycle
                }
            )

            continue


        # ----------------------------------------------------
        # First verification
        # ----------------------------------------------------

        if row is None:

            cycles = [cycle]

            count = 1

            status = "Pending"


            conn.execute(
                """
                INSERT INTO attendance
                (
                    user_id,
                    date,
                    slot,
                    verification_count,
                    verification_cycles,
                    recognition_time,
                    status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    best_user["id"],
                    today,
                    slot[0],
                    count,
                    json.dumps(cycles),
                    now.strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    status
                )
            )


        # ----------------------------------------------------
        # Existing Pending record
        # ----------------------------------------------------

        else:

            try:

                cycles = json.loads(
                    row["verification_cycles"] or "[]"
                )

            except Exception:

                cycles = []


            # Do not count the same cycle again
            if cycle not in cycles:

                cycles.append(cycle)


            count = len(cycles)


            if count >= REQUIRED_VERIFICATIONS:

                status = "Present"

            else:

                status = "Pending"


            conn.execute(
                """
                UPDATE attendance
                SET
                    verification_count=?,
                    verification_cycles=?,
                    recognition_time=?,
                    status=?
                WHERE id=?
                """,
                (
                    count,
                    json.dumps(cycles),
                    now.strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    status,
                    row["id"]
                )
            )


        recognized.append(
            {
                "name": best_user["name"],
                "roll_no": best_user["roll_no"],
                "status": status,
                "cycle": cycle
            }
        )


    conn.commit()

    conn.close()


    # --------------------------------------------------------
    # Final response
    # --------------------------------------------------------

    return {
        "success": True,
        "message": "Face recognition completed.",
        "recognized": recognized,
        "face_count": len(locations),
        "slot": slot[0],
        "cycle": cycle
    }


# ============================================================
# FINAL APP START
# ============================================================
REPORT_HTML = """
<!DOCTYPE html>
<html>

<head>

<meta charset="UTF-8">

<meta name="viewport"
content="width=device-width, initial-scale=1.0">

<title>Attendance Report</title>

<style>

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    font-family: Arial, Helvetica, sans-serif;
    background: #f4f7fb;
    color: #172033;
}

.navbar {
    background: linear-gradient(
        135deg,
        #0f172a,
        #1e3a8a
    );

    color: white;
    padding: 16px 28px;

    display: flex;
    justify-content: space-between;
    align-items: center;

    flex-wrap: wrap;
    gap: 12px;
}

.nav-title {
    font-size: 22px;
    font-weight: bold;
}

.nav-links {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
}

.nav-links a {
    color: white;
    text-decoration: none;
    padding: 9px 14px;
    border-radius: 8px;
    background: rgba(255,255,255,0.10);
}

.nav-links a:hover {
    background: rgba(255,255,255,0.20);
}

.container {
    width: 95%;
    max-width: 1400px;
    margin: 25px auto;
}

.header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 15px;
    margin-bottom: 20px;
}

.header h1 {
    margin: 0;
    font-size: 28px;
}

.card {
    background: white;
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 20px;

    box-shadow:
    0 5px 20px
    rgba(15,23,42,0.08);
}

.filters {
    display: flex;
    gap: 15px;
    align-items: end;
    flex-wrap: wrap;
}

.filter {
    display: flex;
    flex-direction: column;
    gap: 6px;
}

.filter label {
    font-size: 13px;
    font-weight: bold;
    color: #475569;
}

input {
    padding: 11px 13px;
    border: 1px solid #d6dce5;
    border-radius: 9px;
    font-size: 14px;
}

button {
    padding: 11px 18px;
    border: none;
    border-radius: 9px;
    background: #2563eb;
    color: white;
    font-weight: bold;
    cursor: pointer;
}

.print-btn {
    background: #0f172a;
}

.table-wrapper {
    overflow-x: auto;
}

table {
    width: 100%;
    border-collapse: collapse;
    min-width: 750px;
}

th {
    background: #0f172a;
    color: white;
    padding: 13px;
    text-align: center;
}

td {
    padding: 12px;
    border-bottom: 1px solid #e5e7eb;
    text-align: center;
}

td:nth-child(1),
td:nth-child(2) {
    text-align: left;
}

.present {
    color: #15803d;
    font-weight: bold;
}

.absent {
    color: #dc2626;
    font-weight: bold;
}

.pending {
    color: #d97706;
    font-weight: bold;
}

.percentage {
    font-weight: bold;
    color: #2563eb;
}

@media(max-width:600px) {

    .container {
        width: 92%;
    }

    .header h1 {
        font-size: 22px;
    }

}

@media print {

    body {
        background: white;
    }

    .navbar,
    .filters,
    .print-btn {
        display: none !important;
    }

    .card {
        box-shadow: none;
    }

}

</style>

</head>

<body>

<div class="navbar">

    <div class="nav-title">
        🤖 AI Face Attendance
    </div>

    <div class="nav-links">

        <a href="{{ url_for('dashboard') }}">
            Dashboard
        </a>

        <a href="{{ url_for('attendance') }}">
            Attendance
        </a>

        <a href="{{ url_for('report') }}">
            Reports
        </a>

        <a href="{{ url_for('admin') }}">
            Students
        </a>

        <a href="{{ url_for('logout') }}">
            Logout
        </a>

    </div>

</div>

<div class="container">

<div class="header">

    <div>

        <h1>
            📊 Attendance Report
        </h1>

        <p>
            Student-wise attendance summary
        </p>

    </div>

    <button
        class="print-btn"
        onclick="window.print()">

        🖨 Print Report

    </button>
    <a
    href="{{ url_for(
        'export_csv',
        from_date=from_date,
        to_date=to_date
    ) }}"
    style="
        display:inline-block;
        padding:11px 18px;
        border-radius:9px;
        background:#16a34a;
        color:white;
        text-decoration:none;
        font-weight:bold;
        margin-left:8px;
    ">

    📥 Export CSV

</a>

</div>

<div class="card">

    <form
        method="GET"
        action="{{ url_for('report') }}">

        <div class="filters">

            <div class="filter">

                <label>
                    From Date
                </label>

                <input
                    type="date"
                    name="from_date"
                    value="{{ from_date }}">

            </div>

            <div class="filter">

                <label>
                    To Date
                </label>

                <input
                    type="date"
                    name="to_date"
                    value="{{ to_date }}">

            </div>

            <button type="submit">
                View Report
            </button>

        </div>

    </form>

</div>

<div class="card">

    <h2>
        Attendance Summary
    </h2>

    <p>

        From:
        <strong>
            {{ from_date }}
        </strong>

        &nbsp;&nbsp;

        To:
        <strong>
            {{ to_date }}
        </strong>

    </p>

    <div class="table-wrapper">

        <table>

            <thead>

                <tr>

                    <th>
                        Roll No
                    </th>

                    <th>
                        Student Name
                    </th>

                    <th>
                        Present
                    </th>

                    <th>
                        Absent
                    </th>

                    <th>
                        Pending
                    </th>

                    <th>
                        Attendance %
                    </th>

                </tr>

            </thead>

            <tbody>

            {% for student in report_data %}

                <tr>

                    <td>
                        {{ student.roll_no }}
                    </td>

                    <td>
                        {{ student.name }}
                    </td>

                    <td class="present">
                        {{ student.present }}
                    </td>

                    <td class="absent">
                        {{ student.absent }}
                    </td>

                    <td class="pending">
                        {{ student.pending }}
                    </td>

                    <td class="percentage">
                        {{ student.percentage }}%
                    </td>

                </tr>

            {% endfor %}

            </tbody>

        </table>

    </div>

</div>

</div>

</body>

</html>
"""
ADMIN_HTML = """
<!DOCTYPE html>
<html>

<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<title>Student Management</title>

<style>

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    font-family: Arial, sans-serif;
    background: #f4f7fb;
    color: #172033;
}

.navbar {
    background: linear-gradient(135deg, #0f172a, #1e3a8a);
    color: white;
    padding: 16px 28px;

    display: flex;
    justify-content: space-between;
    align-items: center;

    flex-wrap: wrap;
    gap: 12px;
}

.nav-title {
    font-size: 22px;
    font-weight: bold;
}

.nav-links {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
}

.nav-links a {
    color: white;
    text-decoration: none;
    padding: 9px 14px;
    border-radius: 8px;
    background: rgba(255,255,255,0.10);
}

.nav-links a:hover {
    background: rgba(255,255,255,0.20);
}

.container {
    width: 95%;
    max-width: 1400px;
    margin: 25px auto;
}

.header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 15px;
    margin-bottom: 20px;
}

.header h1 {
    margin: 0;
    font-size: 28px;
}

.card {
    background: white;
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 20px;

    box-shadow: 0 5px 20px rgba(15,23,42,0.08);
}

.stats {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 15px;
    margin-bottom: 20px;
}

.stat {
    background: white;
    border-radius: 14px;
    padding: 20px;
    box-shadow: 0 5px 20px rgba(15,23,42,0.08);
}

.stat-title {
    color: #64748b;
    font-size: 14px;
}

.stat-value {
    font-size: 28px;
    font-weight: bold;
    margin-top: 8px;
    color: #2563eb;
}

.button {
    display: inline-block;
    padding: 11px 18px;
    border-radius: 9px;
    background: #2563eb;
    color: white;
    text-decoration: none;
    font-weight: bold;
}

.button:hover {
    background: #1d4ed8;
}

.table-wrapper {
    overflow-x: auto;
}

table {
    width: 100%;
    border-collapse: collapse;
    min-width: 850px;
}

th {
    background: #0f172a;
    color: white;
    padding: 13px;
    text-align: center;
}

td {
    padding: 12px;
    border-bottom: 1px solid #e5e7eb;
    text-align: center;
}

.student-name {
    font-weight: bold;
    text-align: left;
}

.delete-btn {
    background: #dc2626;
    color: white;
    border: none;
    padding: 8px 12px;
    border-radius: 7px;
    cursor: pointer;
}

.delete-btn:hover {
    background: #b91c1c;
}

.register-btn {
    background: #16a34a;
}

.register-btn:hover {
    background: #15803d;
}

.badge {
    padding: 5px 10px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: bold;
    background: #dcfce7;
    color: #15803d;
}

@media(max-width:800px) {

    .stats {
        grid-template-columns: 1fr;
    }

    .header h1 {
        font-size: 22px;
    }

}

@media print {

    .navbar,
    .button,
    .delete-btn {
        display: none !important;
    }

    body {
        background: white;
    }

    .card,
    .stat {
        box-shadow: none;
    }

}

</style>

</head>

<body>

<div class="navbar">

    <div class="nav-title">
        🤖 AI Face Attendance
    </div>

    <div class="nav-links">

        <a href="{{ url_for('dashboard') }}">
            Dashboard
        </a>

        <a href="{{ url_for('attendance') }}">
            Attendance
        </a>

        <a href="{{ url_for('report') }}">
            Reports
        </a>

        <a href="{{ url_for('admin') }}">
            Students
        </a>

        <a href="{{ url_for('logout') }}">
            Logout
        </a>

    </div>

</div>


<div class="container">


<div class="header">

    <div>

        <h1>
            👨‍🎓 Student Management
        </h1>

        <p>
            Manage registered students and face data
        </p>

    </div>

    <div>

        <a
            href="{{ url_for('register') }}"
            class="button register-btn">

            ➕ Register Student

        </a>
        <a
    href="{{ url_for('change_password') }}"
    class="button">

    🔐 Change Password

</a>

        <button
            onclick="window.print()"
            class="button">

            🖨 Print

        </button>

    </div>

</div>


<div class="stats">

    <div class="stat">

        <div class="stat-title">
            Total Students
        </div>

        <div class="stat-value">
            {{ users|length }}
        </div>

    </div>


    <div class="stat">

        <div class="stat-title">
            Face Recognition
        </div>

        <div class="stat-value">
            AI Active
        </div>

    </div>


    <div class="stat">

        <div class="stat-title">
            System Status
        </div>

        <div class="stat-value">
            Online
        </div>

    </div>

</div>


<div class="card">

    <h2>
        Registered Students
    </h2>

    <div class="table-wrapper">

        <table>

            <thead>

                <tr>

                    <th>
                        #
                    </th>

                    <th>
                        Roll No
                    </th>

                    <th>
                        Student Name
                    </th>

                    <th>
                        Email
                    </th>

                    <th>
                        Mobile
                    </th>

                    <th>
                        Registered On
                    </th>

                    <th>
                        Face
                    </th>

                    <th>
                        Action
                    </th>

                </tr>

            </thead>

            <tbody>

            {% for user in users %}

                <tr>

                    <td>
                        {{ loop.index }}
                    </td>

                    <td>
                        {{ user.roll_no }}
                    </td>

                    <td class="student-name">
                        {{ user.name }}
                    </td>

                    <td>
                        {{ user.email }}
                    </td>

                    <td>
                        {{ user.mobile }}
                    </td>

                    <td>
                        {{ user.created_at }}
                    </td>

                    <td>

                        <span class="badge">
                            ✓ Registered
                        </span>

                    </td>

                    <td>

                        <form
                            method="POST"
                            action="{{ url_for(
                                'delete_student',
                                user_id=user.id
                            ) }}"
                            onsubmit="return confirm(
                                'Are you sure you want to delete this student?'
                            );">

                            <button
                                type="submit"
                                class="delete-btn">

                                🗑 Delete

                            </button>

                        </form>

                    </td>

                </tr>

            {% else %}

                <tr>

                    <td colspan="8">

                        No students registered yet.

                    </td>

                </tr>

            {% endfor %}

            </tbody>

        </table>

    </div>

</div>


</div>

</body>

</html>
"""
@app.route("/change-password", methods=["GET", "POST"])
@login_required
def change_password():

    message = ""
    error = ""

    if request.method == "POST":

        old_password = request.form.get(
            "old_password", ""
        )

        new_password = request.form.get(
            "new_password", ""
        )

        confirm_password = request.form.get(
            "confirm_password", ""
        )

        if not old_password or not new_password or not confirm_password:

            error = "All fields are required."

        elif len(new_password) < 6:

            error = "New password must contain at least 6 characters."

        elif new_password != confirm_password:

            error = "New password and confirm password do not match."

        else:

            conn = get_db()

            admin = conn.execute(
                """
                SELECT *
                FROM admins
                WHERE id=?
                """,
                (session["admin_id"],)
            ).fetchone()

            if admin and check_password_hash(
                admin["password_hash"],
                old_password
            ):

                new_hash = generate_password_hash(
                    new_password
                )

                conn.execute(
                    """
                    UPDATE admins
                    SET password_hash=?
                    WHERE id=?
                    """,
                    (
                        new_hash,
                        session["admin_id"]
                    )
                )

                conn.commit()

                message = "Password changed successfully."

            else:

                error = "Current password is incorrect."

            conn.close()

    return render_template_string(
        CHANGE_PASSWORD_HTML,
        message=message,
        error=error
    )
CHANGE_PASSWORD_HTML = """
<!DOCTYPE html>
<html>

<head>

<meta charset="UTF-8">

<meta name="viewport"
content="width=device-width, initial-scale=1.0">

<title>Change Admin Password</title>

<style>

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    font-family: Arial, sans-serif;
    background: #f4f7fb;
    color: #172033;
}

.navbar {
    background: linear-gradient(
        135deg,
        #0f172a,
        #1e3a8a
    );

    color: white;
    padding: 16px 28px;

    display: flex;
    justify-content: space-between;
    align-items: center;

    flex-wrap: wrap;
    gap: 12px;
}

.nav-title {
    font-size: 22px;
    font-weight: bold;
}

.nav-links {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
}

.nav-links a {
    color: white;
    text-decoration: none;
    padding: 9px 14px;
    border-radius: 8px;
    background: rgba(255,255,255,0.10);
}

.nav-links a:hover {
    background: rgba(255,255,255,0.20);
}

.container {
    width: 92%;
    max-width: 550px;
    margin: 50px auto;
}

.card {
    background: white;
    border-radius: 18px;
    padding: 30px;

    box-shadow:
    0 8px 30px
    rgba(15,23,42,0.10);
}

h1 {
    margin-top: 0;
    color: #0f172a;
}

.subtitle {
    color: #64748b;
    margin-bottom: 25px;
}

.form-group {
    margin-bottom: 18px;
}

label {
    display: block;
    margin-bottom: 7px;
    font-weight: bold;
    color: #334155;
}

input {
    width: 100%;
    padding: 12px 14px;

    border: 1px solid #d6dce5;
    border-radius: 9px;

    font-size: 15px;
}

input:focus {
    outline: none;
    border-color: #2563eb;
}

button {
    width: 100%;
    padding: 13px;

    border: none;
    border-radius: 9px;

    background: #2563eb;
    color: white;

    font-size: 15px;
    font-weight: bold;

    cursor: pointer;
}

button:hover {
    background: #1d4ed8;
}

.success {
    background: #dcfce7;
    color: #15803d;

    padding: 12px;
    border-radius: 8px;

    margin-bottom: 18px;
}

.error {
    background: #fee2e2;
    color: #dc2626;

    padding: 12px;
    border-radius: 8px;

    margin-bottom: 18px;
}

.back {
    display: block;
    text-align: center;

    margin-top: 18px;

    color: #2563eb;
    text-decoration: none;
}

.back:hover {
    text-decoration: underline;
}

</style>

</head>

<body>


<div class="navbar">

    <div class="nav-title">
        🤖 AI Face Attendance
    </div>

    <div class="nav-links">

        <a href="{{ url_for('dashboard') }}">
            Dashboard
        </a>

        <a href="{{ url_for('attendance') }}">
            Attendance
        </a>

        <a href="{{ url_for('report') }}">
            Reports
        </a>

        <a href="{{ url_for('admin') }}">
            Students
        </a>

        <a href="{{ url_for('logout') }}">
            Logout
        </a>

    </div>

</div>


<div class="container">

<div class="card">

    <h1>
        🔐 Change Admin Password
    </h1>

    <p class="subtitle">
        Update your administrator account password.
    </p>


    {% if message %}

    <div class="success">
        ✓ {{ message }}
    </div>

    {% endif %}


    {% if error %}

    <div class="error">
        ⚠ {{ error }}
    </div>

    {% endif %}


    <form
        method="POST"
        action="{{ url_for('change_password') }}">


        <div class="form-group">

            <label>
                Current Password
            </label>

            <input
                type="password"
                name="old_password"
                placeholder="Enter current password"
                required>

        </div>


        <div class="form-group">

            <label>
                New Password
            </label>

            <input
                type="password"
                name="new_password"
                placeholder="Enter new password"
                minlength="6"
                required>

        </div>


        <div class="form-group">

            <label>
                Confirm New Password
            </label>

            <input
                type="password"
                name="confirm_password"
                placeholder="Confirm new password"
                minlength="6"
                required>

        </div>


        <button type="submit">

            🔑 Change Password

        </button>

    </form>


    <a
        href="{{ url_for('admin') }}"
        class="back">

        ← Back to Student Management

    </a>

</div>

</div>


</body>

</html>
"""
@app.route("/export-csv")
@login_required
def export_csv():

    import csv
    from io import StringIO
    from flask import Response

    from_date = request.args.get(
        "from_date",
        datetime.now().strftime("%Y-%m-%d")
    )

    to_date = request.args.get(
        "to_date",
        datetime.now().strftime("%Y-%m-%d")
    )

    conn = get_db()

    users = conn.execute(
        """
        SELECT *
        FROM users
        ORDER BY roll_no
        """
    ).fetchall()

    records = conn.execute(
        """
        SELECT *
        FROM attendance
        WHERE date BETWEEN ? AND ?
        ORDER BY date, slot
        """,
        (from_date, to_date)
    ).fetchall()

    conn.close()

    output = StringIO()

    writer = csv.writer(output)

    writer.writerow([
        "Roll No",
        "Student Name",
        "Email",
        "Mobile",
        "Date",
        "Slot",
        "Status",
        "Verification Count",
        "Recognition Time"
    ])

    user_map = {}

    for user in users:
        user_map[user["id"]] = user

    for record in records:

        day = datetime.strptime(
            record["date"],
            "%Y-%m-%d"
        )

        # Sunday holiday exclude
        if day.weekday() == 6:
            continue

        user = user_map.get(
            record["user_id"]
        )

        if user is None:
            continue

        writer.writerow([
            user["roll_no"],
            user["name"],
            user["email"],
            user["mobile"],
            record["date"],
            record["slot"],
            record["status"],
            record["verification_count"],
            record["recognition_time"]
        ])

    response = Response(
        output.getvalue(),
        mimetype="text/csv"
    )

    response.headers[
        "Content-Disposition"
    ] = (
        "attachment; "
        "filename=attendance_report_"
        + from_date
        + "_to_"
        + to_date
        + ".csv"
    )

    return response
if __name__ == "__main__":
    
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )