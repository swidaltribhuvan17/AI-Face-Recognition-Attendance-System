
AI FACE RECOGNITION ATTENDANCE SYSTEM
====================================

1. PROJECT DESCRIPTION
----------------------
The AI Face Recognition Attendance System is a web-based application
developed using Python, Flask, OpenCV, face-recognition and SQLite.

The system identifies registered students using facial recognition
and records their attendance.

2. MAIN FEATURES
----------------
- Admin login
- Student registration
- Face recognition
- Attendance recording
- Hourly attendance slots
- Daily attendance report
- Monthly attendance report
- Student-wise attendance
- CSV report export, if enabled in the application

3. PROJECT STRUCTURE
--------------------
AI_Face_Recognition_Attendance_System/
    app.py
    attendance.db
    requirements.txt
    README.txt

4. SOFTWARE REQUIREMENTS
------------------------
- Python 3.11
- Flask
- OpenCV
- NumPy
- face-recognition
- dlib-bin
- Pillow
- Gunicorn for online hosting
- Google Chrome or another modern browser

5. RUN THE PROJECT LOCALLY
--------------------------
Open a terminal in the project folder.

Install the required libraries:

    python -m pip install -r requirements.txt

Start the application:

    python app.py

Open this address in your browser:

    http://127.0.0.1:5000

6. ONLINE DEPLOYMENT
--------------------
The project can be deployed using GitHub and Render.

Build command:

    pip install -r requirements.txt

Start command:

    gunicorn app:app

7. CAMERA REQUIREMENTS
----------------------
The browser needs permission to use the camera.
A public website should use HTTPS for browser camera access.

The server must have the required face-recognition libraries
installed to process the images received from the browser.

8. DATABASE AND PRIVACY
-----------------------
SQLite is used to store student and attendance information.

Do not publish real student names, mobile numbers, face encodings,
passwords or attendance records in a public repository.

A free hosting service may not preserve changes made to a local
SQLite database after restarts or redeployments.

9. IMPORTANT NOTE
-----------------
This project is intended for educational demonstration.
Test all functions after deployment before using it for real
attendance records.
