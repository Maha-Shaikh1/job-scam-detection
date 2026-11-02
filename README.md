# JobShield – Job Scam Detection System

**Course:** MAAI902 – Natural Language Processing & Large Language Models  
**University:** University of Wollongong in Dubai (UOWD)  
**Student:** SHAIKH MAHA | **Student ID:** 1560980 

## About the Project
JobShield is a web-based system that detects fraudulent job postings using Natural Language Processing and Machine Learning. Users can paste a job description and the system predicts whether it is **Real** or **Fake**, along with a confidence score. It also includes user login/registration, an admin dashboard with reports, and resume/job-hunting safety tips.

## Features
- Job posting scam detection (text + numeric features)
- User registration, login and password recovery
- Admin panel – manage users, view reports and dashboard
- Resume tips and safety guidelines pages
- Results page with prediction and confidence score

## Tech Stack
- **Backend:** Python, Flask
- **ML/NLP:** scikit-learn, TF-IDF, pickle models
- **Database:** SQLite (database.db, jobshield.db)
- **Frontend:** HTML, CSS, JavaScript

## Project Structure
- `app.py` – main Flask application
- `train_model.py` – trains the ML model from the dataset
- `database.py` – database setup and queries
- `make_admin.py` – creates an admin account
- `admin_*.py / admin_*.html` – admin pages
- `*.html` – frontend pages (index, login, register, result, etc.)
- `script.js`, `static/` – frontend scripts and images
- `job_scam_model.pkl`, `numeric_model.pkl`, `scam_threshold.pkl` – trained models

## Dataset
Dataset: *Real or Fake – Fake Job Posting Prediction* (Kaggle, 17,880 job postings). It is included as `fake_job_postings.zip` – extract it in the project folder before running `train_model.py`.

**Note:** Pre-trained models (`.pkl`) are already included, so the app runs directly without retraining.

## How to Run
1. Install Python 3.8+
2. Install dependencies: `pip install -r requirements.txt`
3. Open the project folder in VS Code and open **two terminals**:

   **Terminal 1 – Backend (API):**
   `python app.py`
   (runs on http://127.0.0.1:5000)

   **Terminal 2 – Frontend:**
   `python -m http.server 5500`

4. Open the browser at: http://127.0.0.1:5500/index.html

## Admin Access
Run `python make_admin.py` to create an admin account, then log in through the admin login page.

## Author
Shaikh Maha – MAAI902, UOWD, 2026
