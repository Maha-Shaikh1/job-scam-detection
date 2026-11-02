import os
import pickle
import sqlite3
import re
import base64
import hashlib

from io import BytesIO
from datetime import timedelta

from flask import Flask, request, jsonify, session, Response
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.graphics.shapes import Drawing, Wedge, Circle, String, Rect, Line
from reportlab.graphics import renderPDF
import math

# Optional imports for image/PDF processing
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# Try easyocr first (no system install needed), fallback to tesseract
HAS_OCR = False
OCR_ENGINE = None
try:
    import easyocr
    _ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    HAS_OCR    = True
    OCR_ENGINE = "easyocr"
    print("✅ EasyOCR available")
except ImportError:
    try:
        import pytesseract
        HAS_OCR    = True
        OCR_ENGINE = "tesseract"
        print("✅ Tesseract OCR available")
    except ImportError:
        print("⚠️  No OCR engine found. Run: pip install easyocr")

try:
    import fitz  # PyMuPDF
    HAS_PDF = True
    print("✅ PyMuPDF available")
except ImportError:
    HAS_PDF = False
    print("⚠️  PyMuPDF not installed — run: pip install PyMuPDF")

# =========================================================
# APP CONFIG
# =========================================================

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "jobshield-secret-2026")

app.config.update(
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=False,
    PERMANENT_SESSION_LIFETIME=timedelta(hours=5),
    SESSION_COOKIE_HTTPONLY=True,
)
app.config["SESSION_COOKIE_DOMAIN"] = False

# ── FIX: Accept requests from all common local dev origins ──
ALLOWED_ORIGINS = [
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    "http://127.0.0.1:5501",
    "http://localhost:5501",
    "http://127.0.0.1:3000",
    "http://localhost:3000",
    "http://127.0.0.1:8080",
    "http://localhost:8080",
]

CORS(
    app,
    supports_credentials=True,
    origins=ALLOWED_ORIGINS,
    allow_headers=["Content-Type", "X-Requested-With"],
    methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
)

# =========================================================
# ADMIN SECURITY (ENV + LOCKOUT)
# =========================================================

ADMIN_USER = os.environ.get("ADMIN_USER", "admin")
ADMIN_PASS = os.environ.get("ADMIN_PASS", "admin123")  # ← FIX: default fallback so admin always exists in dev

MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_MINUTES = 5

# =========================================================
# LOAD MODEL BUNDLE
# =========================================================

with open("job_scam_model.pkl", "rb") as f:
    _bundle = pickle.load(f)

text_model       = _bundle["text_pipeline"]
numeric_model    = _bundle["numeric_pipeline"]
NUMERIC_FEATURES = _bundle["numeric_features"]
SCAM_KEYWORDS    = _bundle["scam_keywords"]
SCAM_THRESHOLD   = _bundle["threshold"]
ENSEMBLE_WEIGHTS = _bundle["ensemble_weights"]
FRAUD_CLASS_IDX  = _bundle["fraud_class_index"]
NUM_FRAUD_IDX    = _bundle["num_fraud_index"]

print(f"✅ Model bundle loaded | Threshold: {SCAM_THRESHOLD}")

# =========================================================
# DATABASE INIT
# =========================================================

def init_db():
    conn = sqlite3.connect("database.db")
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT,
        role TEXT DEFAULT 'user',
        security_question TEXT,
        security_answer TEXT
    )""")

    c.execute("""
    CREATE TABLE IF NOT EXISTS dashboard_reports(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        job_text TEXT,
        prediction TEXT,
        confidence REAL,
        explanation TEXT,
        file_hash TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )""")

    # ── Migration: add file_hash column if upgrading existing DB ──
    try:
        c.execute("ALTER TABLE dashboard_reports ADD COLUMN file_hash TEXT")
        conn.commit()
    except sqlite3.OperationalError:
        pass  # column already exists

    c.execute("""
    CREATE TABLE IF NOT EXISTS admin_reports(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        message TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )""")

    # Track login failures to block brute-force
    c.execute("""
    CREATE TABLE IF NOT EXISTS login_attempts(
        username TEXT PRIMARY KEY,
        fail_count INTEGER DEFAULT 0,
        locked_until TEXT,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )""")

    # Default admin account (ENV-based password)
    c.execute("SELECT id FROM users WHERE username=?", (ADMIN_USER,))
    if not c.fetchone():
        if not ADMIN_PASS:
            print("⚠️ ADMIN_PASS environment variable not set. Admin will NOT be created.")
        else:
            c.execute("""
                INSERT INTO users(username, password, role, security_question, security_answer)
                VALUES (?, ?, ?, ?, ?)
            """, (
                ADMIN_USER,
                generate_password_hash(ADMIN_PASS),
                "admin",
                "Admin secret?",
                generate_password_hash("jobshield")
            ))
            print(f"✅ Admin created: {ADMIN_USER} / {ADMIN_PASS}")

    conn.commit()
    conn.close()

init_db()

# =========================================================
# HELPERS
# =========================================================

def extract_numeric_features(job_text: str, extra: dict) -> list:
    text_lower      = job_text.lower()
    has_logo        = float(extra.get("has_logo", 0))
    telecommuting   = float(extra.get("telecommuting", 0))
    scam_kw_count   = sum(1 for kw in SCAM_KEYWORDS if kw in text_lower)
    has_phone       = float(bool(re.search(r'\b\d{10}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b', job_text)))
    is_short_desc   = float(len(job_text) < 200)
    no_company      = float(len(str(extra.get("company_profile", "")).strip()) == 0)
    no_requirements = float(len(str(extra.get("requirements", "")).strip()) == 0)
    has_salary_info = float(len(str(extra.get("salary_range", "")).strip()) > 0)
    return [has_logo, telecommuting, scam_kw_count, has_phone,
            is_short_desc, no_company, no_requirements, has_salary_info]


def heuristic_boost(job_text: str, base_score: float):
    text   = job_text.lower()
    boost  = 0.0
    flags  = []

    has_email   = bool(re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', job_text))
    has_website = bool(re.search(r'www\.|http|\.com|\.in|\.org', job_text))
    has_official_contact = has_email or has_website

    no_skills       = any(x in text for x in ['no skills', 'no experience', 'no qualification', 'unskilled', 'freshers only'])
    urgent          = any(x in text for x in ['urgent', 'immediately', 'limited seats', 'hurry', 'apply fast', 'last date'])
    vague_training  = any(x in text for x in ['training provided', 'training session', 'will train', 'we will teach', 'training given'])
    free_stuff      = any(x in text for x in ['free visa', 'free ticket', 'free accommodation', 'free food', 'free stay'])
    fee_required    = any(x in text for x in ['processing fee', 'registration fee', 'security deposit', 'pay to join', 'joining fee'])
    gender_target   = any(x in text for x in ['only for women', 'only for men', 'housewives', 'female candidates only', 'male candidates only'])
    unrealistic_sal = bool(re.search(r'(\$\s*\d{4,}|\d{4,}\s*(?:per day|daily|\/day|per hour|\/hour|hour:|hrs?))', text))

    if no_skills and urgent and not has_official_contact:
        boost += 0.15
        flags.append("no qualifications + urgency tactics without official contact")
    if free_stuff:
        boost += 0.20
        flags.append("unrealistic free benefits (visa/ticket/accommodation)")
    if fee_required and not has_official_contact:
        boost += 0.25
        flags.append("upfront fee demanded with no official contact")
    if vague_training and no_skills and not has_official_contact:
        boost += 0.12
        flags.append("vague training promise with no skills needed")
    if gender_target and no_skills and urgent:
        boost += 0.12
        flags.append("targeted gender hiring with no skills + urgency")

    # ── BUG FIX: has_brand_name moved outside unrealistic_sal block ──
    if unrealistic_sal and not has_official_contact:
        boost += 0.15
        flags.append("unrealistic salary claim with no official contact")

    has_brand_name = bool(re.search(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)?\b', job_text))
    if has_brand_name and not has_official_contact:
        boost += 0.25
        flags.append("Brand name used but no official contact found — possible impersonation")

    if re.search(r'\d+\s*task\s*=', text):
        boost += 0.25
        flags.append("task-based earning scheme detected")
    if any(x in text for x in ["same day pay", "daily payment", "instant payment"]):
        boost += 0.15
        flags.append("instant payment promise detected")



    # ── NEW: WhatsApp / Telegram link detection ──
    if re.search(r'wa\.me\/|whatsapp\.com\/|t\.me\/|bit\.ly|tinyurl\.com', text):
        boost += 0.15
        flags.append("Suspicious link found (WhatsApp/Telegram/shortened URL) — common in scam recruitment")

    # ── NEW: Urgency language pattern (NLP-style regex) ──
    urgency_patterns = [
        r'limited\s+seats?\s+(available|left|only)',
        r'apply\s+(today|now|immediately|fast|asap)',
        r'last\s+\d+\s+(seats?|spots?|positions?)',
        r'offer\s+expires?\s+(today|soon|midnight)',
        r'only\s+\d+\s+(vacancy|vacancies|openings?)\s+left',
        r'hiring\s+closes?\s+(today|tomorrow|soon)',
    ]
    urgency_found = [p for p in urgency_patterns if re.search(p, text)]
    if len(urgency_found) >= 2:
        boost += 0.12
        flags.append("Multiple urgency pressure tactics detected — designed to rush you into applying")

    # ── NEW: Salary sanity check (Indian context — ₹5000+/day is suspicious) ──
    sal_match = re.search(r'(\d+)\s*(?:rs\.?|₹|inr)?\s*(?:per day|\/day|daily)', text, re.IGNORECASE)
    if sal_match:
        daily_amt = int(sal_match.group(1))
        if daily_amt >= 3000:
            boost += 0.15
            flags.append(f"Unrealistic daily salary (₹{daily_amt}/day) — genuine entry-level jobs do not pay this much daily")

    # ── NEW: Hinglish scam pattern (common in Indian WhatsApp job scams) ──
    hinglish_scam = [
        r'ghar\s+baithe\s+(kama|earn|paise)',
        r'(kaam|kam)\s+(karo|karen).{0,30}(paise|paisa|income)',
        r'(daily|roz)\s+(paise|paisa|earning|income)',
        r'sirf\s+\d+\s+(ghante|hour).{0,20}(earn|kama|income)',
        r'(part\s*time|parttime).{0,30}(ghar|घर|home).{0,30}(kama|earn)',
    ]
    hinglish_found = [p for p in hinglish_scam if re.search(p, text, re.IGNORECASE)]
    if hinglish_found:
        boost += 0.15
        flags.append("Hinglish scam pattern detected — 'ghar baithe kamao' style posts are commonly fraudulent")

    boosted = min(base_score + boost, 0.99)
    return boosted, flags


MANDATORY_JOB_KEYWORDS = [
    "hiring", "vacancy", "apply now", "apply here", "job opening",
    "we are hiring", "urgently hiring", "walk in interview", "walk-in",
    "job description", "offer letter", "appointment letter", "joining letter",
    "salary offered", "ctc", "lpa", "per month salary", "fixed salary",
    "job profile", "designation", "fresher apply", "freshers wanted",
    "candidates required", "candidates needed", "recruitment",
    "job vacancy", "required candidate", "wanted candidate",
    "immediate joining", "job apply", "resume send", "cv send",
    "send resume", "send cv", "whatsapp your cv", "call for interview",
    "selected candidates", "shortlisted candidates", "interview date",
    "joining date", "offer accepted", "hr contact", "hr email",
    "jobs in", "opening for", "opening at", "position open",
    "post:", "post :", "role:", "role :", "job title"
]

SUPPORTING_JOB_KEYWORDS = [
    "salary", "experience required", "qualification required",
    "part time", "full time", "work from home", "remote job",
    "internship", "earn monthly", "earn daily", "income",
    "no experience needed", "training provided", "job opportunity",
    "limited seats", "urgent requirement", "immediate requirement"
]

def is_job_related(text: str) -> bool:
    text_lower = text.lower()
    broad_indicators = [
        "job", "hiring", "vacancy", "apply", "salary", "ctc",
        "earn", "per day", "per month", "work from home",
        "interview", "joining", "position", "role",
        "candidate", "recruitment", "offer letter",
        "appointment letter", "hr", "resume", "cv",
        "opening", "required", "urgent"
    ]
    broad_score = sum(1 for kw in broad_indicators if kw in text_lower)
    mandatory_count = sum(1 for kw in MANDATORY_JOB_KEYWORDS if kw in text_lower)
    supporting_count = sum(1 for kw in SUPPORTING_JOB_KEYWORDS if kw in text_lower)
    if broad_score >= 2:
        return True
    if mandatory_count >= 1 and (mandatory_count >= 2 or supporting_count >= 1):
        return True
    return False


def predict_scam(job_text: str, extra: dict) -> dict:
    text_prob   = text_model.predict_proba([job_text])[0][FRAUD_CLASS_IDX]
    num_vec     = extract_numeric_features(job_text, extra)
    num_prob    = numeric_model.predict_proba([num_vec])[0][NUM_FRAUD_IDX]
    base_score  = text_prob * ENSEMBLE_WEIGHTS["text"] + num_prob * ENSEMBLE_WEIGHTS["numeric"]

    final_score, h_flags = heuristic_boost(job_text, base_score)

    if final_score >= 0.56:
        label = "Scam Job"
        confidence = final_score
    elif final_score >= 0.26:
        label = "Doubtful"
        confidence = final_score
    else:
        label = "Real Job"
        confidence = 1.0 - final_score

    text_lower = job_text.lower()

    scam_signals   = []
    legit_signals  = []

    if any(kw in text_lower for kw in ["guaranteed", "earn", "per day", "easy money", "daily payment", "weekly payment", "guaranteed income"]):
        found = [kw for kw in ["guaranteed","earn","per day","easy money","daily payment","weekly payment","guaranteed income"] if kw in text_lower]
        scam_signals.append(f"🔴 Unrealistic Earning Claims — Words like '{', '.join(found)}' were found. Genuine employers never guarantee fixed daily/weekly income. This is a classic trick scammers use to lure job seekers into fake schemes.")

    if any(kw in text_lower for kw in ["no experience", "no qualification", "no skills", "freshers only", "unskilled", "anyone can"]):
        found = [kw for kw in ["no experience","no qualification","no skills","freshers only","unskilled","anyone can"] if kw in text_lower]
        scam_signals.append(f"🔴 No Qualifications Required — Phrases like '{', '.join(found)}' detected. Real companies always mention minimum educational or skill requirements.")

    if re.search(r'\b\d{10}\b', job_text):
        num = re.search(r'\b\d{10}\b', job_text).group()
        scam_signals.append(f"🔴 Personal Phone Number Present — A 10-digit number ({num}) was found directly in the post. Legitimate companies communicate through official email IDs, not personal WhatsApp or phone numbers.")

    if any(kw in text_lower for kw in ["bank account", "aadhaar", "send your", "deposit", "pan card", "otp", "account number"]):
        found = [kw for kw in ["bank account","aadhaar","send your","deposit","pan card","otp","account number"] if kw in text_lower]
        scam_signals.append(f"🔴 Requests Sensitive Personal/Financial Info — Words like '{', '.join(found)}' found. No real employer asks for your Aadhaar, PAN, bank details, or OTP during recruitment.")

    if any(kw in text_lower for kw in ["urgent", "immediately", "limited seats", "hurry", "apply fast", "last chance", "today only", "expires"]):
        found = [kw for kw in ["urgent","immediately","limited seats","hurry","apply fast","last chance","today only","expires"] if kw in text_lower]
        scam_signals.append(f"🔴 High-Pressure Urgency Tactics — Phrases like '{', '.join(found)}' are designed to rush you into applying without thinking.")

    if any(kw in text_lower for kw in ["registration fee", "processing fee", "security deposit", "joining fee", "pay to join", "training fee", "kit fee"]):
        found = [kw for kw in ["registration fee","processing fee","security deposit","joining fee","pay to join","training fee","kit fee"] if kw in text_lower]
        scam_signals.append(f"🔴 Upfront Fee Demanded — '{', '.join(found)}' detected. Legitimate companies NEVER charge candidates any fee.")

    if any(kw in text_lower for kw in ["free visa", "free ticket", "free accommodation", "overseas", "abroad"]):
        found = [kw for kw in ["free visa","free ticket","free accommodation","overseas","abroad"] if kw in text_lower]
        scam_signals.append(f"🔴 Unrealistic Free Benefits — '{', '.join(found)}' found. Promises of free visa, air tickets, or accommodation are hallmarks of international job fraud.")

    if any(kw in text_lower for kw in ["whatsapp", "telegram", "call now", "contact immediately", "msg only"]):
        found = [kw for kw in ["whatsapp","telegram","call now","contact immediately","msg only"] if kw in text_lower]
        scam_signals.append(f"🟡 Informal Contact Method — '{', '.join(found)}' found. Authentic job postings use official email IDs and company portals, not personal WhatsApp or Telegram.")

    if any(kw in text_lower for kw in ["work from home", "work at home", "home based"]) and not any(kw in text_lower for kw in ["pvt ltd","private limited","technologies","solutions","company"]):
        scam_signals.append("🟡 Unverified Work From Home Claim — WFH jobs are common but scams abuse this. Without a verified company name, this could be a data entry fraud or fake part-time scheme.")



    # ── NEW signal: suspicious links ──
    if re.search(r'wa\.me\/|whatsapp\.com\/|t\.me\/|bit\.ly|tinyurl\.com', job_text, re.IGNORECASE):
        scam_signals.append("🔴 Suspicious Link Detected — WhatsApp/Telegram links or shortened URLs are heavily used in job scams to avoid traceability.")

    # ── NEW signal: unrealistic Indian daily salary ──
    sal_match = re.search(r'(\d+)\s*(?:rs\.?|₹|inr)?\s*(?:per day|\/day|daily)', job_text, re.IGNORECASE)
    if sal_match and int(sal_match.group(1)) >= 3000:
        scam_signals.append(f"🔴 Unrealistic Daily Pay — ₹{sal_match.group(1)}/day claimed. Entry-level genuine jobs in India do not offer such daily payouts.")

    # ── NEW signal: hinglish scam patterns ──
    hinglish_patterns = [
        r'ghar\s+baithe\s+(kama|earn|paise)',
        r'(kaam|kam)\s+(karo|karen).{0,30}(paise|paisa|income)',
        r'(daily|roz)\s+(paise|paisa|earning|income)',
        r'sirf\s+\d+\s+(ghante|hour).{0,20}(earn|kama|income)',
    ]
    if any(re.search(p, job_text, re.IGNORECASE) for p in hinglish_patterns):
        scam_signals.append("🔴 Hinglish Scam Language — 'Ghar baithe kamao' style content detected. These posts are almost always fraudulent work-from-home schemes.")

    if re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', job_text):
        email = re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', job_text).group()
        legit_signals.append(f"✅ Official Email Address Found — '{email}' is present. Legitimate recruiters always provide a company email for communication.")

    if re.search(r'www\.|https?://', job_text):
        legit_signals.append("✅ Company Website/URL Provided — A web link was found. Real companies share their official website, which you can verify independently.")

    if any(kw in text_lower for kw in ["ctc", "lpa", "per annum", "salary range", "fixed salary", "salary:"]):
        legit_signals.append("✅ Structured Salary Information — CTC or salary range is mentioned clearly. Legitimate job postings are transparent about compensation.")

    if any(kw in text_lower for kw in ["degree", "bachelor", "master", "b.tech", "mba", "graduate", "diploma", "b.sc", "b.com"]):
        legit_signals.append("✅ Education Requirements Specified — Minimum educational qualifications are mentioned, which is standard in genuine job postings.")

    if any(kw in text_lower for kw in ["interview", "selection process", "hr round", "technical round", "written test", "aptitude"]):
        legit_signals.append("✅ Formal Selection Process Mentioned — Interview rounds or selection stages are described. This indicates a structured hiring process.")

    if any(kw in text_lower for kw in ["pvt ltd", "private limited", "ltd", "inc", "corporation", "technologies", "solutions", "services"]):
        legit_signals.append("✅ Registered Company Name Pattern — Company name suggests a registered organization. You can verify this on MCA website at mca.gov.in.")

    if any(kw in text_lower for kw in ["pf", "esic", "provident fund", "health insurance", "medical", "bonus", "leave"]):
        legit_signals.append("✅ Employee Benefits Mentioned — Statutory benefits like PF, ESIC, or health insurance are mentioned.")

    for hf in h_flags:
        scam_signals.append(f"🔴 AI Pattern Detected: {hf} — This pattern was flagged by the machine learning model as commonly associated with fraudulent job postings.")

    if label == "Scam Job":
        explanation_parts = [f"⚠️ SCAM ALERT — This job posting shows {len(scam_signals)} suspicious indicator(s):\n"]
        explanation_parts += scam_signals
        if legit_signals:
            explanation_parts.append(f"\nSome legitimate signals found (but outweighed by scam indicators):")
            explanation_parts += legit_signals
        explanation_parts.append(f"\n📊 AI Scam Score: {round(final_score*100,1)}% — Do NOT share personal details or pay any fees.")
    elif label == "Doubtful":
        explanation_parts = [f"⚠️ DOUBTFUL JOB — This posting has mixed signals. Proceed with caution.\n"]
        if scam_signals:
            explanation_parts.append("Suspicious indicators found:")
            explanation_parts += scam_signals
        if legit_signals:
            explanation_parts.append(f"\nLegitimate signals found:")
            explanation_parts += legit_signals
        explanation_parts.append(f"\n📊 AI Risk Score: {round(final_score*100,1)}% — Verify company thoroughly before sharing any personal details.")
    else:
        explanation_parts = [f"✅ LEGITIMATE JOB — This posting shows {len(legit_signals)} credibility indicator(s):\n"]
        explanation_parts += legit_signals
        if scam_signals:
            explanation_parts.append(f"\nMinor concerns found (but insufficient to classify as scam):")
            explanation_parts += scam_signals
        explanation_parts.append(f"\n📊 AI Legitimacy Score: {round((1-final_score)*100,1)}% — Still verify company independently before applying.")

    explanation = "\n".join(explanation_parts)

    return {
        "label":         label,
        "confidence":    round(confidence * 100, 2),
        "explanation":   explanation,
        "scam_signals":  scam_signals,
        "legit_signals": legit_signals,
        "probs": {
            "scam":  round(final_score, 4),
            "legit": round(1 - final_score, 4)
        }
    }

# =========================================================
# AUTH ROUTES
# =========================================================

@app.route("/register", methods=["POST"])
def register():
    data = request.json or {}
    required = ["username", "password", "security_question", "security_answer"]
    if not all(k in data and data[k] for k in required):
        return jsonify({"status": "error", "message": "All fields required"}), 400

    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("SELECT id FROM users WHERE username=?", (data["username"],))
    if c.fetchone():
        conn.close()
        return jsonify({"status": "exists", "message": "Username already taken!"})

    c.execute("""
        INSERT INTO users(username, password, security_question, security_answer)
        VALUES (?, ?, ?, ?)
    """, (
        data["username"],
        generate_password_hash(data["password"]),
        data["security_question"],
        generate_password_hash(data["security_answer"])
    ))
    conn.commit()
    conn.close()
    return jsonify({"status": "success", "message": "Account created! Please login."})


def _bump_login_fail(c, username: str):
    c.execute("SELECT fail_count FROM login_attempts WHERE username=?", (username,))
    row = c.fetchone()
    fail_count = (row[0] if row else 0) + 1

    locked_until = None
    if fail_count >= MAX_LOGIN_ATTEMPTS:
        c.execute("SELECT datetime('now', ?)", (f'+{LOCKOUT_MINUTES} minutes',))
        locked_until = c.fetchone()[0]

    c.execute("""
        INSERT INTO login_attempts(username, fail_count, locked_until, updated_at)
        VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(username) DO UPDATE SET
            fail_count=excluded.fail_count,
            locked_until=excluded.locked_until,
            updated_at=CURRENT_TIMESTAMP
    """, (username, fail_count, locked_until))


@app.route("/login", methods=["POST"])
def login():
    data = request.json or {}
    username = (data.get("username") or "").strip()
    password = (data.get("password") or "").strip()

    if not username or not password:
        return jsonify({"status": "error", "message": "Username & password required"}), 400

    conn = sqlite3.connect("database.db")
    c = conn.cursor()

    c.execute("SELECT fail_count, locked_until FROM login_attempts WHERE username=?", (username,))
    row = c.fetchone()

    if row and row[1]:
        locked_until = row[1]
        c.execute("SELECT (datetime(?) > datetime('now'))", (locked_until,))
        still_locked = c.fetchone()[0]
        if still_locked:
            conn.close()
            return jsonify({
                "status": "locked",
                "message": f"Too many failed attempts. Try again after {LOCKOUT_MINUTES} minutes."
            }), 429
        else:
            c.execute("UPDATE login_attempts SET fail_count=0, locked_until=NULL, updated_at=CURRENT_TIMESTAMP WHERE username=?", (username,))
            conn.commit()

    c.execute("SELECT password, role FROM users WHERE username=?", (username,))
    user = c.fetchone()

    if not user:
        _bump_login_fail(c, username)
        conn.commit()
        conn.close()
        return jsonify({"status": "error", "message": "Invalid username or password"}), 401

    if check_password_hash(user[0], password):
        c.execute("""
            INSERT INTO login_attempts(username, fail_count, locked_until, updated_at)
            VALUES (?, 0, NULL, CURRENT_TIMESTAMP)
            ON CONFLICT(username) DO UPDATE SET fail_count=0, locked_until=NULL, updated_at=CURRENT_TIMESTAMP
        """, (username,))
        conn.commit()
        conn.close()

        session.permanent = True
        session["username"] = username
        session["role"] = user[1]
        return jsonify({"status": "success", "username": username, "role": user[1]})

    _bump_login_fail(c, username)
    conn.commit()
    conn.close()
    return jsonify({"status": "error", "message": "Invalid username or password"}), 401


@app.route("/admin/login", methods=["POST"])
def admin_login():
    data = request.json or {}
    username = (data.get("username") or "").strip()
    password = (data.get("password") or "").strip()

    if not username or not password:
        return jsonify({"status": "error", "message": "Username & password required"}), 400

    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("SELECT password, role FROM users WHERE username=?", (username,))
    user = c.fetchone()
    conn.close()

    if not user:
        return jsonify({"status": "error", "message": "Invalid credentials"}), 401

    if user[1] != "admin":
        return jsonify({"status": "error", "message": "Not an admin account"}), 403

    if check_password_hash(user[0], password):
        session.clear()
        session.permanent = True
        session["username"] = username
        session["role"] = "admin"
        return jsonify({"status": "success", "username": username, "role": "admin"})

    return jsonify({"status": "error", "message": "Invalid credentials"}), 401


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"status": "success"})


@app.route("/me")
def me():
    username = session.get("username")
    if not username:
        return jsonify({"error": "login"}), 401
    return jsonify({"username": username, "role": session.get("role", "user")})


# =========================================================
# FORGOT PASSWORD
# =========================================================

@app.route("/get-security-question", methods=["POST"])
def get_security_question():
    data     = request.json or {}
    username = data.get("username", "").strip()

    if not username:
        return jsonify({"status": "error", "message": "Username required"})

    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("SELECT security_question FROM users WHERE username=?", (username,))
    row = c.fetchone()
    conn.close()

    if not row:
        return jsonify({"status": "error", "message": "Username not found!"})

    return jsonify({"status": "success", "security_question": row[0]})


@app.route("/reset-password", methods=["POST"])
def reset_password():
    data     = request.json or {}
    username = data.get("username", "").strip()
    answer   = data.get("security_answer", "").strip()
    new_pass = data.get("new_password", "").strip()

    if not username or not answer or not new_pass:
        return jsonify({"status": "error", "message": "All fields required"})

    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("SELECT security_answer FROM users WHERE username=?", (username,))
    row = c.fetchone()

    if not row:
        conn.close()
        return jsonify({"status": "error", "message": "User not found!"})

    if not check_password_hash(row[0], answer):
        conn.close()
        return jsonify({"status": "error", "message": "Wrong answer!"})

    c.execute("UPDATE users SET password=? WHERE username=?",
              (generate_password_hash(new_pass), username))
    conn.commit()
    conn.close()

    return jsonify({"status": "success", "message": "Password reset successful!"})


# =========================================================
# IMAGE / PDF TEXT EXTRACTION
# =========================================================

@app.route("/extract-image", methods=["POST"])
def extract_image():
    username = session.get("username")
    if not username:
        return jsonify({"error": "login", "message": "Please login first"}), 401

    if "file" not in request.files:
        return jsonify({"error": "no_file", "message": "No file uploaded"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "no_file", "message": "No file selected"}), 400

    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in ["jpg", "jpeg", "png", "webp", "bmp"]:
        return jsonify({"error": "invalid_type",
                        "message": "❌ Only JPG, PNG, WEBP images allowed."}), 400

    if not HAS_OCR:
        return jsonify({
            "error": "ocr_unavailable",
            "message": "❌ OCR not available. Run: pip install easyocr  —  then restart app.py"
        }), 503

    try:
        img_bytes = file.read()

        # ── DUPLICATE CHECK: same user + same file ──
        file_hash = hashlib.sha256(img_bytes).hexdigest()
        conn = sqlite3.connect("database.db")
        c = conn.cursor()
        c.execute("""
            SELECT id, prediction, confidence, explanation, created_at
            FROM dashboard_reports
            WHERE username=? AND file_hash=?
            ORDER BY id DESC LIMIT 1
        """, (username, file_hash))
        existing = c.fetchone()
        conn.close()
        if existing:
            rid, pred, conf, expl, cat = existing
            scam_prob = round(conf * 100, 1)
            is_scam = pred == "Scam Job"
            is_doubtful = pred == "Doubtful"
            disp_conf = round(conf * 100 if (is_scam or is_doubtful) else (1 - conf) * 100, 2)
            return jsonify({
                "duplicate": True,
                "message": "⚠️ You have already analyzed this file before.",
                "report_id": rid,
                "label": pred,
                "confidence": disp_conf,
                "scam_prob": scam_prob,
                "explanation": expl,
                "created_at": cat
            }), 200

        if OCR_ENGINE == "easyocr":
            import numpy as np
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            import cv2
            img_cv    = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            results   = _ocr_reader.readtext(img_cv)
            extracted_text = " ".join([r[1] for r in results]).strip()
        else:
            from PIL import Image as PILImage
            import pytesseract
            img = PILImage.open(BytesIO(img_bytes))
            extracted_text = pytesseract.image_to_string(img).strip()

        if len(extracted_text) < 20:
            return jsonify({
                "error": "no_text",
                "message": "⚠️ Could not extract readable text. Please upload a clearer image."
            }), 400

        if not is_job_related(extracted_text):
            return jsonify({
                "error": "not_job_related",
                "message": "⚠️ This image doesn't seem to be job-related. Please upload a job description, offer letter, or recruitment post only."
            }), 400

        return jsonify({"text": extracted_text, "chars": len(extracted_text), "file_hash": file_hash})

    except Exception as e:
        return jsonify({"error": "extract_failed", "message": f"❌ Error: {str(e)}"}), 500


@app.route("/extract-pdf", methods=["POST"])
def extract_pdf():
    username = session.get("username")
    if not username:
        return jsonify({"error": "login", "message": "Please login first"}), 401

    if "file" not in request.files:
        return jsonify({"error": "no_file", "message": "No file uploaded"}), 400

    file = request.files["file"]
    ext  = file.filename.rsplit(".", 1)[-1].lower() if file.filename else ""
    if ext != "pdf":
        return jsonify({"error": "invalid_type",
                        "message": "❌ Only PDF files allowed here."}), 400

    if not HAS_PDF:
        return jsonify({
            "error": "pdf_unavailable",
            "message": "❌ PDF parsing not available. Run: pip install PyMuPDF"
        }), 503

    try:
        pdf_bytes = file.read()

        # ── DUPLICATE CHECK: same user + same file ──
        file_hash = hashlib.sha256(pdf_bytes).hexdigest()
        conn = sqlite3.connect("database.db")
        c = conn.cursor()
        c.execute("""
            SELECT id, prediction, confidence, explanation, created_at
            FROM dashboard_reports
            WHERE username=? AND file_hash=?
            ORDER BY id DESC LIMIT 1
        """, (username, file_hash))
        existing = c.fetchone()
        conn.close()
        if existing:
            rid, pred, conf, expl, cat = existing
            scam_prob = round(conf * 100, 1)
            is_scam = pred == "Scam Job"
            is_doubtful = pred == "Doubtful"
            disp_conf = round(conf * 100 if (is_scam or is_doubtful) else (1 - conf) * 100, 2)
            return jsonify({
                "duplicate": True,
                "message": "⚠️ You have already analyzed this file before.",
                "report_id": rid,
                "label": pred,
                "confidence": disp_conf,
                "scam_prob": scam_prob,
                "explanation": expl,
                "created_at": cat
            }), 200

        doc       = fitz.open(stream=pdf_bytes, filetype="pdf")
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        doc.close()
        full_text = full_text.strip()

        if len(full_text) < 30:
            return jsonify({
                "error": "no_text",
                "message": "⚠️ Could not extract text from this PDF. It may be scanned/image-based."
            }), 400

        if not is_job_related(full_text):
            return jsonify({
                "error": "not_job_related",
                "message": "⚠️ This PDF doesn't appear to be job-related. Please upload only job descriptions, offer letters, or recruitment documents."
            }), 400

        trimmed = full_text[:3000]
        return jsonify({"text": trimmed, "chars": len(full_text), "file_hash": file_hash})

    except Exception as e:
        return jsonify({"error": "extract_failed", "message": f"❌ Error reading PDF: {str(e)}"}), 500


# =========================================================
# PREDICTION
# =========================================================

@app.route("/predict", methods=["POST"])
def predict():
    username = session.get("username")
    if not username:
        return jsonify({"error": "login", "message": "Please login first"}), 401

    data     = request.json or {}
    job_text = data.get("job_text", "").strip()
    if not job_text:
        return jsonify({"error": "empty", "message": "Job text cannot be empty"}), 400

    extra = {
        "has_logo":        data.get("has_logo", 0),
        "telecommuting":   data.get("telecommuting", 0),
        "company_profile": data.get("company_profile", ""),
        "requirements":    data.get("requirements", ""),
        "salary_range":    data.get("salary_range", ""),
    }

    result = predict_scam(job_text, extra)

    scam_prob_for_db = result["probs"]["scam"]
    file_hash = data.get("file_hash", None)

    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("""
        INSERT INTO dashboard_reports(username, job_text, prediction, confidence, explanation, file_hash)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (username, job_text, result["label"], scam_prob_for_db, result["explanation"], file_hash))
    rid = c.lastrowid
    conn.commit()
    conn.close()

    # ── FIX: Return BOTH 'label' AND 'prediction' so frontend works regardless ──
    return jsonify({
        "label":         result["label"],
        "prediction":    result["label"],
        "confidence":    result["confidence"],
        "ai_explain":    result["explanation"],
        "scam_signals":  result.get("scam_signals", []),
        "legit_signals": result.get("legit_signals", []),
        "report_id":     rid,
        "probs":         result["probs"]
    })


# =========================================================
# USER REPORTS
# =========================================================

@app.route("/reports")
def reports():
    username = session.get("username")
    if not username:
        return jsonify({"error": "login"}), 401

    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("""
        SELECT id, prediction, confidence, created_at
        FROM dashboard_reports
        WHERE username=?
        ORDER BY id DESC
    """, (username,))
    rows = c.fetchall()
    conn.close()

    return jsonify([
        {
            "id":         r[0],
            "label":      r[1],
            "confidence": round((r[2] if r[1] in ("Scam Job", "Doubtful") else 1-r[2]) * 100, 2),
            "created_at": r[3].split(" ")[0] if r[3] else "—"
        }
        for r in rows
    ])


def make_bar_graph(scam_pct, legit_pct):
    width  = 200
    height = 90
    d      = Drawing(width, height)

    scam_color  = colors.HexColor("#e53935")
    legit_color = colors.HexColor("#00897b")
    bg_color    = colors.HexColor("#dde3ef")
    label_color = colors.HexColor("#5a6a85")
    bar_h       = 18
    bar_max_w   = 130
    x_start     = 58
    y_scam      = 52
    y_legit     = 20

    bg1 = Rect(x_start, y_scam, bar_max_w, bar_h)
    bg1.fillColor = bg_color; bg1.strokeColor = None
    d.add(bg1)
    fill_w1 = max(int(bar_max_w * scam_pct), 2)
    f1 = Rect(x_start, y_scam, fill_w1, bar_h)
    f1.fillColor = scam_color; f1.strokeColor = None
    d.add(f1)
    d.add(String(x_start - 4, y_scam + 6, "Scam Risk", textAnchor='end',
                 fontSize=7, fontName='Helvetica-Bold', fillColor=scam_color))
    d.add(String(x_start + bar_max_w + 5, y_scam + 6, f"{round(scam_pct*100)}%",
                 textAnchor='start', fontSize=8, fontName='Helvetica-Bold', fillColor=scam_color))

    bg2 = Rect(x_start, y_legit, bar_max_w, bar_h)
    bg2.fillColor = bg_color; bg2.strokeColor = None
    d.add(bg2)
    fill_w2 = max(int(bar_max_w * legit_pct), 2)
    f2 = Rect(x_start, y_legit, fill_w2, bar_h)
    f2.fillColor = legit_color; f2.strokeColor = None
    d.add(f2)
    d.add(String(x_start - 4, y_legit + 6, "Legit Score", textAnchor='end',
                 fontSize=7, fontName='Helvetica-Bold', fillColor=legit_color))
    d.add(String(x_start + bar_max_w + 5, y_legit + 6, f"{round(legit_pct*100)}%",
                 textAnchor='start', fontSize=8, fontName='Helvetica-Bold', fillColor=legit_color))

    d.add(String(x_start + bar_max_w/2, height - 6, "Risk Analysis",
                 textAnchor='middle', fontSize=9, fontName='Helvetica-Bold',
                 fillColor=label_color))

    return d


@app.route("/reports/<int:rid>/download")
def download(rid):
    username = session.get("username")
    if not username:
        return jsonify({"error": "login"}), 401

    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("""
        SELECT prediction, confidence, explanation, job_text, created_at
        FROM dashboard_reports WHERE id=? AND username=?
    """, (rid, username))
    r = c.fetchone()
    conn.close()

    if not r:
        return "Not found", 404

    prediction, confidence, explanation, job_text, created_at = r
    scam_prob  = confidence
    legit_prob = 1.0 - scam_prob
    is_scam    = prediction == "Scam Job"
    conf_pct   = round(scam_prob * 100 if is_scam else legit_prob * 100, 2)

    CARD_BG     = colors.HexColor("#f4f7fb")
    ACCENT      = colors.HexColor("#0055cc")
    SCAM_RED    = colors.HexColor("#e53935")
    REAL_GREEN  = colors.HexColor("#00897b")
    TEXT_MAIN   = colors.HexColor("#1a1a2e")
    TEXT_MUTED  = colors.HexColor("#5a6a85")
    VERDICT_COL = SCAM_RED if is_scam else REAL_GREEN

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=18*mm, rightMargin=18*mm,
        topMargin=14*mm, bottomMargin=14*mm
    )

    def sty(name, **kw):
        return ParagraphStyle(name, **kw)

    st_title   = sty("title",   fontName="Helvetica-Bold", fontSize=22, textColor=TEXT_MAIN,   spaceAfter=2, leading=26)
    st_sub     = sty("sub",     fontName="Helvetica",      fontSize=9,  textColor=TEXT_MUTED)
    st_label   = sty("label",   fontName="Helvetica-Bold", fontSize=9,  textColor=TEXT_MUTED,  spaceAfter=2)
    st_value   = sty("value",   fontName="Helvetica-Bold", fontSize=15, textColor=VERDICT_COL)
    st_conf    = sty("conf",    fontName="Helvetica-Bold", fontSize=15, textColor=ACCENT)
    st_exlabel = sty("exlabel", fontName="Helvetica-Bold", fontSize=10, textColor=ACCENT,       spaceAfter=4)
    st_jobbody = sty("jobbody", fontName="Helvetica",      fontSize=8,  textColor=TEXT_MUTED,   leading=13)
    st_footer  = sty("footer",  fontName="Helvetica",      fontSize=8,  textColor=TEXT_MUTED,   alignment=1)

    def page_bg(canvas, doc):
        w, h = A4
        canvas.saveState()
        canvas.setFillColor(colors.HexColor("#ffffff"))
        canvas.rect(0, 0, w, h, fill=1, stroke=0)
        canvas.restoreState()

    story = []

    header_tbl = Table([[
        Paragraph("<b>🤖 JobShield AI</b>", sty("hl", fontName="Helvetica-Bold", fontSize=13, textColor=ACCENT)),
        Paragraph(f"Report #{rid}", sty("hr", fontName="Helvetica", fontSize=9, textColor=TEXT_MUTED, alignment=2))
    ]], colWidths=["60%","40%"])
    header_tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0,0),(-1,-1), CARD_BG),
        ("TOPPADDING",   (0,0),(-1,-1), 10), ("BOTTOMPADDING",(0,0),(-1,-1), 10),
        ("LEFTPADDING",  (0,0),(-1,-1), 14), ("RIGHTPADDING", (0,0),(-1,-1), 14),
        ("VALIGN",       (0,0),(-1,-1), "MIDDLE"),
    ]))
    story.append(header_tbl)
    story.append(Spacer(1, 6*mm))
    story.append(Paragraph("Analysis Report", st_title))
    story.append(Paragraph(f"Generated: {created_at}  ·  Analyzed by: {username}", st_sub))
    story.append(Spacer(1, 5*mm))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1f3050")))
    story.append(Spacer(1, 5*mm))

    conf_pct_clamped = max(min(conf_pct, 100), 0)
    bar_w_fill  = max(int(90 * conf_pct_clamped / 100), 1)
    bar_w_empty = 90 - bar_w_fill
    bar_tbl = Table([["", ""]], colWidths=[bar_w_fill*mm, bar_w_empty*mm])
    bar_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(0,0), VERDICT_COL),
        ("BACKGROUND",    (1,0),(1,0), colors.HexColor("#253650")),
        ("TOPPADDING",    (0,0),(-1,-1), 3), ("BOTTOMPADDING",(0,0),(-1,-1), 3),
        ("LEFTPADDING",   (0,0),(-1,-1), 0), ("RIGHTPADDING", (0,0),(-1,-1), 0),
    ]))

    donut = make_bar_graph(scam_prob, legit_prob)
    legend_tbl = Table([[Spacer(1,1)]], colWidths=[160])
    legend_tbl.setStyle(TableStyle([
        ("LEFTPADDING",(0,0),(-1,-1),0),("RIGHTPADDING",(0,0),(-1,-1),0),
        ("TOPPADDING",(0,0),(-1,-1),0),("BOTTOMPADDING",(0,0),(-1,-1),0),
    ]))

    verdict_emoji = "🚨 Scam Detected" if is_scam else "✅ Legitimate Job"
    left_cell  = [Paragraph("VERDICT", st_label), Paragraph(verdict_emoji, st_value),
                  Spacer(1,4*mm), Paragraph("CONFIDENCE", st_label),
                  Paragraph(f"{conf_pct}%", st_conf), Spacer(1,3*mm), bar_tbl]
    right_cell = [donut, legend_tbl]

    main_row = Table([[left_cell, right_cell]], colWidths=["55%","45%"])
    main_row.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), CARD_BG),
        ("TOPPADDING",    (0,0),(-1,-1), 12), ("BOTTOMPADDING",(0,0),(-1,-1), 12),
        ("LEFTPADDING",   (0,0),(0,-1),  16), ("RIGHTPADDING", (-1,0),(-1,-1),10),
        ("VALIGN",        (0,0),(-1,-1), "TOP"),
    ]))
    story.append(main_row)
    story.append(Spacer(1, 5*mm))

    explain_lines = explanation.split("\n")
    explain_paras = []
    for line in explain_lines:
        if not line.strip():
            explain_paras.append(Spacer(1, 3*mm))
            continue
        if line.startswith("🔴"):
            col = SCAM_RED
        elif line.startswith("✅"):
            col = REAL_GREEN
        elif line.startswith("🟡"):
            col = colors.HexColor("#f59e0b")
        elif line.startswith("⚠️") or line.startswith("📊"):
            col = VERDICT_COL
        else:
            col = TEXT_MAIN
        explain_paras.append(
            Paragraph(line, sty("ep", fontName="Helvetica", fontSize=9,
                                textColor=col, leading=14, spaceAfter=2))
        )

    explain_content = [[Paragraph("🤖 AI Detailed Analysis", st_exlabel)]] + [[p] for p in explain_paras]
    explain_tbl = Table(explain_content, colWidths=["100%"])
    explain_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), CARD_BG),
        ("TOPPADDING",    (0,0),(-1,-1), 8), ("BOTTOMPADDING",(0,0),(-1,-1), 8),
        ("LEFTPADDING",   (0,0),(-1,-1), 16), ("RIGHTPADDING", (0,0),(-1,-1), 16),
        ("LINEABOVE",     (0,0),(-1,0),  2,   VERDICT_COL),
    ]))
    story.append(explain_tbl)
    story.append(Spacer(1, 5*mm))

    if job_text:
        preview = job_text[:400] + ("..." if len(job_text) > 400 else "")
        job_tbl = Table([
            [Paragraph("Job Description (Preview)", sty("jl", fontName="Helvetica-Bold", fontSize=9, textColor=TEXT_MUTED))],
            [Paragraph(preview, st_jobbody)],
        ], colWidths=["100%"])
        job_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,-1), colors.HexColor("#f4f7fb")),
            ("TOPPADDING",    (0,0),(-1,-1), 10), ("BOTTOMPADDING",(0,0),(-1,-1), 10),
            ("LEFTPADDING",   (0,0),(-1,-1), 14), ("RIGHTPADDING", (0,0),(-1,-1), 14),
        ]))
        story.append(job_tbl)
        story.append(Spacer(1, 5*mm))

    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#dde3ef")))
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph("Generated by JobShield AI  ·  Protect yourself from fraudulent job postings", st_footer))

    doc.build(story, onFirstPage=page_bg, onLaterPages=page_bg)
    buffer.seek(0)

    return Response(
        buffer.getvalue(),
        mimetype="application/pdf",
        headers={"Content-Disposition": f"attachment;filename=report_{rid}.pdf"}
    )


# =========================================================
# ADMIN ROUTES
# =========================================================

def require_admin(fn):
    from functools import wraps
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if session.get("role") != "admin":
            return jsonify({"error": "Unauthorized"}), 403
        return fn(*args, **kwargs)
    return wrapper


@app.route("/admin/users")
@require_admin
def admin_users():
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("SELECT username, role FROM users ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    return jsonify([{"username": r[0], "role": r[1]} for r in rows])


@app.route("/admin/all-reports")
@require_admin
def admin_all_reports():
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("""
        SELECT username, prediction, confidence, created_at
        FROM dashboard_reports
        ORDER BY id DESC
        LIMIT 100
    """)
    rows = c.fetchall()
    conn.close()
    return jsonify([
        {
            "username":   r[0],
            "label":      r[1],
            "confidence": round(r[2] * 100, 2),
            "created_at": r[3]
        }
        for r in rows
    ])


@app.route("/admin/me")
def admin_me():
    if session.get("role") == "admin":
        return jsonify({
            "username": session.get("username"),
            "role": "admin"
        })
    return jsonify({"error": "Unauthorized"}), 403


@app.route("/report-admin", methods=["POST"])
def report_admin():
    username = session.get("username")
    if not username:
        return jsonify({"error": "login"}), 401

    data = request.json or {}
    msg  = data.get("message", "").strip()
    if not msg:
        return jsonify({"status": "error", "message": "Message required"}), 400

    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("INSERT INTO admin_reports(username, message) VALUES (?, ?)", (username, msg))
    conn.commit()
    conn.close()

    return jsonify({"status": "success"})


@app.route("/admin/delete-user/<username>", methods=["DELETE"])
@require_admin
def delete_user(username):
    if username == ADMIN_USER:
        return jsonify({"error": "Cannot delete admin"}), 403

    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("DELETE FROM users WHERE username=?", (username,))
    conn.commit()
    conn.close()

    return jsonify({"status": "deleted"})


@app.route("/admin/analytics")
@require_admin
def admin_analytics():
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("""
        SELECT prediction, COUNT(*)
        FROM dashboard_reports
        GROUP BY prediction
    """)
    data = dict(c.fetchall())
    conn.close()

    return jsonify({
        "scam": data.get("Scam Job", 0),
        "real": data.get("Real Job", 0)
    })


@app.route("/admin/messages")
@require_admin
def admin_messages():
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("""
        SELECT username, message, created_at
        FROM admin_reports
        ORDER BY id DESC
    """)
    rows = c.fetchall()
    conn.close()

    return jsonify([
        {"username": r[0], "message": r[1], "time": r[2]}
        for r in rows
    ])


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
