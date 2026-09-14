from flask import Flask, render_template, request, redirect, url_for, session, flash
import os
import re
import json
import requests
import mysql.connector
import pymupdf
import pytesseract

from PIL import Image
from werkzeug.security import generate_password_hash, check_password_hash


# =========================================================
# FLASK SETUP
# =========================================================

app = Flask(__name__)

app.secret_key = os.environ.get(
    "AI_ANALYSE_SECRET_KEY",
    "ai-analyse-secret-key"
)


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Allowed upload extensions
ALLOWED_EXTENSIONS = {
    "pdf",
    "txt",
    "jpg",
    "jpeg",
    "png",
    "webp"
}


# Tesseract path
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


# Ollama model
OLLAMA_MODEL = "nemotron-3-ultra:cloud"

OLLAMA_URL = "http://localhost:11434/api/generate"


# =========================================================
# MYSQL CONNECTION
# =========================================================

def get_db_connection():

    try:

        db_host = os.environ.get("DB_HOST")
        print("DB_HOST VALUE:", db_host)
        db_port = os.environ.get("DB_PORT")
        db_user = os.environ.get("DB_USER")
        db_password = os.environ.get("DB_PASSWORD")
        db_name = os.environ.get("DB_NAME")

        print("========== DATABASE DEBUG ==========")
        
        print("DB_PORT:", db_port)
        print("DB_USER:", db_user)
        print("DB_PASSWORD SET:", bool(db_password))
        print("DB_NAME:", db_name)
        print("====================================")

        db = mysql.connector.connect(
            host=db_host,
            port=int(db_port) if db_port else 3306,
            user=db_user,
            password=db_password,
            database=db_name
        )

        print("MySQL Database Connected Successfully!")

        return db

    except mysql.connector.Error as err:

        print("========== MYSQL CONNECTION ERROR ==========")
        print("ERROR:", err)
        print("============================================")

        return None

    except Exception as e:

        print("========== GENERAL DATABASE ERROR ==========")
        print("ERROR:", e)
        print("============================================")

        return None


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def allowed_file(filename):

    if "." not in filename:
        return False

    extension = filename.rsplit(".", 1)[1].lower()

    return extension in ALLOWED_EXTENSIONS


def login_required():

    return "user_id" in session


# =========================================================
# OCR / TEXT EXTRACTION
# =========================================================

def extract_pdf_text(file_path):

    extracted_text = ""

    try:

        document = pymupdf.open(file_path)

        # First try normal PDF text extraction
        for page in document:

            page_text = page.get_text()

            if page_text.strip():

                extracted_text += page_text + "\n"

        # If no text was found, use OCR
        if not extracted_text.strip():

            extracted_text = ""

            for page in document:

                pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))

                image = Image.frombytes(
                    "RGB",
                    [pix.width, pix.height],
                    pix.samples
                )

                ocr_text = pytesseract.image_to_string(image)

                extracted_text += ocr_text + "\n"

            print("PDF extraction method: OCR")

        else:

            print("PDF extraction method: PyMuPDF")

        document.close()

        return extracted_text.strip()

    except Exception as e:

        print("PDF extraction error:", e)

        return ""


def extract_image_text(file_path):

    try:

        image = Image.open(file_path)

        text = pytesseract.image_to_string(image)

        print("Image extraction method: Tesseract OCR")

        return text.strip()

    except Exception as e:

        print("Image OCR error:", e)

        return ""


def extract_txt_text(file_path):

    try:

        with open(
            file_path,
            "r",
            encoding="utf-8",
            errors="ignore"
        ) as file:

            text = file.read()

        print("Text file extraction completed.")

        return text.strip()

    except Exception as e:

        print("TXT extraction error:", e)

        return ""


# =========================================================
# OLLAMA AI ANALYSIS
# =========================================================

def analyze_with_ollama(text):

    if not text or not text.strip():

        return {
            "ai_probability": 0,
            "human_probability": 0,
            "verdict": "Unable to Analyse",
            "confidence": "Low",
            "humanization_score": 0,
            "humanization_level": "Unknown",
            "reason": "No readable text was found for analysis."
        }

    # Limit extremely large text
    analysis_text = text[:30000]

    prompt = f"""
You are an AI text analysis system.

Analyze the following text and estimate whether it appears:
1. AI-generated
2. Human-written
3. AI-generated but humanized/edited

IMPORTANT:
Do not simply say that AI detection is always certain.
Give a reason based on writing characteristics.

Look for:
- sentence structure
- repetition
- predictability
- wording
- unnatural consistency
- generic phrasing
- variation in sentence length
- personal/natural expression
- editing or humanization patterns

Return ONLY valid JSON.

Required JSON format:

{{
    "ai_probability": number,
    "human_probability": number,
    "verdict": "AI Generated" or "Human Written" or "Likely Humanized",
    "confidence": "Low" or "Medium" or "High",
    "humanization_score": number,
    "humanization_level": "Low" or "Medium" or "High",
    "reason": "Brief explanation in 2-3 sentences"
}}

The probabilities must be between 0 and 100.
The humanization score must be between 0 and 100.

Text to analyze:

{analysis_text}
"""

    try:

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "think": False,
                "keep_alive": "10m",
                "options": {
                    "temperature": 0,
                    "num_predict": 180
                }
            },
            timeout=120
        )

        if response.status_code != 200:

            print(
                "Ollama HTTP Error:",
                response.status_code
            )

            return fallback_analysis(text)

        data = response.json()

        raw_response = data.get(
            "response",
            ""
        ).strip()

        print("\n========== OLLAMA RESPONSE ==========")
        print(raw_response)
        print("======================================\n")

        result = None

        # Try direct JSON
        try:

            result = json.loads(raw_response)

        except Exception:

            pass

        # Try JSON inside response
        if result is None:

            json_match = re.search(
                r"\{.*\}",
                raw_response,
                re.DOTALL
            )

            if json_match:

                try:

                    result = json.loads(
                        json_match.group(0)
                    )

                except Exception:

                    result = None

        if not isinstance(result, dict):

            return fallback_analysis(text)

        # Numeric values
        ai_probability = float(
            result.get(
                "ai_probability",
                50
            )
        )

        human_probability = float(
            result.get(
                "human_probability",
                50
            )
        )

        humanization_score = float(
            result.get(
                "humanization_score",
                50
            )
        )

        # Keep values between 0 and 100
        ai_probability = max(
            0,
            min(100, ai_probability)
        )

        human_probability = max(
            0,
            min(100, human_probability)
        )

        humanization_score = max(
            0,
            min(100, humanization_score)
        )

        verdict = str(
            result.get(
                "verdict",
                "Unable to Analyse"
            )
        )

        confidence = str(
            result.get(
                "confidence",
                "Medium"
            )
        )

        humanization_level = str(
            result.get(
                "humanization_level",
                "Medium"
            )
        )

        reason = str(
            result.get(
                "reason",
                "The model did not provide a detailed explanation."
            )
        )

        return {
            "ai_probability": round(
                ai_probability,
                2
            ),
            "human_probability": round(
                human_probability,
                2
            ),
            "verdict": verdict,
            "confidence": confidence,
            "humanization_score": round(
                humanization_score,
                2
            ),
            "humanization_level": humanization_level,
            "reason": reason
        }

    except Exception as e:

        print(
            "Ollama Error:",
            e
        )

        return fallback_analysis(text)


# =========================================================
# FALLBACK ANALYSIS
# =========================================================

def fallback_analysis(text):

    words = text.split()

    if not words:

        return {
            "ai_probability": 0,
            "human_probability": 0,
            "verdict": "Unable to Analyse",
            "confidence": "Low",
            "humanization_score": 0,
            "humanization_level": "Unknown",
            "reason": "There was not enough readable text to analyse."
        }

    # Basic fallback only.
    # This is NOT intended to replace the Ollama analysis.

    sentences = re.split(
        r"[.!?]+",
        text
    )

    sentences = [
        s.strip()
        for s in sentences
        if s.strip()
    ]

    if len(sentences) > 1:

        lengths = [
            len(s.split())
            for s in sentences
        ]

        average_length = sum(lengths) / len(lengths)

        if average_length > 25:

            ai_probability = 65

        else:

            ai_probability = 50

    else:

        ai_probability = 50

    human_probability = 100 - ai_probability

    if ai_probability >= 60:

        verdict = "Likely AI Generated"

        reason = (
            "The AI analysis service could not be reached, "
            "so this is only a basic fallback estimate. "
            "The text shows relatively consistent sentence "
            "patterns."
        )

    else:

        verdict = "Likely Human Written"

        reason = (
            "The AI analysis service could not be reached, "
            "so this is only a basic fallback estimate. "
            "The text shows some natural variation in its "
            "sentence structure."
        )

    return {
        "ai_probability": ai_probability,
        "human_probability": human_probability,
        "verdict": verdict,
        "confidence": "Low",
        "humanization_score": 50,
        "humanization_level": "Medium",
        "reason": reason
    }


# =========================================================
# REGISTER
# =========================================================

@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        username = request.form.get(
            "username",
            ""
        ).strip()

        password = request.form.get(
            "password",
            ""
        )

        confirm_password = request.form.get(
            "confirm_password",
            ""
        )

        if not username or not password or not confirm_password:

            flash(
                "Please fill all fields.",
                "error"
            )

            return redirect(
                url_for("register")
            )

        if password != confirm_password:

            flash(
                "Passwords do not match.",
                "error"
            )

            return redirect(
                url_for("register")
            )

        if len(password) < 6:

            flash(
                "Password must contain at least 6 characters.",
                "error"
            )

            return redirect(
                url_for("register")
            )

        db = get_db_connection()

        if db is None:

            flash(
                "Database connection error.",
                "error"
            )

            return redirect(
                url_for("register")
            )

        cursor = db.cursor()

        try:

            cursor.execute(
                """
                SELECT user_id
                FROM users
                WHERE username = %s
                """,
                (username,)
            )

            existing_user = cursor.fetchone()

            if existing_user:

                flash(
                    "Username already exists.",
                    "error"
                )

                return redirect(
                    url_for("register")
                )

            password_hash = generate_password_hash(
                password
            )

            cursor.execute(
                """
                INSERT INTO users
                (username, password_hash)
                VALUES (%s, %s)
                """,
                (
                    username,
                    password_hash
                )
            )

            db.commit()

            flash(
                "Account created successfully. Please login.",
                "success"
            )

            return redirect(
                url_for("login")
            )

        except mysql.connector.Error as err:

            db.rollback()

            print(
                "Register Database Error:",
                err
            )

            flash(
                "Could not create account.",
                "error"
            )

            return redirect(
                url_for("register")
            )

        finally:

            cursor.close()
            db.close()

    return render_template(
        "register.html"
    )


# =========================================================
# LOGIN
# =========================================================

@app.route("/login", methods=["GET", "POST"])
def login():

    if "user_id" in session:

        return redirect(
            url_for("index")
        )

    if request.method == "POST":

        username = request.form.get(
            "username",
            ""
        ).strip()

        password = request.form.get(
            "password",
            ""
        )

        if not username or not password:

            flash(
                "Please enter username and password.",
                "error"
            )

            return redirect(
                url_for("login")
            )

        db = get_db_connection()

        if db is None:

            flash(
                "Database connection error.",
                "error"
            )

            return redirect(
                url_for("login")
            )

        cursor = db.cursor(
            dictionary=True
        )

        try:

            cursor.execute(
                """
                SELECT user_id, username, password_hash
                FROM users
                WHERE username = %s
                """,
                (username,)
            )

            user = cursor.fetchone()

            if user and check_password_hash(
                user["password_hash"],
                password
            ):

                session.clear()

                session["user_id"] = user["user_id"]

                session["username"] = user["username"]

                return redirect(
                    url_for("index")
                )

            flash(
                "Invalid username or password.",
                "error"
            )

            return redirect(
                url_for("login")
            )

        except mysql.connector.Error as err:

            print(
                "Login Database Error:",
                err
            )

            flash(
                "Database error occurred.",
                "error"
            )

            return redirect(
                url_for("login")
            )

        finally:

            cursor.close()
            db.close()

    return render_template(
        "login.html"
    )


# =========================================================
# LOGOUT
# =========================================================

@app.route("/logout")
def logout():

    session.clear()

    return redirect(
        url_for("login")
    )


# =========================================================
# HOME / DASHBOARD
# =========================================================

@app.route("/")
def index():

    if "user_id" not in session:

        return redirect(
            url_for("login")
        )

    return render_template(
        "index.html"
    )


# =========================================================
# TEXT ANALYSIS
# =========================================================

@app.route("/text-analysis", methods=["GET", "POST"])
def text_analysis():

    if not login_required():

        return redirect(
            url_for("login")
        )

    if request.method == "POST":

        text = request.form.get(
            "analysis_text",
            ""
        ).strip()

        if not text:

            flash(
                "Please enter some text to analyse.",
                "error"
            )

            return redirect(
                url_for("text_analysis")
            )

        analysis = analyze_with_ollama(
            text
        )

        db = get_db_connection()

        if db is None:

            flash(
                "Analysis completed but database connection failed.",
                "error"
            )

            return render_template(
                "result.html",
                analysis=analysis,
                source_type="Text"
            )

        cursor = db.cursor()

        try:

            cursor.execute(
                """
                INSERT INTO ai_analysis
                (
                    upload_id,
                    analysis_text,
                    user_id,
                    ai_probability,
                    human_probability,
                    ai_verdict,
                    confidence,
                    humanization_score,
                    humanization_level,
                    reason
                )
                VALUES
                (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    None,
                    text,
                    session["user_id"],
                    analysis["ai_probability"],
                    analysis["human_probability"],
                    analysis["verdict"],
                    analysis["confidence"],
                    analysis["humanization_score"],
                    analysis["humanization_level"],
                    analysis["reason"]
                )
            )

            db.commit()

            analysis_id = cursor.lastrowid

        except mysql.connector.Error as err:

            db.rollback()

            print(
                "Text Analysis DB Error:",
                err
            )

            analysis_id = None

        finally:

            cursor.close()
            db.close()

        return render_template(
            "result.html",
            analysis=analysis,
            analysis_id=analysis_id,
            source_type="Text"
        )

    return render_template(
        "text_analysis.html"
    )


# =========================================================
# PDF ANALYSIS
# =========================================================

@app.route("/pdf-analysis", methods=["GET", "POST"])
def pdf_analysis():

    if not login_required():

        return redirect(
            url_for("login")
        )

    if request.method == "POST":

        file = request.files.get(
            "file"
        )

        if not file or not file.filename:

            flash(
                "Please select a PDF file.",
                "error"
            )

            return redirect(
                url_for("pdf_analysis")
            )

        if not allowed_file(file.filename):

            flash(
                "Only PDF files are allowed.",
                "error"
            )

            return redirect(
                url_for("pdf_analysis")
            )

        original_filename = file.filename

        safe_filename = re.sub(
            r"[^a-zA-Z0-9_.-]",
            "_",
            original_filename
        )

        file_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            safe_filename
        )

        file.save(file_path)

        extracted_text = extract_pdf_text(
            file_path
        )

        if not extracted_text:

            flash(
                "Could not extract readable text from this PDF.",
                "error"
            )

            return redirect(
                url_for("pdf_analysis")
            )

        analysis = analyze_with_ollama(
            extracted_text
        )

        db = get_db_connection()

        if db is None:

            return render_template(
                "result.html",
                analysis=analysis,
                source_type="PDF"
            )

        cursor = db.cursor()

        try:

            # Save upload
            cursor.execute(
                """
                INSERT INTO uploads
                (
                    user_id,
                    file_name,
                    file_type,
                    file_path
                )
                VALUES
                (%s, %s, %s, %s)
                """,
                (
                    session["user_id"],
                    original_filename,
                    "PDF",
                    file_path
                )
            )

            upload_id = cursor.lastrowid

            # Save extracted content
            cursor.execute(
                """
                INSERT INTO extracted_content
                (
                    upload_id,
                    extracted_text
                )
                VALUES
                (%s, %s)
                """,
                (
                    upload_id,
                    extracted_text
                )
            )

            # Save analysis
            cursor.execute(
                """
                INSERT INTO ai_analysis
                (
                    upload_id,
                    analysis_text,
                    user_id,
                    ai_probability,
                    human_probability,
                    ai_verdict,
                    confidence,
                    humanization_score,
                    humanization_level,
                    reason
                )
                VALUES
                (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    upload_id,
                    extracted_text,
                    session["user_id"],
                    analysis["ai_probability"],
                    analysis["human_probability"],
                    analysis["verdict"],
                    analysis["confidence"],
                    analysis["humanization_score"],
                    analysis["humanization_level"],
                    analysis["reason"]
                )
            )

            analysis_id = cursor.lastrowid

            db.commit()

        except mysql.connector.Error as err:

            db.rollback()

            print(
                "PDF Database Error:",
                err
            )

            analysis_id = None

        finally:

            cursor.close()
            db.close()

        return render_template(
            "result.html",
            analysis=analysis,
            analysis_id=analysis_id,
            source_type="PDF",
            filename=original_filename
        )

    return render_template(
        "upload.html",
        file_type="PDF",
        page_title="PDF Analyser",
        upload_route="pdf_analysis"
    )


# =========================================================
# IMAGE ANALYSIS
# =========================================================

@app.route("/image-analysis", methods=["GET", "POST"])
def image_analysis():

    if not login_required():

        return redirect(
            url_for("login")
        )

    if request.method == "POST":

        file = request.files.get(
            "file"
        )

        if not file or not file.filename:

            flash(
                "Please select an image.",
                "error"
            )

            return redirect(
                url_for("image_analysis")
            )

        extension = file.filename.rsplit(
            ".",
            1
        )[-1].lower()

        if extension not in {
            "jpg",
            "jpeg",
            "png",
            "webp"
        }:

            flash(
                "Only JPG, JPEG, PNG and WEBP images are allowed.",
                "error"
            )

            return redirect(
                url_for("image_analysis")
            )

        original_filename = file.filename

        safe_filename = re.sub(
            r"[^a-zA-Z0-9_.-]",
            "_",
            original_filename
        )

        file_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            safe_filename
        )

        file.save(file_path)

        extracted_text = extract_image_text(
            file_path
        )

        if not extracted_text:

            flash(
                "No readable text was found in the image.",
                "error"
            )

            return redirect(
                url_for("image_analysis")
            )

        analysis = analyze_with_ollama(
            extracted_text
        )

        db = get_db_connection()

        if db is None:

            return render_template(
                "result.html",
                analysis=analysis,
                source_type="Image"
            )

        cursor = db.cursor()

        try:

            # Save upload
            cursor.execute(
                """
                INSERT INTO uploads
                (
                    user_id,
                    file_name,
                    file_type,
                    file_path
                )
                VALUES
                (%s, %s, %s, %s)
                """,
                (
                    session["user_id"],
                    original_filename,
                    "IMAGE",
                    file_path
                )
            )

            upload_id = cursor.lastrowid

            # Save OCR text
            cursor.execute(
                """
                INSERT INTO extracted_content
                (
                    upload_id,
                    extracted_text
                )
                VALUES
                (%s, %s)
                """,
                (
                    upload_id,
                    extracted_text
                )
            )

            # Save analysis
            cursor.execute(
                """
                INSERT INTO ai_analysis
                (
                    upload_id,
                    analysis_text,
                    user_id,
                    ai_probability,
                    human_probability,
                    ai_verdict,
                    confidence,
                    humanization_score,
                    humanization_level,
                    reason
                )
                VALUES
                (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    upload_id,
                    extracted_text,
                    session["user_id"],
                    analysis["ai_probability"],
                    analysis["human_probability"],
                    analysis["verdict"],
                    analysis["confidence"],
                    analysis["humanization_score"],
                    analysis["humanization_level"],
                    analysis["reason"]
                )
            )

            analysis_id = cursor.lastrowid

            db.commit()

        except mysql.connector.Error as err:

            db.rollback()

            print(
                "Image Database Error:",
                err
            )

            analysis_id = None

        finally:

            cursor.close()
            db.close()

        return render_template(
            "result.html",
            analysis=analysis,
            analysis_id=analysis_id,
            source_type="Image",
            filename=original_filename
        )

    return render_template(
        "upload.html",
        file_type="Image",
        page_title="Image Analyser",
        upload_route="image_analysis"
    )


# =========================================================
# HISTORY
# =========================================================

@app.route("/history")
def history():

    if not login_required():

        return redirect(
            url_for("login")
        )

    db = get_db_connection()

    if db is None:

        flash(
            "Database connection error.",
            "error"
        )

        return redirect(
            url_for("index")
        )

    cursor = db.cursor(
        dictionary=True
    )

    try:

        cursor.execute(
            """
            SELECT
                a.analysis_id,
                a.ai_probability,
                a.human_probability,
                a.ai_verdict,
                a.confidence,
                a.humanization_score,
                a.humanization_level,
                a.reason,
                a.analyzed_at,
                a.analysis_text,
                u.file_name,
                u.file_type
            FROM ai_analysis a
            LEFT JOIN uploads u
                ON a.upload_id = u.upload_id
            WHERE a.user_id = %s
            ORDER BY a.analyzed_at DESC
            """,
            (session["user_id"],)
        )

        history_data = cursor.fetchall()

        return render_template(
            "history.html",
            history=history_data
        )

    except mysql.connector.Error as err:

        print(
            "History Database Error:",
            err
        )

        return render_template(
            "history.html",
            history=[]
        )

    finally:

        cursor.close()
        db.close()


# =========================================================
# VIEW OLD RESULT
# =========================================================

@app.route("/view-result/<int:analysis_id>")
def view_result(analysis_id):

    if not login_required():

        return redirect(
            url_for("login")
        )

    db = get_db_connection()

    if db is None:

        flash(
            "Database connection error.",
            "error"
        )

        return redirect(
            url_for("history")
        )

    cursor = db.cursor(
        dictionary=True
    )

    try:

        cursor.execute(
            """
            SELECT
                a.analysis_id,
                a.analysis_text,
                a.ai_probability,
                a.human_probability,
                a.ai_verdict,
                a.confidence,
                a.humanization_score,
                a.humanization_level,
                a.reason,
                a.analyzed_at,
                u.file_name,
                u.file_type
            FROM ai_analysis a
            LEFT JOIN uploads u
                ON a.upload_id = u.upload_id
            WHERE
                a.analysis_id = %s
                AND a.user_id = %s
            """,
            (
                analysis_id,
                session["user_id"]
            )
        )

        result = cursor.fetchone()

        if not result:

            flash(
                "Analysis result not found.",
                "error"
            )

            return redirect(
                url_for("history")
            )

        analysis = {
            "ai_probability": result["ai_probability"],
            "human_probability": result["human_probability"],
            "verdict": result["ai_verdict"],
            "confidence": result["confidence"],
            "humanization_score": result["humanization_score"],
            "humanization_level": result["humanization_level"],
            "reason": result["reason"]
        }

        return render_template(
            "result.html",
            analysis=analysis,
            analysis_id=result["analysis_id"],
            source_type=result["file_type"] or "Text",
            filename=result["file_name"],
            old_result=True,
            analysis_text=result["analysis_text"],
            analyzed_at=result["analyzed_at"]
        )

    except mysql.connector.Error as err:

        print(
            "View Result Database Error:",
            err
        )

        flash(
            "Could not load result.",
            "error"
        )

        return redirect(
            url_for("history")
        )

    finally:

        cursor.close()
        db.close()


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    print("\n====================================")
    print("       AI ANALYSE APPLICATION")
    print("====================================")

    db = get_db_connection()

    if db:

        print("MySQL Database Connected Successfully!")

        db.close()

    else:

        print("MySQL Database Connection Failed!")

    print("Ollama Model:", OLLAMA_MODEL)
    print("====================================\n")

    app.run(
        debug=True
    )
