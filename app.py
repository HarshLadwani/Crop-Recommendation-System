from flask import Flask, render_template, request, redirect, url_for, session, send_file
import joblib
import numpy as np
import sqlite3
import bcrypt
from datetime import datetime
import pdfkit
from io import BytesIO

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load trained model
model = joblib.load('crop_recommendation_model.pkl')

# PDFKit configuration
config = pdfkit.configuration(wkhtmltopdf=r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe')

# Initialize database
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute(''' 
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password'].encode('utf-8')
        hashed = bcrypt.hashpw(password, bcrypt.gensalt())

        try:
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed))
            conn.commit()
            conn.close()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            return 'Username already exists!'
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password'].encode('utf-8')

        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('SELECT * FROM users WHERE username = ?', (username,))
        user = c.fetchone()
        conn.close()

        if user and bcrypt.checkpw(password, user[2]):
            session['user'] = username
            return redirect(url_for('recommend'))
        else:
            return 'Invalid username or password!'
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('home'))

@app.route('/recommend', methods=['GET'])
def recommend():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('recommend.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'user' not in session:
        return redirect(url_for('login'))

    try:
        form_values = {
            'N': request.form['N'],
            'P': request.form['P'],
            'K': request.form['K'],
            'temperature': request.form['temperature'],
            'humidity': request.form['humidity'],
            'ph': request.form['ph'],
            'rainfall': request.form['rainfall']
        }

        session['form_values'] = form_values
        return redirect(url_for('generate_report'))

    except Exception as e:
        return render_template('recommend.html', prediction_text=f"Error: {str(e)}")

@app.route('/generate_report')
def generate_report():
    if 'user' not in session or 'form_values' not in session:
        return redirect(url_for('login'))

    try:
        form_values = {k: float(v) for k, v in session['form_values'].items()}

        prediction = model.predict([list(form_values.values())])[0]
        probabilities = model.predict_proba([list(form_values.values())])[0]
        classes = model.classes_

        top_indices = np.argsort(probabilities)[::-1][:3]
        top_crops = [(classes[i], round(probabilities[i] * 100, 2)) for i in top_indices]

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return render_template('report.html',
                               user=session['user'],
                               input_values=form_values,
                               prediction=prediction,
                               top_crops=top_crops,
                               timestamp=timestamp)
    except Exception as e:
        return f"Error in generating report: {str(e)}"

@app.route('/download_report', methods=['POST'])
def download_report():
    if 'user' not in session:
        return redirect(url_for('login'))

    try:
        form_values = {
            'N': float(request.form['N']),
            'P': float(request.form['P']),
            'K': float(request.form['K']),
            'temperature': float(request.form['temperature']),
            'humidity': float(request.form['humidity']),
            'ph': float(request.form['ph']),
            'rainfall': float(request.form['rainfall'])
        }

        prediction = model.predict([list(form_values.values())])[0]
        probabilities = model.predict_proba([list(form_values.values())])[0]
        classes = model.classes_

        top_indices = np.argsort(probabilities)[::-1][:3]
        top_crops = [(classes[i], round(probabilities[i] * 100, 2)) for i in top_indices]

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        rendered_html = render_template('report.html',
                                        user=session['user'],
                                        input_values=form_values,
                                        prediction=prediction,
                                        top_crops=top_crops,
                                        timestamp=timestamp)

        pdf = pdfkit.from_string(rendered_html, False, configuration=config)

        return send_file(BytesIO(pdf),
                         as_attachment=True,
                         download_name="crop_recommendation_report.pdf",
                         mimetype='application/pdf')

    except Exception as e:
        return f"Error in generating PDF report: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)