

# Resume Assistant 

AI-powered tool for analyzing resumes and answering common interview questions.

---

## Key Features

### 1. Predefined Q\&A

Provides instant, intelligent answers to five common interview questions using keyword matching.:
 - What should we know about your life story in a few sentences?
 - What’s your #1 superpower?
 - What are the top 3 areas you’d like to grow in?
 - What misconception do your coworkers have about you?
 - How do you push your boundaries and limits?

### 2. Default Resume Analysis

Analyzes a preloaded resume (e.g., Manu Mishra) to generate personalized interview responses.

### 3. Custom Resume Upload

Upload your own resume (PDF, DOC, or DOCX), and the assistant will automatically analyze and switch context to your document.

---

## Setup Instructions

### Step 1: Install Dependencies

Run the following command in your terminal to install required Python packages:

```bash
pip install flask flask-cors google-generativeai PyPDF2 python-docx docx2txt werkzeug
```

---

### Step 2: Configure API Key

In `app.py`, set your Google Generative AI API key:

```python
GOOGLE_API_KEY = "your_google_api_key_here"
```

---

### Step 3: Save Project Files

Create the following files and paste the respective code:

* `app.py` – Backend logic
* `index.html` – Frontend user interface

---

### Step 4: Start the Application

Run the Flask app using:

```bash
python app.py
```

---

### Step 5: Open in Browser

Once the server is running, access the application at:

```
http://localhost:5000
```
