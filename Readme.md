# resume_assistant

- Key Features & Setup

Predefined Q&A - Instant answers for 5 common interview questions with smart matching
Default Resume Analysis - AI answers questions about Manu Mishra's pre-loaded professional background
Custom Resume Upload - Upload PDF/DOC/DOCX, AI instantly switches to analyze your resume content

## 💻 **Setup Commands**

1. **Install Dependencies**
   ```bash
   pip install flask flask-cors google-generativeai PyPDF2 python-docx docx2txt werkzeug
   ```

2. **Configure API Key** (in app.py)
   ```python
   GOOGLE_API_KEY = "your_google_api_key_here"
   ```

3. **Save Files** - Copy provided code into:
   - `app.py` (backend)
   - `index.html` (frontend)

4. **Run Application**
   ```bash
   python app.py
   ```

5. **Access Interface**
   ```
   http://localhost:5000
   ```
