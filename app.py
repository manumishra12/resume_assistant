from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import google.generativeai as genai
import PyPDF2
import re
import time
from difflib import SequenceMatcher
import os
from werkzeug.utils import secure_filename
import docx2txt
import tempfile

# Configuration
GOOGLE_API_KEY = "AIzaSyBFni23FxQZIKsdP2bMLrJrMuP_YwQy3M4"
RESUME_PATH = "C:\Users\conne\Desktop\resume_bot\Resume_ManuMishra.pdf"
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx'}

app = Flask(__name__)
CORS(app)  # Enable CORS for all domains
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max file size

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Default resume content based on Manu Mishra's actual resume
DEFAULT_RESUME_CONTENT = """
Manu Mishra
+91-8920429057 | connectmanumishra@gmail.com | linkedin.com/in/manu12mishra | github.com/manumishra12

Education
Vellore Institute of Technology, Chennai (Sep. 2021 – Sep 2025)
Bachelor of Technology in Computer Science | CGPA: 8.20
• Sponsorship and Management Lead at REVA, Special Team
• Programming and Electrical Member at HumanoidX
• Content and Technical member at VITrendZ and Webster's

Experience
AI Engineer Intern - Mondee (Present, Hyderabad - onsite)
• Worked on GenAI and designing prompts for Aarna, Mondee's AI app, to generate personalized travel experiences and dynamic itineraries
• Built multi-agent Agentic AI workflows using LangChain, LangGraph, custom ML models, APIs, and full-stack tools

Software Developer Intern - DigiQuanta (Jan 2025 - Mar 2025, Hyderabad - onsite)
• Developed a lip-syncing pipeline for dubbed videos by aligning audio with visual frames using facial keypoint detection, computer vision, and temporal modeling
• Built a voice-enabled medical assistant using speech recognition and GenAI APIs to provide real-time responses for preliminary diagnostics and health queries

Junior Researcher - Indian Institute of Technology, BHU (June 2024 – July 2024, Varanasi - onsite)
• Engineered a Cuffless Blood Pressure estimation system utilizing ECG, PPG, and advanced machine learning techniques for precise and non-invasive monitoring
• Enhanced healthcare solutions through innovative signal processing and real-time data analysis

Projects
SmartGrader (VLM, LLM, Unsloth, Llamafactory)
• Built an AI grading system using fine-tuned LLMs and Qwen-VL to interpret handwritten diagrams, pseudocode and structured content
• Enhanced multimodal reasoning and problem-solving by incorporating Gemma3 for instructions and analytical depth

Nexus Search (Agentic Search, LangChain, LangGraph, Tavily)
• Nexus Search delivers intelligent, multi-modal search and content analysis through a 13-agent architecture built with LangChain, LangGraph, and Tavily enabling professional document generation and advanced query resolution

Talk2Me (MERN, Socket.io, TailwindCSS, Daisy UI, Zustand)
• Developed a MERN stack real-time chat app with JWT authentication, Socket.io messaging, online status, state management, and robust error handling

Technical Skills
Languages: Python, Java, C++, ReactJS, NextJS, JavaScript
Developer Tools: Cloud basics, MongoDB, SQL, NodeJS, ExpressJS, Figma, AdobeXD
Technologies: Git/GitHub, Machine Learning, Deep Learning, Generative AI, LangChain/Graph
Relevant Coursework: DSA, OOPS, System Design, Software Engineering

Publications
Cuffless BP Estimation (Deep Learning, Machine Learning, SWIN Transformers) - Published
• Developed cuffless blood pressure estimation system using bispectrum and bicoherence images for accurate predictions, with a SWIN Transformer model for non-invasive monitoring

A System and Method for Automated Waste Segregation (Application no: 202441064693) - Patented
• An AI-powered machine that automates construction waste segregation, enhancing recycling efficiency and sustainability by diverting waste from landfills and integrating with waste management systems

Achievements
• Secured 15th place among 100+ teams in the Smart India Hackathon 2023 as a grand finalist
• Won 2nd position in the Smart India Hackathon 2022 for the "Money Plant" smart bin for waste segregation
• Got first prize in an IoT Hackathon for an automated solution for waste management
• Received the Rookie Team Award at Hackhub 2022 for a Smart Attendance System supporting online and offline use
"""

# Predefined Q&A pairs
FIXED_QA = {
    "what should we know about your life story in a few sentences": 
        "I'm a driven and curious learner with a passion for solving real-world problems using technology. I come from a background in engineering and have worked on impactful projects ranging from AI to full-stack development.",
    
    "what's your #1 superpower": 
        "My superpower is adaptability — I can quickly learn, pivot, and deliver in dynamic environments.",

    "what are the top 3 areas you'd like to grow in": 
        "First, I'd like to enhance my leadership and mentorship skills. Second, I want to deepen my expertise in AI. Third, I want to improve my public speaking and storytelling.",

    "what misconception do your coworkers have about you": 
        "People often think I prefer working solo, but I actually thrive in collaborative environments — I just like to think things through quietly first.",

    "how do you push your boundaries and limits": 
        "I regularly take on challenges that scare me a bit — whether it's a new tech stack, a live demo, or a leadership role. I believe growth lies just outside the comfort zone."
}

class VoiceAssistantBackend:
    def __init__(self):
        # Configure Gemini AI
        genai.configure(api_key=GOOGLE_API_KEY)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Resume content storage
        self.default_resume_content = DEFAULT_RESUME_CONTENT
        self.uploaded_resume_content = None
        self.current_resume = 'default'
        
        # Try to load original resume file if available
        try:
            original_resume = self.load_resume_from_file()
            if original_resume:
                self.default_resume_content = original_resume
        except Exception as e:
            print(f"Could not load original resume file, using default content: {e}")
    
    def allowed_file(self, filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
    def load_resume_from_file(self):
        """Load and extract text from resume PDF file"""
        if not os.path.exists(RESUME_PATH):
            return None
            
        try:
            with open(RESUME_PATH, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                return text
        except Exception as e:
            print(f"Error loading resume file: {e}")
            return None
    
    def process_uploaded_file(self, file_path):
        """Process uploaded resume file and extract text"""
        try:
            file_extension = file_path.rsplit('.', 1)[1].lower()
            
            if file_extension == 'pdf':
                return self.extract_text_from_pdf(file_path)
            elif file_extension in ['doc', 'docx']:
                return self.extract_text_from_docx(file_path)
            else:
                raise ValueError("Unsupported file format")
                
        except Exception as e:
            raise Exception(f"Error processing file: {str(e)}")
    
    def extract_text_from_pdf(self, file_path):
        """Extract text from PDF file"""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text.strip()
    
    def extract_text_from_docx(self, file_path):
        """Extract text from DOCX file"""
        try:
            import docx
            doc = docx.Document(file_path)
            text = []
            for paragraph in doc.paragraphs:
                text.append(paragraph.text)
            return '\n'.join(text)
        except ImportError:
            # Fallback to docx2txt if python-docx is not available
            return docx2txt.process(file_path)
    
    def get_current_resume_content(self):
        """Get the currently active resume content"""
        if self.current_resume == 'uploaded' and self.uploaded_resume_content:
            return self.uploaded_resume_content
        else:
            return self.default_resume_content
    
    def set_resume_type(self, resume_type):
        """Set the current resume type to use"""
        self.current_resume = resume_type
    
    def similarity(self, a, b):
        """Calculate similarity between two strings"""
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()
    
    def find_best_match(self, question):
        """Find the best matching predefined question"""
        best_match = None
        best_score = 0
        threshold = 0.6  # Minimum similarity threshold
        
        for predefined_q in FIXED_QA.keys():
            score = self.similarity(question, predefined_q)
            if score > best_score and score > threshold:
                best_score = score
                best_match = predefined_q
        
        return best_match, best_score
    
    def clean_text_for_speech(self, text):
        """Clean text by removing markdown formatting and unwanted characters for TTS"""
        if not text:
            return ""
        
        # Remove markdown formatting
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Remove **bold**
        text = re.sub(r'\*(.*?)\*', r'\1', text)      # Remove *italic*
        text = re.sub(r'__(.*?)__', r'\1', text)      # Remove __bold__
        text = re.sub(r'_(.*?)_', r'\1', text)        # Remove _italic_
        
        # Remove markdown headers
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
        
        # Remove markdown links [text](url) - keep just the text
        text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
        
        # Remove markdown code blocks
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        text = re.sub(r'`(.*?)`', r'\1', text)  # Inline code
        
        # Remove markdown lists markers
        text = re.sub(r'^[\s]*[-*+]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
        
        # Remove extra whitespace and newlines
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters that don't contribute to speech
        text = re.sub(r'[#@$%^&(){}[\]|\\~`]', '', text)
        
        # Clean up common formatting artifacts
        text = text.replace('•', '')  # Bullet points
        text = text.replace('→', 'to')  # Arrows
        text = text.replace('–', '-')  # Em dash
        text = text.replace('—', '-')  # En dash
        
        # Remove multiple periods
        text = re.sub(r'\.{2,}', '.', text)
        
        # Clean up quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        return text.strip()
    
    def get_answer(self, question, resume_type='default'):
        """Get answer for the given question using specified resume"""
        try:
            # Set the current resume type
            self.set_resume_type(resume_type)
            
            # First, check if it matches any predefined questions
            best_match, score = self.find_best_match(question)
            
            if best_match:
                answer = FIXED_QA[best_match]
                return {
                    'answer': answer,
                    'source': 'predefined',
                    'cleaned_answer': self.clean_text_for_speech(answer)
                }
            
            # If no match found, use Gemini to search current resume
            current_resume = self.get_current_resume_content()
            if current_resume:
                prompt = f"""
                Based on the following resume content, answer the question: "{question}"
                
                Resume Content:
                {current_resume}
                
                Please provide a concise and conversational answer based on the information in the resume. 
                Use natural speech patterns without markdown formatting, asterisks, or special characters.
                Speak as if you are the person whose resume this is, using first person (I, my, me).
                If the information is not available in the resume, say "I don't have that specific information in my resume."
                
                Keep the response under 100 words and suitable for text-to-speech conversion.
                """
                
                response = self.model.generate_content(prompt)
                answer = response.text
                
                return {
                    'answer': answer,
                    'source': 'resume',
                    'cleaned_answer': self.clean_text_for_speech(answer)
                }
            else:
                return {
                    'answer': "Sorry, I couldn't load the resume content to answer your question.",
                    'source': 'error',
                    'cleaned_answer': "Sorry, I couldn't load the resume content to answer your question."
                }
                
        except Exception as e:
            error_msg = f"Sorry, I encountered an error while processing your question: {str(e)}"
            return {
                'answer': error_msg,
                'source': 'error',
                'cleaned_answer': error_msg
            }

# Initialize the assistant
assistant = VoiceAssistantBackend()

@app.route('/')
def index():
    """Serve the main HTML file"""
    return send_from_directory('.', 'index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({'error': 'Message is required'}), 400
        
        message = data['message'].strip()
        if not message:
            return jsonify({'error': 'Message cannot be empty'}), 400
        
        # Get resume type from request (default or uploaded)
        resume_type = data.get('resume_type', 'default')
        
        # Get response from assistant
        result = assistant.get_answer(message, resume_type)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'error': f'Internal server error: {str(e)}',
            'answer': 'Sorry, something went wrong. Please try again.',
            'source': 'error'
        }), 500

@app.route('/api/upload-resume', methods=['POST'])
def upload_resume():
    """Handle resume file upload"""
    try:
        if 'resume' not in request.files:
            return jsonify({'error': 'No resume file provided'}), 400
        
        file = request.files['resume']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not assistant.allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed. Please upload PDF, DOC, or DOCX files.'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        try:
            # Process the uploaded file
            extracted_content = assistant.process_uploaded_file(file_path)
            
            # Store the extracted content
            assistant.uploaded_resume_content = extracted_content
            assistant.current_resume = 'uploaded'
            
            return jsonify({
                'success': True,
                'message': 'Resume uploaded successfully',
                'content': extracted_content,
                'filename': filename
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Error processing file: {str(e)}'
            }), 400
            
        finally:
            # Clean up uploaded file
            try:
                os.remove(file_path)
            except:
                pass
                
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Upload failed: {str(e)}'
        }), 500

@app.route('/api/resume', methods=['GET'])
def get_resume():
    """Get current resume content"""
    try:
        current_content = assistant.get_current_resume_content()
        if current_content:
            return jsonify({
                'content': current_content,
                'status': 'success',
                'type': assistant.current_resume
            })
        else:
            return jsonify({
                'content': '',
                'status': 'error',
                'message': 'Resume content not available'
            })
    except Exception as e:
        return jsonify({
            'content': '',
            'status': 'error',
            'message': f'Error loading resume: {str(e)}'
        }), 500

@app.route('/api/resume/reset', methods=['POST'])
def reset_resume():
    """Reset to default resume"""
    try:
        assistant.current_resume = 'default'
        assistant.uploaded_resume_content = None
        
        return jsonify({
            'success': True,
            'message': 'Reset to default resume',
            'content': assistant.default_resume_content
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error resetting resume: {str(e)}'
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time(),
        'default_resume_available': bool(assistant.default_resume_content),
        'uploaded_resume_available': bool(assistant.uploaded_resume_content),
        'current_resume': assistant.current_resume,
        'gemini_configured': bool(GOOGLE_API_KEY)
    })

@app.route('/api/predefined-questions', methods=['GET'])
def get_predefined_questions():
    """Get list of predefined questions"""
    return jsonify({
        'questions': list(FIXED_QA.keys()),
        'count': len(FIXED_QA)
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Create a simple HTML file if it doesn't exist
    if not os.path.exists('index.html'):
        print("Note: Create an index.html file with the frontend code")
        print("The frontend HTML should make requests to /api/chat endpoint")
    
    print("Starting Voice Assistant Backend...")
    print(f"Default resume available: {bool(assistant.default_resume_content)}")
    print(f"Upload folder created: {os.path.exists(UPLOAD_FOLDER)}")
    print(f"Gemini API configured: {bool(GOOGLE_API_KEY)}")
    print("\nAPI Endpoints:")
    print("- POST /api/chat - Send chat messages")
    print("- POST /api/upload-resume - Upload resume file")
    print("- GET /api/resume - Get current resume content")
    print("- POST /api/resume/reset - Reset to default resume")
    print("- GET /api/health - Health check")
    print("- GET /api/predefined-questions - Get predefined questions")
    print("\nSupported file formats: PDF, DOC, DOCX")
    print("Max file size: 10MB")
    print("\nStarting server on http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)