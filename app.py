from flask import Flask, render_template, request, jsonify, send_file
import requests
import json
import pdfkit
from datetime import datetime
import os
import tempfile
import PyPDF2
import io

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ollama API configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gpt-oss:20b"  # Using your available model

def extract_pdf_text(pdf_file):
    """Extract text from uploaded PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

def query_ollama(prompt, model=MODEL_NAME):
    """Send a query to Ollama and get response"""
    try:
        # First, let's check what models are available
        models_response = requests.get("http://localhost:11434/api/tags")
        if models_response.status_code == 200:
            available_models = [m['name'] for m in models_response.json().get('models', [])]
            print(f"Available models: {available_models}")
            
            # Try to find the correct model name
            if model not in available_models:
                # Look for similar model names
                for available_model in available_models:
                    if 'gpt' in available_model.lower() or 'oss' in available_model.lower():
                        print(f"Using model: {available_model} instead of {model}")
                        model = available_model
                        break
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }
        
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        
        result = response.json()
        return result.get("response", "No response received")
        
    except requests.exceptions.RequestException as e:
        return f"Error connecting to Ollama: {str(e)}"
    except json.JSONDecodeError as e:
        return f"Error parsing response: {str(e)}"

def html_to_pdf(html_content, output_path):
    """Convert HTML content to PDF"""
    try:
        # Configure wkhtmltopdf options
        options = {
            'page-size': 'A4',
            'margin-top': '0.75in',
            'margin-right': '0.75in',
            'margin-bottom': '0.75in',
            'margin-left': '0.75in',
            'encoding': "UTF-8",
            'no-outline': None
        }
        
        pdfkit.from_string(html_content, output_path, options=options)
        return True
        
    except Exception as e:
        print(f"PDF generation error: {str(e)}")
        return False

@app.route('/')
def index():
    """Main page with form"""
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_content():
    """Generate content using Ollama"""
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')
        
        if not prompt:
            return jsonify({'error': 'No prompt provided'}), 400
        
        # Query Ollama
        response = query_ollama(prompt)
        
        return jsonify({
            'success': True,
            'response': response,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze_pdf', methods=['POST'])
def analyze_pdf():
    """Upload and analyze PDF with Ollama"""
    try:
        if 'pdf_file' not in request.files:
            return jsonify({'error': 'No PDF file uploaded'}), 400
        
        pdf_file = request.files['pdf_file']
        analysis_prompt = request.form.get('analysis_prompt', 'Summarize this document')
        
        if pdf_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not pdf_file.filename.lower().endswith('.pdf'):
            return jsonify({'error': 'Only PDF files are allowed'}), 400
        
        # Extract text from PDF
        pdf_text = extract_pdf_text(pdf_file)
        
        if pdf_text.startswith('Error'):
            return jsonify({'error': pdf_text}), 500
        
        # Create prompt for Ollama
        full_prompt = f"{analysis_prompt}\n\nDocument content:\n{pdf_text[:4000]}"  # Limit to first 4000 chars
        
        # Query Ollama
        response = query_ollama(full_prompt)
        
        return jsonify({
            'success': True,
            'analysis': response,
            'pdf_filename': pdf_file.filename,
            'pdf_text_preview': pdf_text[:500] + "..." if len(pdf_text) > 500 else pdf_text,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
def generate_pdf():
    """Generate PDF from HTML content"""
    try:
        data = request.get_json()
        html_content = data.get('html_content', '')
        filename = data.get('filename', 'output.pdf')
        
        if not html_content:
            return jsonify({'error': 'No HTML content provided'}), 400
        
        # Create temporary file
        temp_dir = tempfile.gettempdir()
        pdf_path = os.path.join(temp_dir, filename)
        
        # Generate PDF
        if html_to_pdf(html_content, pdf_path):
            return send_file(
                pdf_path,
                as_attachment=True,
                download_name=filename,
                mimetype='application/pdf'
            )
        else:
            return jsonify({'error': 'Failed to generate PDF'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/models')
def list_models():
    """List available Ollama models"""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json().get('models', [])
            return jsonify({
                'models': [{'name': m['name'], 'size': m.get('size', 'Unknown')} for m in models]
            })
        else:
            return jsonify({'error': 'Failed to fetch models'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    """Check if Ollama is running"""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            return jsonify({'status': 'Ollama is running', 'models': response.json()})
        else:
            return jsonify({'status': 'Ollama connection failed'}), 500
    except:
        return jsonify({'status': 'Ollama not accessible'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)