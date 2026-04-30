
import os
import sys
import warnings
import logging
import re
from xml.sax.saxutils import escape
import zipfile
import json
import tempfile
import shutil

# Must be set BEFORE any tensorflow/keras imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KERAS_BACKEND'] = 'tensorflow'
warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from flask import Flask, render_template, request, url_for, send_file, jsonify
from werkzeug.utils import secure_filename
import numpy as np
from keras.models import load_model
from keras.utils import img_to_array, load_img
from keras.applications.efficientnet import preprocess_input
import requests
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

API_KEY = os.getenv("OPENROUTER_API_KEY")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FONT_PATHS = {
    "telugu": os.path.join(BASE_DIR, "fonts", "NotoSansTelugu-Regular.ttf", "static", "NotoSansTelugu-Regular.ttf"),
    "hindi":  os.path.join(BASE_DIR, "fonts", "NotoSansDevanagari-Regular.ttf", "static", "NotoSansDevanagari-Regular.ttf"),
    "default": os.path.join(BASE_DIR, "fonts", "NotoSans-Regular.ttf", "static", "NotoSans-Regular.ttf")
}

LANG_LABELS = {
    "english": {
        "title": "Fish Disease Detection Report",
        "subtitle": "AI-Powered Aquaculture Health Analysis",
        "detection_result": "Detection Result",
        "diagnosis": "Diagnosis",
        "confidence": "Confidence",
        "top3": "Top 3 Predictions",
        "disease_info": "Disease Information",
        "symptoms": "Symptoms",
        "causes": "Causes",
        "treatment": "Treatment",
        "prevention": "Prevention",
        "ai_advice": "AI Assistant Suggestions"
    },
    "hindi": {
        "title": "मछली रोग पहचान रिपोर्ट",
        "subtitle": "AI-संचालित एक्वाकल्चर स्वास्थ्य विश्लेषण",
        "detection_result": "पहचान परिणाम",
        "diagnosis": "निदान",
        "confidence": "आत्मविश्वास",
        "top3": "शीर्ष 3 भविष्यवाणियां",
        "disease_info": "रोग की जानकारी",
        "symptoms": "लक्षण",
        "causes": "कारण",
        "treatment": "उपचार",
        "prevention": "रोकथाम",
        "ai_advice": "AI सहायक सुझाव"
    },
    "telugu": {
        "title": "చేపల వ్యాధి నిర్ధారణ నివేదిక",
        "subtitle": "AI-ఆధారిత ఆక్వాకల్చర్ ఆరోగ్య విశ్లేషణ",
        "detection_result": "నిర్ధారణ ఫలితం",
        "diagnosis": "వ్యాధి నిర్ధారణ",
        "confidence": "విశ్వాసం",
        "top3": "టాప్ 3 అంచనాలు",
        "disease_info": "వ్యాధి సమాచారం",
        "symptoms": "లక్షణాలు",
        "causes": "కారణాలు",
        "treatment": "చికిత్స",
        "prevention": "నివారణ",
        "ai_advice": "AI అసిస్టెంట్ సూచనలు"
    }
}

print("[*] Checking fonts...")
for lang, path in FONT_PATHS.items():
    print(f"  {lang}: {os.path.exists(path)}")

# ---------------------------------------------------------------------------
# Null writer to suppress TF stdout noise
# ---------------------------------------------------------------------------
class NullWriter:
    def write(self, s): pass
    def flush(self): pass

# ---------------------------------------------------------------------------
# Patch-loader: fixes Keras 3.13 model on Keras 3.12 environment
# ---------------------------------------------------------------------------
def patch_config(cfg):
    if isinstance(cfg, dict):
        cls   = cfg.get('class_name', '')
        inner = cfg.get('config', {})
        if cls == 'InputLayer' and isinstance(inner, dict):
            if 'batch_shape' in inner and 'shape' not in inner:
                inner['shape'] = inner.pop('batch_shape')[1:]
            inner.pop('optional', None)
        if cls in ('Dense', 'BatchNormalization') and isinstance(inner, dict):
            inner.pop('quantization_config', None)
        return {k: patch_config(v) for k, v in cfg.items()}
    elif isinstance(cfg, list):
        return [patch_config(item) for item in cfg]
    return cfg

def patch_and_load(model_path):
    import keras as _keras
    tmpdir = tempfile.mkdtemp()
    patched = os.path.join(tmpdir, 'patched_model.keras')
    try:
        with zipfile.ZipFile(model_path, 'r') as zin:
            contents = {n: zin.read(n) for n in zin.namelist()}
        cfg_patched = patch_config(json.loads(contents['config.json']))
        contents['config.json'] = json.dumps(cfg_patched).encode('utf-8')
        meta = json.loads(contents.get('metadata.json', b'{}'))
        meta['keras_version'] = _keras.__version__
        contents['metadata.json'] = json.dumps(meta).encode('utf-8')
        with zipfile.ZipFile(patched, 'w', zipfile.ZIP_DEFLATED) as zout:
            for name, data in contents.items():
                zout.writestr(name, data)
        return load_model(patched, compile=False)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
print("[*] Loading model...")
try:
    old_err, old_out = sys.stderr, sys.stdout
    sys.stderr = sys.stdout = NullWriter()
    model = patch_and_load("final_fish_model.keras")
    sys.stderr, sys.stdout = old_err, old_out
    print("[+] Model loaded successfully")
except Exception as e:
    sys.stderr, sys.stdout = old_err, old_out
    print(f"[!] Model loading error: {str(e)[:120]}")
    model = None

# ---------------------------------------------------------------------------
# Class names & disease knowledge base
# ---------------------------------------------------------------------------
class_names = [
    'Bacterial Red disease',
    'Bacterial diseases - Aeromoniasis',
    'Bacterial gill disease',
    'Fungal diseases Saprolegniasis',
    'Healthy Fish',
    'Parasitic diseases',
    'Viral diseases White tail disease'
]

DISEASE_INFO = {
    'Bacterial Red disease': {
        'symptoms': 'Red or hemorrhagic lesions on skin, fins, and gills; lethargy; loss of appetite.',
        'causes': 'Bacteria such as Aeromonas hydrophila; triggered by poor water quality and stress.',
        'treatment': 'Antibiotics (oxytetracycline); salt baths; improve water quality.',
        'prevention': 'Maintain clean water; avoid overcrowding; quarantine new fish.'
    },
    'Bacterial diseases - Aeromoniasis': {
        'symptoms': 'Ulcers, hemorrhages, dropsy (swollen abdomen), pop-eye, fin rot.',
        'causes': 'Aeromonas bacteria; opportunistic in stressed or immunocompromised fish.',
        'treatment': 'Antibiotics; medicated food; improve tank conditions.',
        'prevention': 'Good water quality; stress reduction; balanced nutrition.'
    },
    'Bacterial gill disease': {
        'symptoms': 'Rapid breathing, gaping at surface, pale or swollen gills, mucus excess.',
        'causes': 'Flavobacterium branchiophilum; overcrowding; poor oxygenation.',
        'treatment': 'Chloramine-T treatment; antibiotics; salt baths.',
        'prevention': 'Maintain dissolved oxygen; avoid overcrowding; regular water changes.'
    },
    'Fungal diseases Saprolegniasis': {
        'symptoms': 'White/grey cotton-like patches on skin, fins or eggs.',
        'causes': 'Saprolegnia fungus; wounds, stress, or poor water quality trigger infection.',
        'treatment': 'Malachite green; potassium permanganate; salt baths.',
        'prevention': 'Good water quality; prevent injuries; remove dead tissue promptly.'
    },
    'Healthy Fish': {
        'symptoms': 'No symptoms — fish appears normal and active.',
        'causes': 'N/A',
        'treatment': 'No treatment needed.',
        'prevention': 'Maintain balanced diet, clean water, and regular health monitoring.'
    },
    'Parasitic diseases': {
        'symptoms': 'Scratching/flashing, white spots, mucus excess, fin damage, weight loss.',
        'causes': 'External parasites (Ich, flukes, anchor worms, lice) or internal parasites.',
        'treatment': 'Antiparasitic medications; formalin/salt baths; copper-based treatments.',
        'prevention': 'Quarantine new fish; maintain clean water; regular inspection.'
    },
    'Viral diseases White tail disease': {
        'symptoms': 'White discolouration of tail/body, abnormal swimming, lethargy, high mortality.',
        'causes': 'Macrobrachium rosenbergii nodavirus (MrNV); spreads rapidly in crowded tanks.',
        'treatment': 'No specific cure; supportive care; remove affected fish immediately.',
        'prevention': 'Source pathogen-free stock; strict biosecurity; disinfect equipment.'
    }
}

# ---------------------------------------------------------------------------
# AI helpers
# ---------------------------------------------------------------------------
def call_openrouter(messages, timeout=30):
    """Call OpenRouter API and return content string or None."""
    if not API_KEY:
        print("[!] API_KEY not found in environment.")
        return None
    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:5000",
                "X-Title": "AquaHealth AI"
            },
            json={
                "model": "openai/gpt-4o-mini", 
                "messages": messages,
                "temperature": 0.3
            },
            timeout=timeout
        )
        if resp.status_code == 200:
            return resp.json()['choices'][0]['message']['content']
        else:
            print(f"[!] OpenRouter error {resp.status_code}: {resp.text}")
    except Exception as e:
        print(f"[!] OpenRouter exception: {e}")
    return None

def get_disease_advice(disease):
    """Get structured AI advice for a detected disease using OpenRouter."""
    prompt = (
        f"You are a highly qualified aquaculture veterinarian. A fish is diagnosed with: **{disease}**.\n\n"
        f"Provide a clear, structured response with these exact sections:\n"
        f"1. Overview (1-2 sentences)\n"
        f"2. Symptoms\n"
        f"3. Causes\n"
        f"4. Recommended Treatment\n"
        f"5. Prevention Tips\n"
        f"6. Feeding Recommendations during recovery\n\n"
        f"Be concise, scientifically accurate, and practical for fish farmers."
    )
    result = call_openrouter([{"role": "user", "content": prompt}])
    return result if result else "The AI Assistant is currently unavailable. Please check your API key or connection."

def answer_chat(question, disease, language='english'):
    """Answer a user question in the context of the detected disease."""
    lang_instruction = (
        f"You MUST respond ONLY in natural, professional {language}. "
        f"Do NOT mix English words unless they are technical aquaculture terms. "
        f"Ensure the grammar is perfect and the tone is helpful for a fish farmer."
        if language.lower() != 'english'
        else "Respond in clear, professional English."
    )
    messages = [
        {
            "role": "system",
            "content": (
                f"You are a world-class aquaculture veterinarian assistant. "
                f"The fish is diagnosed with **{disease}**. "
                f"{lang_instruction} Provide practical, scientifically accurate advice. "
                f"Format your response clearly and concisely."
            )
        },
        {"role": "user", "content": question}
    ]
    result = call_openrouter(messages)
    return result if result else "The AI Assistant is currently unavailable."

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files.get('image')
        if not file or file.filename == "":
            return jsonify({"error": "No file uploaded"}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        img       = load_img(filepath, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        if model is None:
            return jsonify({"error": "Model unavailable. Please restart the server."}), 500

        try:
            old_err = sys.stderr
            sys.stderr = NullWriter()
            prediction = model.predict(img_array, verbose=0)[0]
            sys.stderr = old_err
        except Exception as e:
            sys.stderr = old_err
            return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

        predicted_idx   = int(np.argmax(prediction))
        predicted_class = class_names[predicted_idx]
        confidence      = float(np.max(prediction))

        top3_idx = np.argsort(prediction)[-3:][::-1]
        top3 = [
            {"label": class_names[i], "confidence": round(float(prediction[i]) * 100, 1)}
            for i in top3_idx
        ]

        advice = get_disease_advice(predicted_class)

        return jsonify({
            "prediction": predicted_class,
            "confidence": round(confidence * 100, 1),
            "img_path":   url_for('static', filename=f'uploads/{filename}'),
            "advice":     advice,
            "top3":       top3,
            "disease_info": DISEASE_INFO.get(predicted_class, {})
        })
    except Exception as e:
        print(f"[!] Predict error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data     = request.json
        question = data.get('message', '').strip()
        disease  = data.get('disease', 'Unknown')
        language = data.get('language', 'english')

        if not question:
            return jsonify({"error": "Empty message"}), 400

        response = answer_chat(question, disease, language)
        return jsonify({"response": response})
    except Exception as e:
        print(f"[!] Chat error: {e}")
        return jsonify({"error": str(e)}), 500

# ---------------------------------------------------------------------------
# PDF Report — fixed and enhanced
# ---------------------------------------------------------------------------
@app.route('/download-pdf', methods=['POST'])
def download_pdf():
    try:
        data       = request.json or {}
        disease    = data.get('disease', 'Unknown')
        advice     = data.get('advice', '')
        confidence = data.get('confidence', 0)
        top3       = data.get('top3', [])
        language   = data.get('language', 'english').lower()
        info       = DISEASE_INFO.get(disease, {})
        labels     = LANG_LABELS.get(language, LANG_LABELS["english"])

        buffer = BytesIO()
        doc    = SimpleDocTemplate(
            buffer,
            rightMargin=inch * 0.75, leftMargin=inch * 0.75,
            topMargin=inch * 0.75,   bottomMargin=inch * 0.75
        )

        # Font selection - Using unique names per language
        font_name = "Helvetica" # Default fallback
        
        try:
            if "telugu" in language and os.path.exists(FONT_PATHS["telugu"]):
                font_name = "TeluguFont"
                pdfmetrics.registerFont(TTFont(font_name, FONT_PATHS["telugu"]))
            elif "hindi" in language and os.path.exists(FONT_PATHS["hindi"]):
                font_name = "HindiFont"
                pdfmetrics.registerFont(TTFont(font_name, FONT_PATHS["hindi"]))
            elif os.path.exists(FONT_PATHS["default"]):
                font_name = "DefaultFont"
                pdfmetrics.registerFont(TTFont(font_name, FONT_PATHS["default"]))
        except Exception as fe:
            print(f"[!] Font registration error: {fe}")
            font_name = "Helvetica"

        # Styles
        title_style = ParagraphStyle(
            'Title', fontName=font_name, fontSize=18, textColor=colors.HexColor('#1a56db'),
            spaceAfter=6, alignment=1
        )
        subtitle_style = ParagraphStyle(
            'Subtitle', fontName=font_name, fontSize=10, textColor=colors.grey,
            spaceAfter=14, alignment=1
        )
        section_style = ParagraphStyle(
            'Section', fontName=font_name, fontSize=14, textColor=colors.HexColor('#1e3a5f'),
            spaceBefore=14, spaceAfter=6, fontWeight='bold'
        )
        body_style = ParagraphStyle(
            'Body', fontName=font_name, fontSize=11, textColor=colors.HexColor('#333333'),
            spaceAfter=8, leading=18  # Increased leading for Hindi/Telugu matras
        )
        highlight_style = ParagraphStyle(
            'Highlight', fontName=font_name, fontSize=11, textColor=colors.HexColor('#155e75'),
            spaceAfter=6, backColor=colors.HexColor('#e0f2fe'), leftIndent=8, rightIndent=8,
            leading=16
        )
        warning_style = ParagraphStyle(
            'Warning', fontName=font_name, fontSize=11, textColor=colors.HexColor('#7c2d12'),
            spaceAfter=6, backColor=colors.HexColor('#fef3c7'), leftIndent=8, rightIndent=8,
            leading=16
        )

        story = []

        # Header
        story.append(Paragraph(labels["title"], title_style))
        story.append(Paragraph(labels["subtitle"], subtitle_style))
        story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#1a56db')))
        story.append(Spacer(1, 12))

        # Detection Result
        story.append(Paragraph(labels["detection_result"], section_style))
        is_healthy = disease == 'Healthy Fish'
        result_style = highlight_style if is_healthy else warning_style
        story.append(Paragraph(f"{labels['diagnosis']}: {disease}", result_style))
        story.append(Paragraph(f"{labels['confidence']}: {confidence}%", result_style))
        story.append(Spacer(1, 8))

        # Top 3 predictions
        if top3:
            story.append(Paragraph(labels["top3"], section_style))
            for i, item in enumerate(top3, 1):
                story.append(Paragraph(
                    f"{i}. {item['label']} — {item['confidence']}%", body_style
                ))
            story.append(Spacer(1, 8))

        # Disease Information
        if info:
            story.append(HRFlowable(width="100%", thickness=1, color=colors.lightgrey))
            story.append(Paragraph(labels["disease_info"], section_style))
            if info.get('symptoms'):
                story.append(Paragraph(f"<b>{labels['symptoms']}:</b> {info['symptoms']}", body_style))
            if info.get('causes'):
                story.append(Paragraph(f"<b>{labels['causes']}:</b> {info['causes']}", body_style))
            if info.get('treatment'):
                story.append(Paragraph(f"<b>{labels['treatment']}:</b> {info['treatment']}", body_style))
            if info.get('prevention'):
                story.append(Paragraph(f"<b>{labels['prevention']}:</b> {info['prevention']}", body_style))
            story.append(Spacer(1, 8))

        # AI Assistant Matter (The latest advice/chat response)
        if advice:
            story.append(HRFlowable(width="100%", thickness=1, color=colors.lightgrey))
            story.append(Paragraph(labels["ai_advice"], section_style))
            
            # Use regex for better markdown handling
            import re
            from xml.sax.saxutils import escape

            for line in advice.split('\n'):
                original_line = line.strip()
                if not original_line:
                    story.append(Spacer(1, 6))
                    continue

                # 1. Escape XML special characters (&, <, >)
                clean = escape(original_line)
                
                # 2. Bold: **text** -> <b>text</b>
                clean = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', clean)
                
                # 3. Italic: *text* -> <i>text</i> (excluding bullet points)
                # Matches *text* if not preceded by start of line (bullet)
                clean = re.sub(r'(?<!^)\*(.*?)\*', r'<i>\1</i>', clean)
                
                # 4. Bullet points: * or - at start
                if original_line.startswith('* ') or original_line.startswith('- '):
                    # We use original_line[2:] but still need to escape it
                    clean = f"&bull; {escape(original_line[2:])}"
                    # Re-apply bold/italic to the escaped bullet content
                    clean = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', clean)
                    clean = re.sub(r'(?<!^)\*(.*?)\*', r'<i>\1</i>', clean)
                
                try:
                    story.append(Paragraph(clean, body_style))
                except Exception as pe:
                    print(f"[!] Paragraph error: {pe} | Line: {clean}")
                    # Absolute safe fallback: escape everything and remove tags
                    safe_line = escape(original_line).replace('<b>', '').replace('</b>', '').replace('<i>', '').replace('</i>', '')
                    story.append(Paragraph(safe_line, body_style))
            story.append(Spacer(1, 12))

        # Footer
        story.append(Spacer(1, 16))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.lightgrey))
        story.append(Paragraph(
            "Generated by Fish Disease Detection System | For aquaculture use only. "
            "Always consult a licensed veterinarian for treatment decisions.",
            ParagraphStyle('Footer', fontName=font_name, fontSize=8,
                           textColor=colors.grey, alignment=1, spaceBefore=4)
        ))

        doc.build(story)
        buffer.seek(0)
        return send_file(
            buffer,
            as_attachment=True,
            download_name=f"Fish_Report_{disease.replace(' ','_')}.pdf",
            mimetype="application/pdf"
        )
    except Exception as e:
        print(f"[!] PDF error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("[*] Starting Flask app...")
    app.run(debug=True, use_reloader=False)
