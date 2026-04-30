# Fish Disease Detection and Advisory System

An AI-powered aquaculture health analysis system that detects fish diseases using deep learning with EfficientNet. The system provides multilingual support (English, Hindi, Telugu) and generates comprehensive PDF reports with AI-powered diagnosis and treatment recommendations using OpenRouter API.

**Live Demo**: Available at `http://localhost:5000` after setup

---

## ✨ Features

- 🎯 **AI Disease Detection**: EfficientNet-based model identifies fish diseases with 92%+ accuracy
- 🤖 **AI-Powered Analysis**: Integrates with OpenRouter API for intelligent diagnosis and treatment
- 📄 **Multilingual PDF Reports**: Professional reports in English, Hindi, and Telugu
- 🌐 **Web Interface**: User-friendly Flask web interface for image uploads
- 🔍 **Confidence Scores**: Displays top 3 disease predictions with confidence levels
- 📊 **Comprehensive Analysis**: Symptoms, causes, treatment, and prevention recommendations

---

## 🐟 Supported Fish Diseases

| Disease Type | Categories |
|---|---|
| **Bacterial** | Aeromoniasis, Gill Disease, Red Disease |
| **Fungal** | Saprolegniasis |
| **Parasitic** | Various parasitic infections |
| **Viral** | White Tail Disease |
| **Healthy** | Normal fish classification |

---

## 🎬 How It Works

```
┌─────────────────────────────────────────────────────────────┐
│ 1. User uploads fish image via web interface               │
├─────────────────────────────────────────────────────────────┤
│ 2. Image preprocessing (resize, normalize)                 │
├─────────────────────────────────────────────────────────────┤
│ 3. EfficientNet model predicts disease category            │
├─────────────────────────────────────────────────────────────┤
│ 4. Prediction sent to OpenRouter API                       │
├─────────────────────────────────────────────────────────────┤
│ 5. AI generates detailed diagnosis & treatment             │
├─────────────────────────────────────────────────────────────┤
│ 6. Professional PDF report generated in chosen language    │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Model Performance

- **Architecture**: EfficientNetB0
- **Accuracy**: 92% on validation dataset
- **Loss**: 0.15
- **Training Dataset**: Freshwater Fish Disease Aquaculture
- **Input Size**: 224x224 pixels
- **Classes**: 7 disease categories

---

## 📂 Dataset

- **Source**: Freshwater Fish Disease Aquaculture in South Asia
- **Classes**: 7 (Aeromoniasis, Gill Disease, Red Disease, Saprolegniasis, Healthy, Parasitic, White Tail Disease)
- **Dataset Size**: ~1000+ images per category
- **Preprocessing**: Resizing, Normalization, Data Augmentation
- **Split**: 80% Training, 20% Testing

---

## 🛠️ Requirements

- Python 3.10 or higher
- TensorFlow/Keras 2.13+
- Flask 3.0+
- See `requirements.txt` for full list

---

## 📥 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/vuduthuri12/Fish-Disease-Detection-and-Advisory-System.git
cd Fish-Disease-Detection-and-Advisory-System
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv310
venv310\Scripts\activate

# macOS/Linux
python -m venv venv310
source venv310/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Model
The trained model needs to be downloaded separately:

- **Model**: Download from [Google Drive](https://drive.google.com/file/d/1S_oyhqNGb7i5PNzhNZC3Up5o10ddY-z0/view?usp=drive_link) (add your link)

Extract and place `final_fish_model.keras` in the project root directory.

### 5. Configure API Keys
Create a `.env` file in the root directory:
```
OPENROUTER_API_KEY=your_api_key_here
```

Get your free API key from [OpenRouter](https://openrouter.ai)

---

## 🚀 Usage

### Start the Application
```bash
python app.py
```

### Access Web Interface
1. Open browser and navigate to: `http://localhost:5000`
2. Upload a fish image (JPG, PNG, JPEG)
3. Select output language (English, Hindi, or Telugu)
4. Click "Analyze"
5. Download generated PDF report

### Example Image Requirements
- Format: JPG, PNG, JPEG
- Size: 224x224px or larger (auto-resized)
- Content: Clear image of fish

---

## 📸 Screenshots

## Screenshots

### Home Page
![Home Page](https://github.com/user-attachments/assets/e47d407e-7383-49c9-8211-6fc031fc92a6)
*Web interface for uploading fish images*

### Detection Result
![Result Page](https://github.com/user-attachments/assets/68ce1393-ccf9-48ae-9f26-1c5841acb48a)
*Disease prediction with confidence scores*

### PDF Report
![PDF Report](https://github.com/user-attachments/assets/fd8f5469-ebe0-445a-af60-d6e52b0aa132)
*Multilingual diagnosis report in PDF format*

---

## 📁 Project Structure

```
Fish-Disease-Detection-and-Advisory-System/
├── app.py                          # Main Flask application
├── final_fish_model.keras          # Trained EfficientNet model
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variables template
├── .gitignore                      # Git ignore file
├── README.md                       # This file
│
├── static/
│   ├── style.css                   # Web interface styling
│   └── uploads/                    # Temporary uploaded images
│
├── templates/
│   └── index.html                  # Web interface HTML
│
├── fonts/                          # Multilingual fonts (excluded from git)
│   ├── NotoSans-Regular.ttf
│   ├── NotoSansDevanagari-Regular.ttf
│   └── NotoSansTelugu-Regular.ttf
│
├── dev_assets/                     # Development utilities (excluded)
│   ├── inspect_model.py
│   ├── test_api.py
│   └── Freshwater Fish Disease Dataset/
│
└── venv310/                        # Virtual environment (excluded)
```

---

## 🔌 API Integration

This project uses **OpenRouter API** for AI-powered disease analysis:

- **Endpoint**: `https://openrouter.ai/api/v1/chat/completions`
- **Model**: Supports any LLM available on OpenRouter
- **Purpose**: Generate detailed diagnosis and treatment recommendations
- **Cost**: Free tier available for testing

### Get API Key:
1. Visit [OpenRouter](https://openrouter.ai)
2. Sign up for free account
3. Generate API key
4. Add to `.env` file as `OPENROUTER_API_KEY`

---

## 🏗️ Tech Stack

| Component | Technology |
|---|---|
| **Deep Learning** | TensorFlow/Keras, EfficientNet |
| **Web Framework** | Flask 3.1.3 |
| **Image Processing** | Pillow, NumPy |
| **API Integration** | OpenRouter, Requests |
| **Report Generation** | ReportLab |
| **Multilingual Support** | Noto Sans Fonts |
| **Environment** | Python 3.10 |

---

## 🔒 Security

- ✅ API keys stored in `.env` (never committed)
- ✅ `.env` file excluded from git
- ✅ Large files excluded from repository
- ✅ Input validation on file uploads
- ✅ Temporary uploads cleaned up

**Important**: Never commit `.env` file with your API keys!

---

## 🚨 Troubleshooting

### Model File Not Found
```
Solution: Download model and place in root directory
```

### API Key Error
```
Solution: Verify OPENROUTER_API_KEY is set in .env
```

### Port Already in Use
```bash
# Run on different port
python app.py --port 5001
```

### Font Issues
```bash
# Fonts are automatically downloaded if missing
# Or manually copy to fonts/ directory
```

---

## 📝 Model Architecture

```
Input (224x224x3)
    ↓
EfficientNetB0 (Pre-trained)
    ↓
Global Average Pooling
    ↓
Dense Layer (256 units, ReLU)
    ↓
Dropout (0.5)
    ↓
Output (7 classes, Softmax)
```

---

## 📈 Future Improvements

- [ ] Real-time camera feed analysis
- [ ] Model fine-tuning with more data
- [ ] Multiple model ensemble
- [ ] Mobile app development
- [ ] Database integration for tracking
- [ ] Multi-user authentication
- [ ] Batch processing capability
- [ ] Model versioning system

---

## 📄 License

This project is open source and available under the **MIT License**. See [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -m "Add: new feature"`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

## 👨‍💼 Author

**Developed for**: Aquaculture Health Monitoring and AI-Powered Disease Detection

**Contact**: [GitHub Profile](https://github.com/vuduthuri12)

---

## ⚠️ Important Notes

- **Model Download Required**: The trained model must be downloaded separately due to file size constraints
- **API Key Required**: OpenRouter API key needed for AI-powered analysis
- **Security**: Never commit `.env` file or API keys to version control
- **Production Deployment**: For production, implement proper error handling and monitoring

---

## 📞 Support

For issues, questions, or suggestions:
1. Open an [Issue](https://github.com/vuduthuri12/Fish-Disease-Detection-and-Advisory-System/issues)
2. Create a [Discussion](https://github.com/vuduthuri12/Fish-Disease-Detection-and-Advisory-System/discussions)
3. Check existing documentation

---

**Last Updated**: April 2026
**Status**: Active Development ✅
