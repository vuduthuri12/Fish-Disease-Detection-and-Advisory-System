# Fish Disease Detection System

An AI-powered aquaculture health analysis system that detects fish diseases using deep learning. The system provides multilingual support (English, Hindi, Telugu) and generates comprehensive PDF reports with AI-powered diagnosis and treatment recommendations.

## Features

- 🎯 **Disease Detection**: Uses a trained EfficientNet model to identify fish diseases
- 🤖 **AI-Powered Analysis**: Integrates with OpenRouter API for detailed disease analysis and treatment recommendations
- 📄 **PDF Reports**: Generates professional multilingual reports with diagnosis, symptoms, causes, and prevention methods
- 🌐 **Multilingual Support**: English, Hindi, and Telugu language support
- 🖼️ **Web Interface**: Simple Flask-based web interface for easy image uploads

## Supported Fish Diseases

- Bacterial diseases (Aeromoniasis, Gill disease, Red disease)
- Fungal diseases (Saprolegniasis)
- Parasitic diseases
- Viral diseases (White tail disease)
- Healthy Fish classification

## Requirements

- Python 3.10+
- TensorFlow/Keras
- Flask
- Required packages listed in `requirements.txt`

## Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd iomp
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv310
   source venv310/bin/activate  # On Windows: venv310\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   Create a `.env` file in the root directory:
   ```
   OPENROUTER_API_KEY=your_api_key_here
   ```

## Usage

1. **Run the Flask application**:
   ```bash
   python app.py
   ```

2. **Open in browser**:
   Navigate to `http://localhost:5000`

3. **Upload an image**:
   - Select a fish image
   - Choose the output language (English, Hindi, or Telugu)
   - The system will analyze the image and generate a detailed report

## Project Structure

```
iomp/
├── app.py                          # Main Flask application
├── final_fish_model.keras          # Trained deep learning model
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── static/
│   ├── style.css                   # Web interface styling
│   └── uploads/                    # Uploaded images (temporary)
├── templates/
│   └── index.html                  # Web interface template
├── fonts/                          # Multilingual font files
├── dev_assets/                     # Development utilities and datasets
└── venv310/                        # Virtual environment (not pushed to git)
```

## API Integration

This project uses **OpenRouter API** for AI-powered disease analysis. Get your API key from [OpenRouter](https://openrouter.ai) and add it to your `.env` file.

## Model

The disease detection model is based on **EfficientNet** architecture, trained on the Freshwater Fish Disease Aquaculture dataset. Model file: `final_fish_model.keras`

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

**Note**: Make sure to never commit the `.env` file containing your API keys to the repository.
