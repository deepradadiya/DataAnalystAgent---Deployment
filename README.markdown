# Data Analyst Agent

A Streamlit-based intelligent data analyst agent powered by Together.ai. Upload CSV, Excel, PDF, Word, text, or image files to analyze data, generate visualizations, and ask questions.

## Features
- Supports multiple file types (CSV, Excel, PDF, Word, Text, Images)
- Data analysis and visualization
- Interactive Q&A with AI
- Comprehensive report generation

## Deployment on Render
1. **Create a GitHub Repository**:
   - Push `app.py`, `requirements.txt`, `Dockerfile`, `.dockerignore`, and `README.md` to a GitHub repository.

2. **Set Up Render**:
   - Sign up at [render.com](https://render.com/) and create a new Web Service.
   - Connect your GitHub repository.
   - Configure:
     - **Environment**: Docker
     - **Environment Variable**: `TOGETHER_API_KEY` = Your Together.ai API key
   - Deploy the app.

3. **Test the App**:
   - Visit the app URL (e.g., `https://your-app.onrender.com`).
   - Upload a small file (<50MB) to test functionality.

## Requirements
- Python dependencies: See `requirements.txt`
- System dependencies: `tesseract-ocr`, `libtesseract-dev` (installed via `Dockerfile`)

## Notes
- Verify the Together.ai model name in `app.py` (`meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8`).
- The app limits file uploads to 50MB to stay within Render’s free tier constraints.