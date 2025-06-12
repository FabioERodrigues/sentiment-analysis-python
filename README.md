This project contains of two parts:
Part 1 - Comparing Latency and accuracy of traditional and deep learning 
This is contained in 


Part 2 - BERT Sentiment Analysis Web Application

A Flask-based web application that performs sentiment analysis on text data using a pre-trained BERT model.

## Overview

This application allows users to upload CSV or Excel files containing text data, select the column containing the text to analyze, and receive sentiment analysis results classified as positive or negative. The application provides visualizations including word clouds and sentiment distribution charts, and allows users to download the complete analysis results.

## Features

- 📁 **File Upload**: Upload CSV or Excel files with text data
- 🧪 **Test Dataset**: Pre-loaded dataset for testing functionality
- 🧹 **Text Preprocessing**: Clean and preprocess text automatically
- 🤖 **BERT Analysis**: Analyze sentiment using a fine-tuned BERT model
- 📊 **Interactive Visualizations**:
  - Sentiment distribution pie chart
  - Word clouds for positive and negative reviews
- 👀 **In-Browser Results**: View sample results directly in the application
- 💾 **Export Results**: Download complete analysis results as CSV

## Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://csgitlab.reading.ac.uk/ch016067/finalyearproject-31016067.git
   cd finalyearproject-31016067
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Starting the Application

1. **Start the Flask server**:
   ```bash
   python app.py
   ```

2. **Open your web browser** and navigate to:
   ```
   http://localhost:5000/
   ```
   (or use the URL provided in the terminal)

### Using the Application

1. Upload your dataset (CSV or Excel format)
2. Select the column containing text data
3. Wait for the analysis to complete
4. View visualizations and sample results
5. Download the complete analysis results

### Test Dataset

A sample dataset is available for testing:
[Google Sheets Dataset](https://docs.google.com/spreadsheets/d/1KYfF9NLcxk_sZRSYyzCOyb7iUQm-8dqDUJiI-eWy0tc/edit?usp=sharing)

## Optional Installation Steps

If you encounter errors, try these additional installation commands:

```bash
pip install tensorflow flask transformers pandas matplotlib wordcloud numpy tf-keras
```

Or create a conda environment:
```bash
conda create -n tf_env python=3.10
conda activate tf_env
```

## File Structure

```
sentiment-analysis-app/
├── app.py                    # Main Flask application
├── requirements.txt          # Python dependencies
├── templates/               # HTML templates
│   ├── index.html          # Home page
│   ├── select_column.html  # Column selection page
│   └── results.html        # Results display page
├── models/                 # Directory for the BERT model
│   └── best_bert_model/   # Pre-trained BERT model files
├── uploads/               # Temporary storage for uploaded files
└── results/              # Directory for generated CSV results
```

## Technical Details

- **Backend**: Flask web framework
- **ML Framework**: TensorFlow and Hugging Face Transformers
- **Model**: Fine-tuned BERT for sequence classification
- **Data Visualization**: Matplotlib and WordCloud
- **Frontend**: Bootstrap 5 for responsive design

## Data Processing Pipeline

1. **Upload**: User uploads a CSV or Excel file
2. **Text Selection**: User selects the column containing text data
3. **Preprocessing**: Text is cleaned (URLs, mentions, punctuation removed)
4. **Model Inference**: BERT model classifies each text as positive (1) or negative (0)
5. **Visualization**: Generate charts based on analysis results
6. **Results**: Display sample results and offer full download

## Customization

### Model Replacement
Replace the BERT model in the `models/best_bert_model` directory with any compatible `TFBertForSequenceClassification` model.

### UI Modifications
Modify the HTML templates in the `templates` directory to customize the user interface.

### Text Processing
Adjust text cleaning and preprocessing in the `clean_text()` function within `app.py`.

## Troubleshooting

### Common Issues

**Model Loading Error**
- Ensure you have the correct model files in the `models/best_bert_model` directory

**Memory Issues**
- For large datasets, consider processing in batches or increasing server memory

**TensorFlow Warnings**
- The app disables oneDNN optimizations by default to avoid common warnings

### Getting Help

If you encounter issues:
1. Check that all dependencies are properly installed
2. Verify your Python version (3.8+ required)
3. Ensure the BERT model files are in the correct directory
4. Try running with a smaller dataset first

## Acknowledgments

- This project uses the [Hugging Face Transformers](https://huggingface.co/transformers/) library
- UI design powered by [Bootstrap 5](https://getbootstrap.com/)
- Built with [Flask](https://flask.palletsprojects.com/) web framework

## License

This project is part of a final year project submission.
