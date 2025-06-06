# sentiment-analysis-python
Latency and accuracy analysis of deep learning vs traditional sentiment analysis models.

This project contains two parts the research part which involves benchmarking latency and accuracy of traditional machine learning against BERT based models and the second part is the BERT sentiment analysis Web application.

Part 1 - Latency and accuracy of traditional machine learning against BERT based models



Part 2 -BERT Sentiment Analysis Web Application
A Flask-based web application that performs sentiment analysis on text data using a pre-trained BERT model.
Overview
This application allows users to upload CSV or Excel files containing text data, select the column containing the text to analyze, and receive sentiment analysis results classified as positive or negative. The application provides visualizations including word clouds and sentiment distribution charts, and allows users to download the complete analysis results.
Features
Upload CSV or Excel files with text data
Test dataset -
Clean and preprocess text automatically
Analyze sentiment using a fine-tuned BERT model
Generate interactive visualizations:
Sentiment distribution pie chart
Word clouds for positive and negative reviews
View sample results in-browser
Download complete analysis results as CSV
Installation
Prerequisites
Python 3.8+
pip package manager
Setup
Clone the repository:
git clone https://csgitlab.reading.ac.uk/ch016067/finalyearproject-31016067.git
cd finalyearproject-31016067
Create a virtual environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies:
hpip install -r requirements.txt
Steps
Start the application:
python app.py
optional steps if errors occur:
In terminal
pip install tensorflow flask transformers pandas matplotlib wordcloud numpy tf-keras

conda activate tf_env
conda create -n tf_env python=3.10


Open your web browser and navigate to:
http://localhost:5000/ or the one given in the terminal
Follow the on-screen instructions:
Upload your dataset (CSV or Excel)
Select the column containing
text dataset - https://docs.google.com/spreadsheets/d/1KYfF9NLcxk_sZRSYyzCOyb7iUQm-8dqDUJiI-eWy0tc/edit?usp=sharing
View and download the analysis results
File Structure
sentiment-analysis-app/
├── app.py                # Main Flask application
├── requirements.txt      # Python dependencies
├── templates/            # HTML templates
│   ├── index.html        # Home page
│   ├── select_column.html  # Column selection page
│   └── results.html      # Results display page
├── models/               # Directory for the BERT model
│   └── best_bert_model/  # Pre-trained BERT model files
├── uploads/              # Temporary storage for uploaded files
└── results/              # Directory for generated CSV results
Technical Details
Backend: Flask web framework
ML Framework: TensorFlow and Hugging Face Transformers
Model: Fine-tuned BERT for sequence classification
Data Visualization: Matplotlib and WordCloud
Frontend: Bootstrap 5 for responsive design
Data Processing Pipeline
Upload: User uploads a CSV or Excel file
Text Selection: User selects the column containing text data
Preprocessing: Text is cleaned (URLs, mentions, punctuation removed)
Model Inference: BERT model classifies each text as positive (1) or negative (0)
Visualization: Generate charts based on analysis results
Results: Display sample results and offer full download
Customization
Model: Replace the BERT model in the models/best_bert_model directory with any compatible TFBertForSequenceClassification model
UI: Modify the HTML templates in the templates directory to change the user interface
Processing: Adjust text cleaning and preprocessing in the clean_text() function
Troubleshooting
Model Loading Error: Ensure you have the correct model files in the models/best_bert_model directory
Memory Issues: For large datasets, consider processing in batches or increasing the server memory
TensorFlow Warnings: The app disables oneDNN optimizations by default to avoid common warnings
Acknowledgments
This project uses the Hugging Face Transformers library
UI design powered by Bootstrap 5
