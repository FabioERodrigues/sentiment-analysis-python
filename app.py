# app.py
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations

import os
import re
import io
import base64
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify
from werkzeug.utils import secure_filename
from transformers import BertTokenizer, TFBertForSequenceClassification
from wordcloud import WordCloud

# Initialize Flask app
app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True  # Add this line
app.config['DEBUG'] = True
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MODEL_FOLDER'] = 'models'
app.config['RESULTS_FOLDER'] = 'results'
#app.config['MAX_CONTENT_LENGTH'] = 1000 * 1024 * 1024  # 16MB max upload

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Global variables to store state
current_data = None
bert_model = None
bert_tokenizer = None

@app.route('/test')
def test():
    return "Flask is running!"

def load_model():
    global bert_model, bert_tokenizer
    try:
        model_path = os.path.join(app.config['MODEL_FOLDER'], 'best_bert_model')
        bert_model = TFBertForSequenceClassification.from_pretrained(model_path)
        bert_tokenizer = BertTokenizer.from_pretrained(model_path)
        print("✅ Model loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return False
    

# Clean text - reusing the function from your code
def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)     # Remove mentions
    text = re.sub(r'#', '', text)        # Remove hashtags symbol
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()                  # Convert to lowercase
    return text

# Preprocess the uploaded dataset
def preprocess_data(df, text_column):
    # Make a copy to avoid modifying the original
    processed_df = df.copy()
    
    # Clean text data
    processed_df['cleaned_text'] = processed_df[text_column].apply(clean_text)
    
    # Drop rows with empty text after cleaning
    processed_df = processed_df[processed_df['cleaned_text'].str.len() > 5]
    
    # Reset index
    processed_df = processed_df.reset_index(drop=True)
    
    return processed_df

# Run sentiment analysis using the BERT model
def predict_sentiment(df):
    # Encode the texts
    encodings = bert_tokenizer(
        df['cleaned_text'].tolist(),
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='tf'
    )
    
    # Create TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices((
        {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask']
        }
    )).batch(32)
    
    # Make predictions
    predictions = []
    for batch in dataset:
        outputs = bert_model(batch)
        logits = outputs.logits.numpy()
        preds = np.argmax(logits, axis=1)
        predictions.extend(preds)
    
    # Add predictions to DataFrame
    df['sentiment'] = predictions
    df['sentiment_label'] = df['sentiment'].map({1: 'POSITIVE', 0: 'NEGATIVE'})
    
    return df

# Generate word clouds for positive and negative reviews
def generate_wordclouds(df):
    # Concatenate reviews by sentiment
    negative_reviews = ' '.join(df[df['sentiment'] == 0]['cleaned_text'])
    positive_reviews = ' '.join(df[df['sentiment'] == 1]['cleaned_text'])
    
    # Generate word clouds
    wordcloud_neg = WordCloud(width=800, height=400, background_color='black').generate(negative_reviews)
    wordcloud_pos = WordCloud(width=800, height=400, background_color='black').generate(positive_reviews)
    
    # Create plots and convert to base64 for HTML embedding
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    axes[0].imshow(wordcloud_neg, interpolation='bilinear')
    axes[0].set_title('Negative Reviews')
    axes[0].axis('off')
    
    axes[1].imshow(wordcloud_pos, interpolation='bilinear')
    axes[1].set_title('Positive Reviews')
    axes[1].axis('off')
    
    # Convert plot to base64
    buffer = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    wordcloud_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    
    return wordcloud_image

def generate_sentiment_chart(df):
    # Count sentiments
    sentiment_counts = df['sentiment_label'].value_counts()
    
    # Create color mapping
    color_map = {
            'POSITIVE': '#90ee90',  # lightgreen
            'NEGATIVE': '#f08080'   # lightcoral
        }
    
    # Get colors in correct order based on sentiment_counts index
    colors = [color_map[label] for label in sentiment_counts.index]
    
    # Create pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(
        sentiment_counts,
        labels=sentiment_counts.index,
        autopct='%1.1f%%',
        colors=colors,
        startangle=140
    )
    plt.title('Sentiment Distribution')
    plt.axis('equal')
    
    # Convert to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    pie_chart = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    
    return pie_chart

@app.route('/')
def index():
    global current_data, bert_model, bert_tokenizer
    current_data = None
    
    # Force reload model if missing
    if bert_model is None or bert_tokenizer is None:
        if not load_model():
            return render_template('index.html', model_loaded=False)
    
    return render_template('index.html', model_loaded=True)


@app.route('/upload', methods=['POST'])
def upload_file():
    global current_data
    
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        try:
            # Process the uploaded file
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Determine file type and read accordingly (support CSV and Excel)
            if filename.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif filename.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(file_path)
            else:
                return "Unsupported file format. Please upload a CSV or Excel file."
            
            # Store the dataframe for later use
            current_data = {
                'filename': filename,
                'df': df,
                'columns': df.columns.tolist()
            }
            
            return render_template('select_column.html', 
                                  filename=filename,
                                  columns=df.columns.tolist())
        
        except Exception as e:
            return f"Error processing file: {str(e)}"

@app.route('/process', methods=['POST'])
def process_data():
    global current_data
    
    if current_data is None:
        return redirect(url_for('index'))
    
    text_column = request.form.get('text_column')
    if text_column not in current_data['df'].columns:
        return "Selected column not found in dataset."
    
    try:
        # Preprocess data
        processed_df = preprocess_data(current_data['df'], text_column)
        
        # Run sentiment analysis
        result_df = predict_sentiment(processed_df)
        
        # Generate visualizations
        wordcloud_image = generate_wordclouds(result_df)
        pie_chart = generate_sentiment_chart(result_df)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_filename = f"sentiment_analysis_{timestamp}.csv"
        result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
        result_df.to_csv(result_path, index=False)
        
        # Update current data
        current_data['processed_df'] = result_df
        current_data['result_filename'] = result_filename
        
        # Sample data to display (first 10 rows)
        sample_data = result_df.head(10).to_dict('records')
        
        # Calculate some stats
        total_reviews = len(result_df)
        positive_reviews = (result_df['sentiment'] == 1).sum()
        negative_reviews = (result_df['sentiment'] == 0).sum()
        positive_percent = (positive_reviews / total_reviews) * 100
        negative_percent = (negative_reviews / total_reviews) * 100
        
        return render_template('results.html',
                              sample_data=sample_data,
                              wordcloud_image=wordcloud_image,
                              pie_chart=pie_chart,
                              total_reviews=total_reviews,
                              positive_reviews=positive_reviews,
                              negative_reviews=negative_reviews,
                              positive_percent=round(positive_percent, 2),
                              negative_percent=round(negative_percent, 2),
                              result_filename=result_filename)
    
    except Exception as e:
        return f"Error during processing: {str(e)}"

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(
        os.path.join(app.config['RESULTS_FOLDER'], filename),
        mimetype='text/csv',
        download_name=filename,  # Changed parameter name
        as_attachment=True
    )

if __name__ == '__main__':
    # Only load model once (not in reloader)
    if os.environ.get('WERKZEUG_RUN_MAIN') != 'true':
        load_model()
    app.run(debug=True, host='0.0.0.0', port=5000)