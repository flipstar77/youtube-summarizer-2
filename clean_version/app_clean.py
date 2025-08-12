#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YouTube Summarizer - Clean Application
Core Flask application with essential features only
"""

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file
from datetime import datetime
import os
import json
from dotenv import load_dotenv

# Core components
from enhanced_summarizer import EnhancedSummarizer
from database import DatabaseManager  
from chatbot_qa import VideoQAChatbot
from video_highlight_extractor import VideoHighlightExtractor
from srt_chaptering import SRTChapteringSystem

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-change-this')
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Initialize core components
try:
    db = DatabaseManager()
    summarizer = EnhancedSummarizer()
    chatbot = VideoQAChatbot()
    highlight_extractor = VideoHighlightExtractor()
    srt_system = SRTChapteringSystem()
    print("[OK] Core system initialized")
except Exception as e:
    print(f"[ERROR] System initialization failed: {e}")
    exit(1)

# ============================================================================
# Core Routes
# ============================================================================

@app.route('/')
def index():
    """Dashboard with recent summaries"""
    try:
        summaries = db.get_summaries() or []
        return render_template('index.html', summaries=summaries)
    except Exception as e:
        print(f"[ERROR] Dashboard error: {e}")
        return render_template('index.html', summaries=[])

@app.route('/new')
def new_summary():
    """New summary form"""
    return render_template('new_summary.html')

@app.route('/summarize', methods=['POST'])
def summarize_video():
    """Process video summarization"""
    try:
        url = request.form.get('url', '').strip()
        if not url:
            return jsonify({'status': 'error', 'message': 'URL is required'})
        
        # Get form parameters
        ai_provider = request.form.get('ai_provider', 'openai')
        model_name = request.form.get('model', 'gpt-4o-mini')
        custom_prompt = request.form.get('custom_prompt', '')
        
        print(f"[INFO] Processing video: {url}")
        print(f"[INFO] Using {ai_provider} model: {model_name}")
        
        # Process the video
        result = summarizer.process_video(
            video_url=url,
            ai_provider=ai_provider,
            model_name=model_name,
            custom_prompt=custom_prompt
        )
        
        if result.get('status') == 'success':
            # Save to database
            summary_id = db.save_summary(result)
            result['summary_id'] = summary_id
            
        return jsonify(result)
        
    except Exception as e:
        print(f"[ERROR] Summarization failed: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/summary/<int:summary_id>')
def view_summary(summary_id):
    """View individual summary"""
    try:
        summary = db.get_summary(summary_id)
        if not summary:
            return redirect(url_for('index'))
        return render_template('view_summary.html', summary=summary)
    except Exception as e:
        print(f"[ERROR] View summary error: {e}")
        return redirect(url_for('index'))

# ============================================================================
# Q&A Chatbot Routes
# ============================================================================

@app.route('/chatbot')
def chatbot_interface():
    """Q&A chatbot interface"""
    try:
        summaries = db.get_summaries() or []
        return render_template('chatbot.html', summaries=summaries)
    except Exception as e:
        print(f"[ERROR] Chatbot interface error: {e}")
        return render_template('chatbot.html', summaries=[])

@app.route('/api/ask_question', methods=['POST'])
def ask_question():
    """Process Q&A questions"""
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({'status': 'error', 'message': 'Question is required'})
        
        # Get answer from chatbot
        result = chatbot.ask_question(question)
        return jsonify(result)
        
    except Exception as e:
        print(f"[ERROR] Q&A processing failed: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

# ============================================================================
# Video Highlights Routes  
# ============================================================================

@app.route('/highlights')
def highlights_interface():
    """Video highlights interface"""
    return render_template('highlights.html')

@app.route('/api/summaries')
def get_summaries_api():
    """Get summaries for highlights selection"""
    try:
        summaries = db.get_summaries() or []
        return jsonify({'summaries': summaries})
    except Exception as e:
        return jsonify({'summaries': []})

@app.route('/extract_highlights/<int:video_id>', methods=['POST'])
def extract_highlights(video_id):
    """Extract video highlights"""
    try:
        print(f"[INFO] Extracting highlights for video ID: {video_id}")
        
        # Get video data
        video = db.get_summary(video_id)
        if not video:
            return jsonify({'status': 'error', 'message': 'Video not found'})
        
        # Extract highlights
        result = highlight_extractor.extract_highlights_from_video(
            video_id=str(video_id),
            video_url=video.get('url', ''),
            video_title=video.get('title', 'Unknown')
        )
        
        return jsonify(result)
        
    except Exception as e:
        print(f"[ERROR] Highlight extraction failed: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/download_highlight/<filename>')
def download_highlight(filename):
    """Download highlight clip"""
    try:
        file_path = os.path.join('highlights', filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
        else:
            return "File not found", 404
    except Exception as e:
        return f"Download failed: {str(e)}", 500

# ============================================================================
# SRT Chaptering Routes
# ============================================================================

@app.route('/srt-chaptering')
def srt_chaptering():
    """SRT Upload and Chaptering page"""
    return render_template('srt_chaptering.html')

@app.route('/process_srt', methods=['POST'])
def process_srt():
    """Process uploaded SRT file and create chapters"""
    try:
        if 'srt_file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No SRT file uploaded'})
        
        file = request.files['srt_file']
        if file.filename == '' or not file.filename.lower().endswith('.srt'):
            return jsonify({'status': 'error', 'message': 'Please upload a valid SRT file'})
        
        # Read file content
        srt_content = file.read().decode('utf-8', errors='ignore')
        
        # Get parameters
        video_title = request.form.get('video_title', file.filename)
        chapter_count = int(request.form.get('chapter_count', 8))
        min_duration = int(request.form.get('min_duration', 60))
        
        print(f"[INFO] Processing SRT file: {file.filename}")
        
        # Process SRT
        result = srt_system.create_chapters_from_srt(
            srt_content=srt_content,
            video_title=video_title,
            chapter_count=chapter_count,
            min_chapter_duration=min_duration
        )
        
        if 'error' in result:
            return jsonify({'status': 'error', 'message': result['error']})
        
        return jsonify({
            'status': 'success',
            'message': 'SRT file processed successfully',
            'data': result
        })
        
    except Exception as e:
        print(f"[ERROR] SRT processing failed: {str(e)}")
        return jsonify({'status': 'error', 'message': f'Processing failed: {str(e)}'})

# ============================================================================
# API Routes
# ============================================================================

@app.route('/semantic_search', methods=['POST'])
def semantic_search():
    """Semantic search across summaries"""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({'status': 'error', 'message': 'Query is required'})
        
        # Use chatbot's search capability
        results = chatbot.search_summaries(query, limit=5)
        
        return jsonify({
            'status': 'success',
            'results': results,
            'query': query
        })
        
    except Exception as e:
        print(f"[ERROR] Search failed: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e), 'results': []})

# ============================================================================
# Error Handlers
# ============================================================================

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

# ============================================================================
# Application Startup
# ============================================================================

if __name__ == '__main__':
    # Check required environment variables
    required_vars = ['OPENAI_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {missing_vars}")
        print("Please create a .env file with your API keys")
        exit(1)
    
    # Create necessary directories
    os.makedirs('temp', exist_ok=True)
    os.makedirs('highlights', exist_ok=True)
    os.makedirs('srt_uploads', exist_ok=True)
    
    # Run application
    debug_mode = os.getenv('DEBUG', 'false').lower() == 'true'
    port = int(os.getenv('PORT', 5000))
    
    print(f"🚀 Starting YouTube Summarizer on http://localhost:{port}")
    app.run(debug=debug_mode, host='0.0.0.0', port=port)