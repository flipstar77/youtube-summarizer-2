#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YouTube Summarizer - Enhanced Application
Clean Flask application with all advanced features integrated
"""

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file
from flask_socketio import SocketIO
from datetime import datetime
import os
import json
import asyncio
from dotenv import load_dotenv

# Core components
from enhanced_summarizer import EnhancedSummarizer
from database_clean import DatabaseManager  
from chatbot_qa import VideoQAChatbot
from video_highlight_extractor import VideoHighlightExtractor
from srt_chaptering import SRTChapteringSystem

# Enhanced features
from cache_manager import cache
from websocket_manager import progress_tracker, register_socketio_events, ProgressContext
from batch_processor import BatchVideoProcessor, BatchJob
from pdf_exporter import PDFExporter
from content_categorizer import ContentCategorizer

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-change-this')
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Initialize SocketIO for real-time updates
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
register_socketio_events(socketio)

# Initialize core components
try:
    db = DatabaseManager()
    summarizer = EnhancedSummarizer()
    chatbot = VideoQAChatbot()
    highlight_extractor = VideoHighlightExtractor()
    srt_system = SRTChapteringSystem()
    
    # Enhanced components
    batch_processor = BatchVideoProcessor(summarizer, db)
    pdf_exporter = PDFExporter()
    categorizer = ContentCategorizer(summarizer.openai_client if hasattr(summarizer, 'openai_client') else None)
    
    print("[OK] Enhanced system initialized")
except Exception as e:
    print(f"[ERROR] System initialization failed: {e}")
    exit(1)

# ============================================================================
# Core Routes (Enhanced)
# ============================================================================

@app.route('/')
def index():
    """Enhanced dashboard with analytics"""
    try:
        summaries = db.get_summaries() or []
        
        # Get category statistics
        category_stats = categorizer.get_category_stats(summaries) if summaries else {}
        
        # Recent activity
        recent_summaries = summaries[:10] if summaries else []
        
        return render_template('index_enhanced.html', 
                             summaries=recent_summaries,
                             category_stats=category_stats,
                             total_videos=len(summaries))
    except Exception as e:
        print(f"[ERROR] Dashboard error: {e}")
        return render_template('index_enhanced.html', summaries=[], category_stats={}, total_videos=0)

@app.route('/new')
def new_summary():
    """Enhanced new summary form with batch options"""
    return render_template('new_summary_enhanced.html')

@app.route('/summarize', methods=['POST'])
def summarize_video():
    """Enhanced video summarization with progress tracking"""
    try:
        url = request.form.get('url', '').strip()
        if not url:
            return jsonify({'status': 'error', 'message': 'URL is required'})
        
        # Get form parameters
        ai_provider = request.form.get('ai_provider', 'openai')
        model_name = request.form.get('model', 'gpt-4o-mini')
        custom_prompt = request.form.get('custom_prompt', '')
        user_id = request.form.get('user_id', 'default')
        
        # Create progress tracking job
        with ProgressContext("video_processing", url, user_id) as progress:
            
            progress.update(10, "Extracting video metadata...")
            
            # Process the video
            result = summarizer.process_video(
                video_url=url,
                ai_provider=ai_provider,
                model_name=model_name,
                custom_prompt=custom_prompt
            )
            
            if result.get('status') == 'success':
                progress.update(70, "Categorizing content...")
                
                # Auto-categorize content
                categorization = categorizer.categorize_content(
                    result.get('title', ''),
                    result.get('summary', ''),
                    result.get('transcript', '')
                )
                
                # Add categorization to result
                result['categories'] = categorization.categories
                result['tags'] = categorization.tags
                result['topics'] = categorization.topics
                result['sentiment'] = categorization.sentiment
                result['difficulty_level'] = categorization.difficulty_level
                result['target_audience'] = categorization.target_audience
                
                progress.update(90, "Saving to database...")
                
                # Save to database
                summary_id = db.save_summary(result)
                result['summary_id'] = summary_id
                
                progress.update(100, "Processing completed successfully")
        
        return jsonify(result)
        
    except Exception as e:
        print(f"[ERROR] Summarization failed: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

# ============================================================================
# Batch Processing Routes
# ============================================================================

@app.route('/batch')
def batch_interface():
    """Batch processing interface"""
    templates = batch_processor.get_batch_templates()
    return render_template('batch_processing.html', templates=templates)

@app.route('/process_batch', methods=['POST'])
def process_batch():
    """Process batch of videos"""
    try:
        data = request.get_json()
        
        # Extract URLs from input
        urls_text = data.get('urls', '')
        urls = batch_processor.extract_urls_from_text(urls_text)
        
        if not urls:
            return jsonify({'status': 'error', 'message': 'No valid YouTube URLs found'})
        
        # Create batch job
        batch_job = BatchJob(
            urls=urls,
            ai_provider=data.get('ai_provider', 'openai'),
            model_name=data.get('model', 'gpt-4o-mini'),
            custom_prompt=data.get('custom_prompt', ''),
            max_parallel=int(data.get('max_parallel', 3)),
            user_id=data.get('user_id', 'default')
        )
        
        # Validate batch job
        validation = batch_processor.validate_batch_job(batch_job)
        if not validation['valid']:
            return jsonify({'status': 'error', 'message': ', '.join(validation['errors'])})
        
        # Process batch asynchronously
        import threading
        
        def process_async():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(batch_processor.process_batch(batch_job))
            loop.close()
        
        thread = threading.Thread(target=process_async)
        thread.start()
        
        return jsonify({
            'status': 'started',
            'message': f'Batch processing started for {len(urls)} videos',
            'validation': validation
        })
        
    except Exception as e:
        print(f"[ERROR] Batch processing failed: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

# ============================================================================
# Export Routes
# ============================================================================

@app.route('/export/<int:summary_id>/<format>')
def export_summary(summary_id: int, format: str):
    """Export summary in various formats"""
    try:
        summary = db.get_summary(summary_id)
        if not summary:
            return jsonify({'error': 'Summary not found'}), 404
        
        if format == 'pdf':
            output_path = pdf_exporter.export_single_summary(summary)
            return send_file(output_path, as_attachment=True)
        
        elif format == 'json':
            return jsonify(summary)
        
        elif format == 'txt':
            # Create text export
            text_content = f"""
Title: {summary.get('title', 'N/A')}
URL: {summary.get('url', 'N/A')}
Date: {summary.get('created_at', 'N/A')}

Summary:
{summary.get('summary', 'No summary available')}

Key Points:
{chr(10).join([f"• {point}" for point in summary.get('key_points', [])])}

Categories: {', '.join(summary.get('categories', []))}
Tags: {', '.join(summary.get('tags', []))}
            """
            
            import io
            text_file = io.BytesIO(text_content.encode('utf-8'))
            
            return send_file(
                text_file,
                as_attachment=True,
                download_name=f"summary_{summary_id}.txt",
                mimetype='text/plain'
            )
        
        else:
            return jsonify({'error': 'Unsupported format'}), 400
            
    except Exception as e:
        print(f"[ERROR] Export failed: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/export_batch', methods=['POST'])
def export_batch():
    """Export multiple summaries to PDF report"""
    try:
        data = request.get_json()
        summary_ids = data.get('summary_ids', [])
        
        if not summary_ids:
            return jsonify({'error': 'No summaries selected'}), 400
        
        # Get summaries
        summaries = []
        for summary_id in summary_ids:
            summary = db.get_summary(summary_id)
            if summary:
                summaries.append(summary)
        
        if not summaries:
            return jsonify({'error': 'No valid summaries found'}), 404
        
        # Generate batch PDF
        output_path = pdf_exporter.export_batch_summary(summaries)
        return send_file(output_path, as_attachment=True)
        
    except Exception as e:
        print(f"[ERROR] Batch export failed: {str(e)}")
        return jsonify({'error': str(e)}), 500

# ============================================================================
# Analytics Routes
# ============================================================================

@app.route('/analytics')
def analytics_dashboard():
    """Analytics dashboard"""
    try:
        summaries = db.get_summaries() or []
        
        # Get comprehensive statistics
        category_stats = categorizer.get_category_stats(summaries)
        
        # Processing statistics
        ai_provider_stats = {}
        for summary in summaries:
            provider = summary.get('ai_provider', 'unknown')
            ai_provider_stats[provider] = ai_provider_stats.get(provider, 0) + 1
        
        # Monthly statistics
        monthly_stats = {}
        for summary in summaries:
            created_at = summary.get('created_at', '')
            if created_at:
                try:
                    date_obj = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    month_key = date_obj.strftime('%Y-%m')
                    monthly_stats[month_key] = monthly_stats.get(month_key, 0) + 1
                except:
                    pass
        
        return render_template('analytics.html',
                             category_stats=category_stats,
                             ai_provider_stats=ai_provider_stats,
                             monthly_stats=monthly_stats,
                             total_summaries=len(summaries))
        
    except Exception as e:
        print(f"[ERROR] Analytics error: {e}")
        return render_template('analytics.html', 
                             category_stats={}, 
                             ai_provider_stats={}, 
                             monthly_stats={},
                             total_summaries=0)

# ============================================================================
# Enhanced Q&A Chatbot Routes
# ============================================================================

@app.route('/chatbot')
def chatbot_interface():
    """Enhanced Q&A chatbot interface"""
    try:
        summaries = db.get_summaries() or []
        return render_template('chatbot_enhanced.html', summaries=summaries)
    except Exception as e:
        print(f"[ERROR] Chatbot interface error: {e}")
        return render_template('chatbot_enhanced.html', summaries=[])

@app.route('/api/ask_question', methods=['POST'])
def ask_question():
    """Enhanced Q&A processing with caching"""
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({'status': 'error', 'message': 'Question is required'})
        
        # Check cache first
        cached_answer = cache.get_ai_response(question, 'qa_system')
        if cached_answer:
            return jsonify({
                'status': 'success',
                'answer': cached_answer,
                'cached': True
            })
        
        # Get answer from chatbot
        result = chatbot.ask_question(question)
        
        # Cache the result
        if result.get('status') == 'success':
            cache.cache_ai_response(question, result.get('answer', ''), 'qa_system', expire=1800)
        
        return jsonify(result)
        
    except Exception as e:
        print(f"[ERROR] Q&A processing failed: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

# ============================================================================
# Enhanced Video Highlights Routes  
# ============================================================================

@app.route('/highlights')
def highlights_interface():
    """Enhanced video highlights interface"""
    return render_template('highlights_enhanced.html')

@app.route('/extract_highlights/<int:video_id>', methods=['POST'])
def extract_highlights(video_id):
    """Enhanced highlight extraction with progress tracking"""
    try:
        print(f"[INFO] Extracting highlights for video ID: {video_id}")
        
        # Get video data
        video = db.get_summary(video_id)
        if not video:
            return jsonify({'status': 'error', 'message': 'Video not found'})
        
        # Create progress tracking
        with ProgressContext("highlight_extraction", video.get('url', ''), 'default') as progress:
            
            progress.update(10, "Starting highlight analysis...")
            
            # Extract highlights
            result = highlight_extractor.extract_highlights_from_video(
                video_id=str(video_id),
                video_url=video.get('url', ''),
                video_title=video.get('title', 'Unknown')
            )
            
            progress.update(100, "Highlight extraction completed")
        
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
# Enhanced SRT Chaptering Routes
# ============================================================================

@app.route('/srt-chaptering')
def srt_chaptering():
    """Enhanced SRT Upload and Chaptering page"""
    return render_template('srt_chaptering_enhanced.html')

@app.route('/process_srt', methods=['POST'])
def process_srt():
    """Enhanced SRT processing with progress tracking"""
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
        user_id = request.form.get('user_id', 'default')
        
        # Process with progress tracking
        with ProgressContext("srt_processing", user_id=user_id) as progress:
            
            progress.update(20, "Parsing SRT file...")
            
            # Process SRT
            result = srt_system.create_chapters_from_srt(
                srt_content=srt_content,
                video_title=video_title,
                chapter_count=chapter_count,
                min_chapter_duration=min_duration
            )
            
            if 'error' in result:
                return jsonify({'status': 'error', 'message': result['error']})
            
            progress.update(100, "SRT processing completed")
        
        return jsonify({
            'status': 'success',
            'message': 'SRT file processed successfully',
            'data': result
        })
        
    except Exception as e:
        print(f"[ERROR] SRT processing failed: {str(e)}")
        return jsonify({'status': 'error', 'message': f'Processing failed: {str(e)}'})

# ============================================================================
# WebSocket Progress Routes
# ============================================================================

@app.route('/api/job_status/<job_id>')
def get_job_status(job_id):
    """Get job status via REST API"""
    try:
        job = progress_tracker.get_job_status(job_id)
        if job:
            return jsonify({'status': 'success', 'job': job})
        else:
            return jsonify({'status': 'error', 'message': 'Job not found'}), 404
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/user_jobs/<user_id>')
def get_user_jobs(user_id):
    """Get all jobs for a user"""
    try:
        jobs = progress_tracker.get_user_jobs(user_id)
        return jsonify({'status': 'success', 'jobs': jobs})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# ============================================================================
# Enhanced Search Routes
# ============================================================================

@app.route('/semantic_search', methods=['POST'])
def semantic_search():
    """Enhanced semantic search with caching"""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({'status': 'error', 'message': 'Query is required'})
        
        # Check cache first
        cached_results = cache.get_search_results(query)
        if cached_results:
            return jsonify({
                'status': 'success',
                'results': cached_results,
                'query': query,
                'cached': True
            })
        
        # Use chatbot's search capability
        results = chatbot.search_summaries(query, limit=5)
        
        # Cache results
        cache.cache_search_results(query, results)
        
        return jsonify({
            'status': 'success',
            'results': results,
            'query': query,
            'cached': False
        })
        
    except Exception as e:
        print(f"[ERROR] Search failed: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e), 'results': []})

# ============================================================================
# System Routes
# ============================================================================

@app.route('/api/system_stats')
def system_stats():
    """Get system statistics"""
    try:
        summaries = db.get_summaries() or []
        
        # Active jobs
        active_jobs = len([
            job for job in progress_tracker.active_jobs.values() 
            if job['status'] == 'in_progress'
        ])
        
        # Cache stats (if Redis available)
        cache_stats = {}
        if cache.use_redis:
            try:
                info = cache.redis_client.info()
                cache_stats = {
                    'connected_clients': info.get('connected_clients', 0),
                    'used_memory_human': info.get('used_memory_human', '0B'),
                    'keyspace_hits': info.get('keyspace_hits', 0),
                    'keyspace_misses': info.get('keyspace_misses', 0)
                }
            except:
                cache_stats = {'error': 'Could not get Redis stats'}
        
        return jsonify({
            'total_summaries': len(summaries),
            'active_jobs': active_jobs,
            'cache_enabled': cache.use_redis,
            'cache_stats': cache_stats,
            'uptime': 'N/A'  # Could implement uptime tracking
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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
    os.makedirs('exports', exist_ok=True)
    
    # Clean up old jobs periodically
    progress_tracker.cleanup_completed_jobs()
    
    # Run application with SocketIO
    debug_mode = os.getenv('DEBUG', 'false').lower() == 'true'
    port = int(os.getenv('PORT', 5000))
    
    print(f"🚀 Starting Enhanced YouTube Summarizer on http://localhost:{port}")
    print("Features enabled:")
    print("  ✅ Real-time progress tracking")
    print("  ✅ Batch video processing")
    print("  ✅ PDF export functionality")
    print("  ✅ Auto-categorization & tagging")
    print("  ✅ Redis caching" if cache.use_redis else "  ⚠️  Memory caching (Redis not available)")
    
    socketio.run(app, debug=debug_mode, host='0.0.0.0', port=port)