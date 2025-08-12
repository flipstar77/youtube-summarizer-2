from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory, send_file
from datetime import datetime, timedelta
import os
import json
from dotenv import load_dotenv

from transcript_extractor import TranscriptExtractor
from enhanced_summarizer import EnhancedSummarizer, TextSummarizer
from database import Database
from text_to_speech import TextToSpeech
from supabase_client import SupabaseDatabase
from video_downloader import VideoDownloader
from vector_embeddings import create_embedding_service, SummaryVectorizer
from youtube_metadata import YouTubeMetadata
from chatbot_qa import VideoQAChatbot
from video_highlight_extractor import VideoHighlightExtractor
from channel_subscriptions import ChannelSubscriptionManager
from settings_manager import SettingsManager
from automation_scheduler import AutomationScheduler

load_dotenv()

# Import Data Access Layer and Services
from db.queries import dal
from services.vector_search import vector_search_service
from services.summary import summary_service

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-change-this')
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Force disable all caching
app.jinja_env.auto_reload = True
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Initialize components - choose between SQLite and Supabase
USE_SUPABASE = os.getenv('USE_SUPABASE', 'false').lower() == 'true'

if USE_SUPABASE:
    try:
        db = SupabaseDatabase()
        print("[OK] Using Supabase as database backend")
    except Exception as e:
        print(f"[WARNING] Supabase connection failed, falling back to SQLite: {str(e)}")
        db = Database()
        USE_SUPABASE = False
else:
    db = Database()
    print("[INFO] Using SQLite as database backend")

# Initialize enhanced AI summarizer
try:
    enhanced_summarizer = EnhancedSummarizer()
    print(f"[OK] Enhanced AI summarizer initialized with providers: {list(enhanced_summarizer.providers.keys())}")
except Exception as e:
    print(f"[WARNING] Enhanced summarizer failed, falling back to OpenAI only: {str(e)}")
    enhanced_summarizer = None

# Initialize vector embedding service
try:
    # Try OpenAI embeddings first (higher quality), fallback to sentence transformers
    use_openai = os.getenv('OPENAI_API_KEY') is not None
    embedding_service = create_embedding_service(use_openai=use_openai)
    vectorizer = SummaryVectorizer(embedding_service)
    print(f"[OK] Vector embeddings initialized using {'OpenAI' if use_openai else 'Sentence Transformers'}")
except Exception as e:
    print(f"[WARNING] Vector embeddings disabled: {str(e)}")
    embedding_service = None
    vectorizer = None

@app.route('/')
def index():
    """Dashboard - show all summaries"""
    summaries = db.get_all_summaries()
    
    # Calculate recent summaries (last 7 days)
    recent_count = 0
    if summaries:
        week_ago = (datetime.now() - timedelta(days=7)).isoformat()
        recent_count = sum(1 for s in summaries if s['created_at'] > week_ago)
    
    return render_template('index.html', 
                         summaries=summaries, 
                         recent_count=recent_count)

@app.route('/new', methods=['GET', 'POST'])
def new_summary():
    """Create new summary"""
    if request.method == 'GET':
        # Get available voices for the form
        voices = []
        try:
            elevenlabs_api_key = os.getenv('ELEVENLABS_API_KEY')
            if elevenlabs_api_key:
                tts = TextToSpeech(elevenlabs_api_key)
                voices = tts.get_available_voices()
                print(f"[INFO] Loaded {len(voices)} ElevenLabs voices for form")
            else:
                print("[WARNING] No ElevenLabs API key found")
        except Exception as e:
            print(f"[ERROR] Failed to load voices: {str(e)}")
            print(f"[DEBUG] Exception type: {type(e)}")
            import traceback
            traceback.print_exc()
            pass
        
        print(f"[DEBUG] Passing {len(voices)} voices to template")
        # Add timestamp to force browser cache refresh
        import time
        return render_template('new_summary.html', voices=voices, timestamp=int(time.time()))
    
    try:
        # Get form data
        url = request.form['url']
        summary_type = request.form['summary_type']
        language = request.form.get('language', 'en')
        generate_audio = request.form.get('generate_audio') == 'on'
        generate_srt = request.form.get('generate_srt') == 'on'
        voice_id = request.form.get('voice_id', '21m00Tcm4TlvDq8ikWAM')
        ai_provider = request.form.get('ai_provider', 'openai')
        ai_model = request.form.get('ai_model')
        custom_prompt = request.form.get('custom_prompt', '').strip()
        
        # Validate inputs
        if not url:
            flash('Please provide a YouTube URL', 'error')
            return render_template('new_summary.html')
        
        # Extract transcript
        extractor = TranscriptExtractor()
        transcript_data = extractor.get_transcript(url, language)
        
        # Extract YouTube metadata (title, uploader, etc.)
        metadata_extractor = YouTubeMetadata()
        video_metadata = metadata_extractor.get_video_info(url)
        
        # Generate summary using enhanced AI providers
        if enhanced_summarizer and ai_provider in enhanced_summarizer.providers:
            print(f"[INFO] Using {ai_provider} for summary generation")
            summary = enhanced_summarizer.summarize(
                transcript_data['transcript'], 
                summary_type, 
                provider=ai_provider,
                model=ai_model,
                custom_prompt=custom_prompt if summary_type == 'custom' else None
            )
        else:
            # Fallback to original TextSummarizer
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                flash('No AI provider available. Please configure at least one API key.', 'error')
                return render_template('new_summary.html')
            
            summarizer = TextSummarizer(api_key)
            summary = summarizer.summarize(transcript_data['transcript'], summary_type)
        
        # Use extracted video title and uploader
        title = video_metadata.get('title', f"YouTube Video {transcript_data['video_id']}")
        uploader = video_metadata.get('uploader', 'Unknown Channel')
        duration = video_metadata.get('duration', 0)
        
        # Map new summary types to database-compatible types
        db_summary_type = summary_type
        if summary_type == 'tutorial':
            db_summary_type = 'detailed'  # Map tutorial to detailed for database
        elif summary_type == 'professional':
            db_summary_type = 'detailed'  # Map professional to detailed for database
        
        # Generate audio if requested
        audio_file = None
        if generate_audio:
            try:
                elevenlabs_api_key = os.getenv('ELEVENLABS_API_KEY')
                if elevenlabs_api_key:
                    tts = TextToSpeech(elevenlabs_api_key)
                    audio_filename = f"summary_{transcript_data['video_id']}_{summary_type}.mp3"
                    audio_file_full_path = tts.generate_speech(summary, voice_id, audio_filename)
                    # Convert to relative path for web serving (remove static/ prefix)
                    audio_file = audio_file_full_path.replace('static/', '').replace('static\\', '')
                else:
                    flash('ElevenLabs API key not configured. Audio generation skipped.', 'warning')
            except Exception as e:
                flash(f'Audio generation failed: {str(e)}', 'warning')
        
        # Generate vector embedding if service is available
        embedding = None
        if vectorizer and USE_SUPABASE:
            try:
                vectorized_data = vectorizer.vectorize_summary({
                    'title': title,
                    'summary': summary,
                    'summary_type': summary_type
                })
                embedding = vectorized_data.get('embedding')
                print(f"[INFO] Generated vector embedding for summary")
            except Exception as e:
                print(f"[WARNING] Failed to generate embedding: {str(e)}")

        # Save to database (with vector if available)
        try:
            if USE_SUPABASE and hasattr(db, 'save_summary_with_vector'):
                summary_id = db.save_summary_with_vector(
                    video_id=transcript_data['video_id'],
                    url=url,
                    title=title,
                    summary_type=db_summary_type,  # Use database-compatible type
                    summary=summary,
                    transcript_length=len(transcript_data['transcript']),
                    embedding=embedding,
                    audio_file=audio_file,
                    voice_id=voice_id if audio_file else None,
                    uploader=uploader,
                    duration=duration
                )
            else:
                summary_id = db.save_summary(
                    video_id=transcript_data['video_id'],
                    url=url,
                    title=title,
                    summary_type=db_summary_type,  # Use database-compatible type
                    summary=summary,
                    transcript_length=len(transcript_data['transcript']),
                    audio_file=audio_file,
                    voice_id=voice_id if audio_file else None,
                    uploader=uploader,
                    duration=duration
                )
        except Exception as e:
            # Fallback to basic save without vectors/metadata if schema issues
            print(f"[WARNING] Failed to save with vectors/metadata, using basic save: {str(e)}")
            try:
                summary_id = db.save_summary(
                    video_id=transcript_data['video_id'],
                    url=url,
                    title=title,
                    summary_type=db_summary_type,  # Use database-compatible type
                    summary=summary,
                    transcript_length=len(transcript_data['transcript']),
                    audio_file=audio_file,
                    voice_id=voice_id if audio_file else None
                )
                print(f"[INFO] Summary saved with basic method, ID: {summary_id}")
            except Exception as e2:
                # Final fallback to SQLite if Supabase fails completely
                print(f"[WARNING] Supabase save failed, using SQLite fallback: {str(e2)}")
                from database import Database
                basic_db = Database()
                summary_id = basic_db.save_summary(
                    video_id=transcript_data['video_id'],
                    url=url,
                    title=title,
                    summary_type=db_summary_type,  # Use database-compatible type
                    summary=summary,
                    transcript_length=len(transcript_data['transcript']),
                    audio_file=audio_file,
                    voice_id=voice_id if audio_file else None
                )
        
        # Generate SRT file if requested
        if generate_srt:
            try:
                srt_content = create_srt_from_text(summary, transcript_data.get('duration', 0))
                srt_filename = f"srt/summary_{transcript_data['video_id']}.srt"
                srt_path = os.path.join('static', srt_filename)
                
                # Ensure srt directory exists
                os.makedirs(os.path.dirname(srt_path), exist_ok=True)
                
                # Save SRT file
                with open(srt_path, 'w', encoding='utf-8') as f:
                    f.write(srt_content)
                
                flash('SRT subtitle file created successfully!', 'success')
            except Exception as e:
                flash(f'SRT generation failed: {str(e)}', 'warning')
        
        success_message = 'Summary generated successfully!'
        if generate_srt and generate_audio:
            success_message += ' Audio and SRT files created.'
        elif generate_srt:
            success_message += ' SRT subtitle file created.'
        elif generate_audio:
            success_message += ' Audio file created.'
            
        flash(success_message, 'success')
        return redirect(url_for('view_summary', summary_id=summary_id))
        
    except Exception as e:
        # Log the full error for debugging
        print(f"[ERROR] Summary generation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        flash(f'Error: {str(e)}', 'error')
        
        # Get voices again for the form
        voices = []
        try:
            elevenlabs_api_key = os.getenv('ELEVENLABS_API_KEY')
            if elevenlabs_api_key:
                tts = TextToSpeech(elevenlabs_api_key)
                voices = tts.get_available_voices()
                print(f"[INFO] Loaded {len(voices)} ElevenLabs voices for error case")
            else:
                print("[WARNING] No ElevenLabs API key found for error case")
        except Exception as e:
            print(f"[ERROR] Failed to load voices in error case: {str(e)}")
            pass
        
        return render_template('new_summary.html', voices=voices)

@app.route('/summary/<int:summary_id>')
def view_summary(summary_id):
    """View specific summary"""
    summary = db.get_summary(summary_id)
    
    if not summary:
        flash('Summary not found', 'error')
        return redirect(url_for('index'))
    
    # Get available voices for audio generation
    voices = []
    try:
        elevenlabs_api_key = os.getenv('ELEVENLABS_API_KEY')
        if elevenlabs_api_key:
            tts = TextToSpeech(elevenlabs_api_key)
            voices = tts.get_available_voices()
            print(f"[INFO] Loaded {len(voices)} ElevenLabs voices for summary view")
        else:
            print("[WARNING] No ElevenLabs API key found")
    except Exception as e:
        print(f"[ERROR] Failed to load voices for summary view: {str(e)}")
        pass
    
    return render_template('view_summary.html', summary=summary, voices=voices)

@app.route('/delete/<int:summary_id>', methods=['DELETE'])
def delete_summary(summary_id):
    """Delete summary"""
    try:
        success = db.delete_summary(summary_id)
        
        if success:
            return jsonify({'status': 'success', 'message': 'Summary deleted'})
        else:
            return jsonify({'status': 'error', 'message': 'Summary not found'}), 404
            
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/summaries')
def api_summaries():
    """API endpoint to get all summaries"""
    try:
        summaries = db.get_all_summaries()
        return jsonify({'status': 'success', 'data': summaries})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/summary/<int:summary_id>')
def api_summary(summary_id):
    """API endpoint to get specific summary"""
    try:
        summary = db.get_summary(summary_id)
        if summary:
            return jsonify({'status': 'success', 'data': summary})
        else:
            return jsonify({'status': 'error', 'message': 'Summary not found'}), 404
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/generate_audio/<int:summary_id>', methods=['POST'])
def generate_audio(summary_id):
    """Generate audio for existing summary"""
    try:
        voice_id = request.json.get('voice_id', '21m00Tcm4TlvDq8ikWAM')
        
        # Get the summary
        summary = db.get_summary(summary_id)
        if not summary:
            return jsonify({'status': 'error', 'message': 'Summary not found'}), 404
        
        # Generate audio
        elevenlabs_api_key = os.getenv('ELEVENLABS_API_KEY')
        if not elevenlabs_api_key:
            return jsonify({'status': 'error', 'message': 'ElevenLabs API key not configured'}), 500
        
        tts = TextToSpeech(elevenlabs_api_key)
        audio_filename = f"summary_{summary['video_id']}_{summary['summary_type']}.mp3"
        audio_file_full_path = tts.generate_speech(summary['summary'], voice_id, audio_filename)
        
        # Convert to relative path for web serving (remove static/ prefix)
        audio_file_relative = audio_file_full_path.replace('static/', '').replace('static\\', '')
        
        # Update database
        db.update_audio_file(summary_id, audio_file_relative, voice_id)
        
        return jsonify({
            'status': 'success', 
            'message': 'Audio generated successfully',
            'audio_url': f'/static/{audio_file_relative}'
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/voices')
def get_voices():
    """Get available ElevenLabs voices"""
    try:
        elevenlabs_api_key = os.getenv('ELEVENLABS_API_KEY')
        if not elevenlabs_api_key:
            return jsonify({'status': 'error', 'message': 'ElevenLabs API key not configured'}), 500
        
        tts = TextToSpeech(elevenlabs_api_key)
        voices = tts.get_available_voices()
        
        return jsonify({'status': 'success', 'voices': voices})
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/ai_providers')
def get_ai_providers():
    """Get available AI providers and their models"""
    try:
        if enhanced_summarizer:
            providers = enhanced_summarizer.get_available_providers()
            return jsonify({
                'status': 'success', 
                'providers': providers,
                'default': 'openai' if 'openai' in providers else list(providers.keys())[0] if providers else None
            })
        else:
            # Fallback to OpenAI only
            return jsonify({
                'status': 'success',
                'providers': {
                    'openai': {
                        'models': ['gpt-4', 'gpt-3.5-turbo'],
                        'type': 'openai'
                    }
                },
                'default': 'openai'
            })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/generate_srt/<int:summary_id>', methods=['POST'])
def generate_srt(summary_id):
    """Generate SRT subtitle file from summary"""
    try:
        # Get the summary
        summary = db.get_summary(summary_id)
        if not summary:
            return jsonify({'status': 'error', 'message': 'Summary not found'}), 404
        
        # Use the FFmpeg SRT processor agent to create SRT file
        from datetime import datetime
        import tempfile
        
        # Create SRT content from summary
        srt_content = create_srt_from_text(summary['summary'], summary.get('transcript_length', 0))
        
        # Create temporary file for download
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.srt', delete=False, encoding='utf-8')
        temp_file.write(srt_content)
        temp_file.close()
        
        # Return file for download
        return send_file(
            temp_file.name,
            as_attachment=True,
            download_name=f"summary_{summary['video_id']}.srt",
            mimetype='application/x-subrip'
        )
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

def create_srt_from_text(text, estimated_duration=0):
    """Create SRT format from text content"""
    # Split text into sentences for subtitle segments
    sentences = [s.strip() for s in text.replace('\n', ' ').split('.') if s.strip()]
    
    # Estimate timing - roughly 2-3 seconds per sentence
    base_duration = max(estimated_duration // len(sentences) if sentences else 3, 2)
    
    srt_content = []
    start_time = 0
    
    for i, sentence in enumerate(sentences, 1):
        if not sentence:
            continue
            
        # Calculate duration based on sentence length (reading speed ~200 WPM)
        words = len(sentence.split())
        duration = max(int(words / 3.33), 2)  # 200 WPM = 3.33 words per second
        
        # Format timestamps
        start_hours = start_time // 3600
        start_minutes = (start_time % 3600) // 60
        start_seconds = start_time % 60
        
        end_time = start_time + duration
        end_hours = end_time // 3600
        end_minutes = (end_time % 3600) // 60
        end_seconds_final = end_time % 60
        
        # Add SRT entry
        srt_content.append(f"{i}")
        srt_content.append(f"{start_hours:02d}:{start_minutes:02d}:{start_seconds:02d},000 --> {end_hours:02d}:{end_minutes:02d}:{end_seconds_final:02d},000")
        srt_content.append(f"{sentence}.")
        srt_content.append("")  # Empty line between entries
        
        start_time = end_time + 1  # 1 second gap between subtitles
    
    return '\n'.join(srt_content)

@app.route('/download_video/<int:summary_id>', methods=['POST'])
def download_video(summary_id):
    """Download video with specified options"""
    try:
        # Get the summary
        summary = db.get_summary(summary_id)
        if not summary:
            return jsonify({'status': 'error', 'message': 'Summary not found'}), 404
        
        # Get download options
        data = request.get_json()
        quality = data.get('quality', 'best')
        format_type = data.get('format', 'mp4')
        include_subtitles = data.get('include_subtitles', False)
        
        # Initialize video downloader
        downloader = VideoDownloader()
        
        # Get SRT content if subtitles are requested
        srt_content = None
        if include_subtitles:
            srt_content = create_srt_from_text(summary['summary'], summary.get('transcript_length', 0))
        
        try:
            # Download video
            video_path = downloader.download_video(
                url=summary['url'],
                quality=quality,
                format_type=format_type,
                with_subtitles=include_subtitles,
                srt_content=srt_content
            )
            
            # Get filename from path
            filename = os.path.basename(video_path)
            
            # Return file for download
            return send_file(
                video_path,
                as_attachment=True,
                download_name=filename,
                mimetype=f'video/{format_type}'
            )
            
        except Exception as e:
            return jsonify({'status': 'error', 'message': f'Download failed: {str(e)}'}), 500
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/video_info/<int:summary_id>')
def get_video_info(summary_id):
    """Get video information for download options"""
    try:
        # Get the summary
        summary = db.get_summary(summary_id)
        if not summary:
            return jsonify({'status': 'error', 'message': 'Summary not found'}), 404
        
        # Initialize video downloader
        downloader = VideoDownloader()
        
        # Get video info
        info = downloader.get_video_info(summary['url'])
        
        return jsonify({
            'status': 'success',
            'info': info,
            'quality_options': downloader.get_quality_options(),
            'format_options': downloader.get_download_formats()
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/semantic_search', methods=['POST'])
def semantic_search():
    """Semantic search using vector embeddings with new RPC functions"""
    try:
        if not vectorizer:
            return jsonify({'status': 'error', 'message': 'Vector search not available'}), 400
        
        data = request.get_json()
        query = data.get('query', '').strip()
        
        # Get settings from settings manager with fallbacks
        default_threshold = settings_manager.get_setting('vector_search_threshold', 0.75) if settings_manager else 0.75
        default_limit = settings_manager.get_setting('vector_search_limit', 10) if settings_manager else 10
        vector_search_enabled = settings_manager.get_setting('vector_search_enabled', True) if settings_manager else True
        
        threshold = data.get('threshold', default_threshold)
        limit = data.get('limit', default_limit)
        
        if not query:
            return jsonify({'status': 'error', 'message': 'Query cannot be empty'}), 400
        
        if not vector_search_enabled:
            return jsonify({
                'status': 'disabled', 
                'message': 'Vector search is disabled in settings',
                'results': []
            }), 200
        
        # Use DAL for vector search
        similar_summaries = vector_search_service.search_by_text(query, threshold, limit)
        
        return jsonify({
            'status': 'success',
            'query': query,
            'threshold': threshold,
            'results': similar_summaries,
            'count': len(similar_summaries)
        })
        
    except Exception as e:
        print(f"[ERROR] Semantic search failed: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/similar/<int:summary_id>')
def similar_api(summary_id):
    """Simple similar summaries API - using DAL"""
    count = int(request.args.get("count", 5))
    
    # Validate count parameter
    if count < 1 or count > 50:
        return jsonify({"error": "count must be between 1 and 50"}), 400
    
    items = vector_search_service.find_similar(summary_id, count)
    return jsonify({
        "summary_id": summary_id,
        "count": count,
        "results": items
    })

@app.route('/semantic-search', methods=['POST'])
def semantic_search_rpc():
    """Semantic search using DAL"""
    body = request.get_json(force=True) or {}
    embedding = body.get("embedding")  # Liste[float] Länge 1536
    threshold = float(body.get("threshold", 0.75))
    count = int(body.get("count", 10))

    if not embedding or len(embedding) != 1536:
        return jsonify({"error": "embedding (1536 floats) required"}), 400

    items = vector_search_service.search_by_embedding(embedding, threshold, count)
    return jsonify({"results": items})

@app.route('/vectorize_existing', methods=['POST'])
def vectorize_existing_summaries():
    """Vectorize existing summaries that don't have embeddings"""
    try:
        if not (USE_SUPABASE and embedding_service and vectorizer):
            return jsonify({'status': 'error', 'message': 'Vector service not available'}), 400
        
        # Get summaries without embeddings
        summaries_to_vectorize = db.get_summaries_without_embeddings()
        
        if not summaries_to_vectorize:
            return jsonify({'status': 'success', 'message': 'All summaries already have embeddings'})
        
        vectorized_count = 0
        for summary in summaries_to_vectorize:
            try:
                # Generate embedding
                vectorized_data = vectorizer.vectorize_summary(summary)
                embedding = vectorized_data.get('embedding')
                
                if embedding:
                    # Update database
                    db.update_summary_embedding(summary['id'], embedding)
                    vectorized_count += 1
                    print(f"[INFO] Vectorized summary {summary['id']}: {summary.get('title', 'No title')}")
                    
            except Exception as e:
                print(f"[WARNING] Failed to vectorize summary {summary['id']}: {str(e)}")
                continue
        
        return jsonify({
            'status': 'success',
            'message': f'Vectorized {vectorized_count} summaries',
            'vectorized_count': vectorized_count,
            'total_found': len(summaries_to_vectorize)
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/voice_debug')
def voice_debug():
    """Debug page for voice issues"""
    return send_from_directory('.', 'voice_debug.html')

@app.route('/voice_test')
def voice_test():
    """Simple voice test page"""
    return send_from_directory('.', 'voice_test.html')

@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

# Initialize components
chatbot = None
highlight_extractor = None
subscription_manager = None
settings_manager = None
automation_scheduler = None

try:
    chatbot = VideoQAChatbot()
    print("[OK] Video Q&A Chatbot initialized")
except Exception as e:
    print(f"[WARNING] Chatbot disabled: {str(e)}")

try:
    highlight_extractor = VideoHighlightExtractor()
    print("[OK] Video Highlight Extractor initialized")
except Exception as e:
    print(f"[WARNING] Highlight Extractor disabled: {str(e)}")

try:
    subscription_manager = ChannelSubscriptionManager()
    print("[OK] Channel Subscription Manager initialized")
except Exception as e:
    print(f"[WARNING] Subscription Manager disabled: {str(e)}")

try:
    settings_manager = SettingsManager()
    settings_manager.initialize_default_settings()
    print("[OK] Settings Manager initialized")
except Exception as e:
    print(f"[WARNING] Settings Manager disabled: {str(e)}")

try:
    automation_scheduler = AutomationScheduler()
    automation_scheduler.start()
    print("[OK] Automation Scheduler initialized and started")
except Exception as e:
    print(f"[WARNING] Automation Scheduler disabled: {str(e)}")

def generate_highlight_name(text, index):
    """
    Generiert einen sinnvollen Namen für ein Highlight basierend auf dem Textinhalt
    """
    if not text or len(text.strip()) < 10:
        return f"Highlight #{index}"
    
    # Bereinige und kürze den Text
    clean_text = text.strip()
    
    # Suche nach markanten Wörtern oder Phrasen
    keywords = [
        # Tutorial/Lern-Begriffe
        ("setup", "Setup"), ("install", "Installation"), ("config", "Konfiguration"),
        ("tutorial", "Tutorial"), ("guide", "Anleitung"), ("how to", "Anleitung"),
        ("beginner", "Für Anfänger"), ("advanced", "Erweitert"), ("pro", "Profi-Tipp"),
        
        # Technische Begriffe
        ("error", "Fehler"), ("bug", "Bug"), ("fix", "Lösung"), ("solution", "Lösung"),
        ("update", "Update"), ("upgrade", "Upgrade"), ("feature", "Feature"),
        ("performance", "Performance"), ("optimization", "Optimierung"),
        
        # Gaming/Spiel-Begriffe
        ("level", "Level"), ("strategy", "Strategie"), ("tactic", "Taktik"),
        ("build", "Build"), ("upgrade", "Upgrade"), ("skill", "Skill"),
        ("boss", "Boss"), ("quest", "Quest"), ("achievement", "Erfolg"),
        
        # Allgemeine Begriffe
        ("tip", "Tipp"), ("trick", "Trick"), ("hack", "Hack"), ("secret", "Geheimnis"),
        ("mistake", "Fehler"), ("common", "Häufig"), ("important", "Wichtig"),
        ("warning", "Warnung"), ("note", "Hinweis"), ("remember", "Merken"),
        
        # Deutsche Begriffe
        ("tipp", "Tipp"), ("trick", "Trick"), ("fehler", "Fehler"), 
        ("wichtig", "Wichtig"), ("achtung", "Achtung"), ("hinweis", "Hinweis"),
        ("strategie", "Strategie"), ("anleitung", "Anleitung")
    ]
    
    text_lower = clean_text.lower()
    
    # Suche nach Schlüsselwörtern
    for keyword, label in keywords:
        if keyword in text_lower:
            # Nimm die ersten paar Wörter nach dem Keyword
            keyword_pos = text_lower.find(keyword)
            context_start = max(0, keyword_pos - 20)
            context_end = min(len(clean_text), keyword_pos + len(keyword) + 30)
            context = clean_text[context_start:context_end].strip()
            
            # Bereinige den Kontext
            words = context.split()[:5]  # Maximal 5 Wörter
            context_clean = ' '.join(words)
            
            if len(context_clean) > 10:
                return f"{label}: {context_clean}"
    
    # Fallback: Verwende die ersten Wörter des Textes
    words = clean_text.split()[:4]  # Erste 4 Wörter
    if len(words) >= 2:
        return ' '.join(words)
    
    # Letzter Fallback
    return f"Highlight #{index}"

@app.route('/chatbot')
def chatbot_interface():
    """Chatbot interface for Q&A with video collection"""
    try:
        summaries = db.get_all_summaries()
        return render_template('chatbot.html', summaries=summaries)
    except Exception as e:
        flash(f'Fehler beim Laden der Chatbot-Seite: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/api/ask_question', methods=['POST'])
def api_ask_question():
    """API endpoint for asking questions about the entire video collection"""
    try:
        if not chatbot:
            return jsonify({
                'status': 'error',
                'message': 'Chatbot nicht verfügbar'
            }), 400
        
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({
                'status': 'error',
                'message': 'Frage darf nicht leer sein'
            }), 400
        
        # Ask question to the chatbot
        response = chatbot.ask_question(question, max_videos=5)
        
        return jsonify({
            'status': 'success',
            'answer': response['answer'],
            'sources': response['sources'],
            'question': response['question'],
            'videos_analyzed': response.get('videos_analyzed', 0)
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/chat_transcript', methods=['POST'])
def api_chat_transcript():
    """API endpoint for chatting with a specific video transcript"""
    try:
        if not chatbot:
            return jsonify({
                'status': 'error',
                'message': 'Chatbot nicht verfügbar'
            }), 400
        
        data = request.get_json()
        question = data.get('question', '').strip()
        video_id = data.get('video_id', '').strip()
        
        if not question:
            return jsonify({
                'status': 'error',
                'message': 'Frage darf nicht leer sein'
            }), 400
        
        if not video_id:
            return jsonify({
                'status': 'error',
                'message': 'Video ID erforderlich'
            }), 400
        
        # Chat with specific transcript
        response = chatbot.chat_with_transcript(video_id, question)
        
        return jsonify({
            'status': 'success',
            'answer': response['answer'],
            'video_info': response['video_info'],
            'question': response['question']
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/extract_highlights/<int:summary_id>', methods=['POST'])
def extract_highlights(summary_id):
    """Extrahiert Highlights aus einem Video"""
    try:
        if not highlight_extractor:
            return jsonify({
                'status': 'error',
                'message': 'Highlight Extractor nicht verfügbar'
            }), 400
        
        # Hole Video-Daten aus der Datenbank
        summary = db.get_summary(summary_id)
        if not summary:
            return jsonify({
                'status': 'error', 
                'message': 'Video nicht gefunden'
            }), 404
        
        data = request.get_json() or {}
        highlight_count = data.get('highlight_count', 5)
        min_duration = data.get('min_duration', 10)
        max_duration = data.get('max_duration', 60)
        
        # Hole SRT-Content falls verfügbar
        srt_content = None
        if hasattr(summary, 'get') and summary.get('transcript'):
            # Konvertiere Transcript zu SRT-Format (vereinfacht)
            srt_content = summary.get('transcript')
        
        print(f"[INFO] Starte Highlight-Extraktion für Video-ID: {summary['video_id']}")
        
        # Extrahiere Highlights - übergebe die Database-ID für besseres Lookup
        result = highlight_extractor.extract_highlights_from_video(
            video_id=str(summary_id),  # Verwende die Database-ID  
            video_url=summary['url'],
            srt_content=srt_content,
            highlight_count=highlight_count,
            min_duration=min_duration,
            max_duration=max_duration
        )
        
        # Speichere Highlight-Ergebnisse für Dashboard-Anzeige
        if result.get('status') == 'success':
            try:
                import json
                highlights_data_dir = "D:/mcp/highlights_data"
                os.makedirs(highlights_data_dir, exist_ok=True)
                
                # Speichere als JSON mit Zeitstempel
                result_with_meta = {
                    **result,
                    'summary_id': summary_id,
                    'video_title': summary['title'],
                    'video_url': summary['url'],
                    'created_at': datetime.now().isoformat(),
                    'youtube_video_id': summary['video_id']
                }
                
                json_filename = f"highlights_{summary_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                json_path = os.path.join(highlights_data_dir, json_filename)
                
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(result_with_meta, f, indent=2, ensure_ascii=False)
                    
                print(f"[INFO] Highlight-Ergebnisse gespeichert: {json_filename}")
                
            except Exception as e:
                print(f"[WARNING] Highlight-Ergebnisse konnten nicht gespeichert werden: {str(e)}")
                # Nicht kritisch, Extraktion war erfolgreich
        
        if result.get('status') == 'success':
            return jsonify({
                'status': 'success',
                'message': f'{result.get("total_highlights", 0)} Highlights extrahiert',
                'clips': result.get('clips', []),
                'compilation': result.get('compilation'),
                'video_id': summary['video_id'],
                'title': summary['title']
            })
        else:
            return jsonify({
                'status': 'error',
                'message': result.get('error', 'Highlight-Extraktion fehlgeschlagen')
            }), 500
            
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Fehler bei Highlight-Extraktion: {str(e)}'
        }), 500

@app.route('/highlights')
def highlights_overview():
    """Übersicht aller extrahierten Highlights - gruppiert nach Videos"""
    try:
        highlights_data_dir = "D:/mcp/highlights_data"
        videos_with_highlights = {}
        
        # Lade JSON-basierte Highlight-Daten (neue Version)
        if os.path.exists(highlights_data_dir):
            for file in os.listdir(highlights_data_dir):
                if file.endswith('.json'):
                    try:
                        file_path = os.path.join(highlights_data_dir, file)
                        with open(file_path, 'r', encoding='utf-8') as f:
                            highlight_data = json.load(f)
                        
                        summary_id = highlight_data.get('summary_id')
                        video_title = highlight_data.get('video_title', 'Unknown Video')
                        
                        # Gruppiere nach Video
                        if summary_id not in videos_with_highlights:
                            videos_with_highlights[summary_id] = {
                                'summary_id': summary_id,
                                'video_title': video_title,
                                'video_url': highlight_data.get('video_url', ''),
                                'youtube_video_id': highlight_data.get('youtube_video_id', ''),
                                'analyses': [],
                                'total_highlights': 0,
                                'total_chapters': 0,
                                'total_clips_extracted': 0,
                                'total_clips_failed': 0,
                                'latest_analysis': None
                            }
                        
                        # Füge Analyse hinzu
                        analysis = {
                            'filename': file,
                            'created': datetime.fromisoformat(highlight_data.get('created_at')),
                            'highlights_count': highlight_data.get('highlights_identified', 0),
                            'chapters_count': len(highlight_data.get('chapters', [])),
                            'clips_extracted': highlight_data.get('clips_extracted', 0),
                            'clips_failed': highlight_data.get('clips_failed', 0),
                            'data': highlight_data,
                            'individual_highlights': []
                        }
                        
                        # Extrahiere individuelle Highlights mit sinnvollen Namen
                        clips = highlight_data.get('clips', [])
                        for i, clip in enumerate(clips):
                            # Generiere sinnvollen Namen aus dem Text
                            text = clip.get('text', '')
                            highlight_name = generate_highlight_name(text, i + 1)
                            
                            analysis['individual_highlights'].append({
                                'index': i,
                                'name': highlight_name,
                                'timestamp': f"{clip.get('start_time', '')} - {clip.get('end_time', '')}",
                                'score': clip.get('score', 0),
                                'status': clip.get('status', 'identified_only'),
                                'text': text[:100] + "..." if len(text) > 100 else text,
                                'reason': clip.get('reason', ''),
                                'duration': clip.get('duration', '')
                            })
                        
                        videos_with_highlights[summary_id]['analyses'].append(analysis)
                        
                        # Update totals
                        videos_with_highlights[summary_id]['total_highlights'] += analysis['highlights_count']
                        videos_with_highlights[summary_id]['total_chapters'] += analysis['chapters_count']
                        videos_with_highlights[summary_id]['total_clips_extracted'] += analysis['clips_extracted']
                        videos_with_highlights[summary_id]['total_clips_failed'] += analysis['clips_failed']
                        
                        # Track latest analysis
                        if (not videos_with_highlights[summary_id]['latest_analysis'] or 
                            analysis['created'] > videos_with_highlights[summary_id]['latest_analysis']['created']):
                            videos_with_highlights[summary_id]['latest_analysis'] = analysis
                            
                    except Exception as e:
                        print(f"[WARNING] Fehler beim Laden von {file}: {str(e)}")
                        continue
        
        # Sortiere Videos nach letzter Analyse
        video_list = list(videos_with_highlights.values())
        video_list.sort(key=lambda x: x['latest_analysis']['created'] if x['latest_analysis'] else datetime.min, reverse=True)
        
        # Berechne Gesamtstatistiken
        total_stats = {
            'total_videos': len(video_list),
            'total_analyses': sum(len(v['analyses']) for v in video_list),
            'total_highlights': sum(v['total_highlights'] for v in video_list),
            'total_chapters': sum(v['total_chapters'] for v in video_list),
            'total_clips': sum(v['total_clips_extracted'] for v in video_list)
        }
        
        return render_template('highlights.html', 
                             videos_with_highlights=video_list,
                             total_stats=total_stats)
        
    except Exception as e:
        flash(f'Fehler beim Laden der Highlights: {str(e)}', 'error')
        return render_template('highlights.html', highlights=[])

@app.route('/highlight_details/<filename>')
def get_highlight_details(filename):
    """Holt Details einer Highlight-Analyse"""
    try:
        highlights_data_dir = "D:/mcp/highlights_data"
        file_path = os.path.join(highlights_data_dir, filename)
        
        if os.path.exists(file_path) and filename.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                highlight_data = json.load(f)
            
            return jsonify({
                'status': 'success',
                'data': highlight_data
            })
        else:
            return jsonify({'status': 'error', 'message': 'Highlight-Datei nicht gefunden'}), 404
            
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/extract_single_clip/<int:summary_id>/<int:clip_index>', methods=['POST'])
def extract_single_clip(summary_id, clip_index):
    """Extrahiert einen einzelnen Clip basierend auf gespeicherten Highlight-Daten"""
    try:
        if not highlight_extractor:
            return jsonify({
                'status': 'error',
                'message': 'Highlight Extractor nicht verfügbar'
            }), 400
        
        # Finde die entsprechende Highlight-Analyse
        highlights_data_dir = "D:/mcp/highlights_data"
        highlight_data = None
        
        if os.path.exists(highlights_data_dir):
            for file in os.listdir(highlights_data_dir):
                if file.startswith(f"highlights_{summary_id}_") and file.endswith('.json'):
                    try:
                        file_path = os.path.join(highlights_data_dir, file)
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            if data.get('summary_id') == summary_id:
                                highlight_data = data
                                break
                    except Exception:
                        continue
        
        if not highlight_data:
            return jsonify({
                'status': 'error',
                'message': 'Highlight-Daten nicht gefunden'
            }), 404
        
        clips = highlight_data.get('clips', [])
        if clip_index >= len(clips):
            return jsonify({
                'status': 'error',
                'message': 'Clip-Index ungültig'
            }), 400
        
        target_clip = clips[clip_index]
        
        # Hole Video-Info aus der Datenbank
        summary = db.get_summary(summary_id)
        if not summary:
            return jsonify({
                'status': 'error',
                'message': 'Video nicht in Datenbank gefunden'
            }), 404
        
        # Lade Video herunter falls nötig
        video_path = highlight_extractor._download_video(summary['url'], str(summary_id))
        if not video_path:
            return jsonify({
                'status': 'error',
                'message': 'Video-Download fehlgeschlagen'
            }), 500
        
        # Extrahiere einzelnen Clip
        clip_path = highlight_extractor._extract_video_clip(
            video_path, target_clip, str(summary_id), clip_index + 1
        )
        
        if clip_path:
            # Update den Status in den gespeicherten Daten
            clips[clip_index]['status'] = 'extracted'
            clips[clip_index]['clip_path'] = clip_path
            highlight_data['clips_extracted'] = len([c for c in clips if c.get('status') == 'extracted'])
            highlight_data['clips_failed'] = len([c for c in clips if c.get('status') == 'identified_only'])
            
            # Speichere aktualisierte Daten
            for file in os.listdir(highlights_data_dir):
                if file.startswith(f"highlights_{summary_id}_") and file.endswith('.json'):
                    try:
                        file_path = os.path.join(highlights_data_dir, file)
                        with open(file_path, 'w', encoding='utf-8') as f:
                            json.dump(highlight_data, f, indent=2, ensure_ascii=False)
                        break
                    except Exception:
                        continue
            
            return jsonify({
                'status': 'success',
                'message': 'Clip erfolgreich extrahiert',
                'clip_path': clip_path
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Clip-Extraktion fehlgeschlagen (FFmpeg-Fehler)'
            }), 500
            
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Fehler bei Clip-Extraktion: {str(e)}'
        }), 500

@app.route('/download_highlight/<filename>')
def download_highlight(filename):
    """Download eines Highlight-Clips"""
    try:
        highlights_dir = "D:/mcp/highlights"
        file_path = os.path.join(highlights_dir, filename)
        
        if os.path.exists(file_path) and filename.endswith('.mp4'):
            return send_file(file_path, as_attachment=True)
        else:
            return jsonify({'error': 'Datei nicht gefunden'}), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ==============================
# SUBSCRIPTION MANAGEMENT ROUTES
# ==============================

@app.route('/subscriptions')
def subscriptions_dashboard():
    """Dashboard for managing YouTube channel subscriptions"""
    if not subscription_manager:
        flash('Subscription manager is not available', 'error')
        return redirect('/')
    
    try:
        subscriptions = subscription_manager.get_all_subscriptions()
        discovered_videos = subscription_manager.get_discovered_videos(limit=20)
        
        # Get stats
        stats = {
            'total_subscriptions': len(subscriptions),
            'total_videos': len(discovered_videos),
            'unprocessed_videos': len([v for v in discovered_videos if not v.get('auto_processed', False)])
        }
        
        return render_template('subscriptions.html', 
                             subscriptions=subscriptions,
                             discovered_videos=discovered_videos,
                             stats=stats)
    except Exception as e:
        flash(f'Error loading subscriptions: {str(e)}', 'error')
        return redirect('/')

@app.route('/add_subscription', methods=['POST'])
def add_subscription():
    """Add a new YouTube channel subscription"""
    if not subscription_manager:
        return jsonify({'status': 'error', 'message': 'Subscription manager not available'})
    
    try:
        data = request.get_json()
        channel_url = data.get('channel_url', '').strip()
        auto_process = data.get('auto_process', False)
        
        if not channel_url:
            return jsonify({'status': 'error', 'message': 'Channel URL is required'})
        
        result = subscription_manager.add_subscription(channel_url, auto_process)
        
        if result['success']:
            return jsonify({
                'status': 'success',
                'message': f"Subscription added for {result['channel_name']}! Discovered {result['videos_discovered']} videos."
            })
        else:
            return jsonify({
                'status': 'error', 
                'message': result['error']
            })
            
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/refresh_subscriptions', methods=['POST'])
def refresh_subscriptions():
    """Refresh all subscriptions to find new videos"""
    if not subscription_manager:
        return jsonify({'status': 'error', 'message': 'Subscription manager not available'})
    
    try:
        result = subscription_manager.refresh_all_subscriptions()
        
        if result['success']:
            return jsonify({
                'status': 'success',
                'message': f"Refreshed {result['refreshed_channels']} channels. Found {result['new_videos']} new videos."
            })
        else:
            return jsonify({
                'status': 'error',
                'message': result['error']
            })
            
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/delete_subscription/<int:subscription_id>', methods=['DELETE'])
def delete_subscription(subscription_id):
    """Delete a channel subscription"""
    if not subscription_manager:
        return jsonify({'status': 'error', 'message': 'Subscription manager not available'})
    
    try:
        if subscription_manager.delete_subscription(subscription_id):
            return jsonify({'status': 'success', 'message': 'Subscription deleted successfully'})
        else:
            return jsonify({'status': 'error', 'message': 'Failed to delete subscription'})
            
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/process_discovered_video/<int:video_id>', methods=['POST'])
def process_discovered_video(video_id):
    """Process a discovered video from subscriptions"""
    if not subscription_manager:
        return jsonify({'status': 'error', 'message': 'Subscription manager not available'})
    
    try:
        # Get discovered video details
        discovered_videos = subscription_manager.get_discovered_videos()
        video = next((v for v in discovered_videos if v['id'] == video_id), None)
        
        if not video:
            return jsonify({'status': 'error', 'message': 'Video not found'})
        
        # Redirect to new summary creation with pre-filled URL
        return jsonify({
            'status': 'redirect',
            'url': f"/new?url={video['url']}"
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

# ==============================
# SETTINGS MANAGEMENT ROUTES
# ==============================

@app.route('/settings')
def settings_dashboard():
    """Settings dashboard for API keys and configuration"""
    if not settings_manager:
        flash('Settings manager is not available', 'error')
        return redirect('/')
    
    try:
        # Get settings by category
        api_keys = settings_manager.get_settings_by_category('api_keys')
        database_settings = settings_manager.get_settings_by_category('database')
        app_settings = settings_manager.get_settings_by_category('application')
        subscription_settings = settings_manager.get_settings_by_category('subscriptions')
        vector_search_settings = settings_manager.get_settings_by_category('vector_search')
        
        return render_template('settings.html',
                             api_keys=api_keys,
                             database_settings=database_settings,
                             app_settings=app_settings,
                             subscription_settings=subscription_settings,
                             vector_search_settings=vector_search_settings)
    except Exception as e:
        flash(f'Error loading settings: {str(e)}', 'error')
        return redirect('/')

@app.route('/update_setting', methods=['POST'])
def update_setting():
    """Update a setting value"""
    if not settings_manager:
        return jsonify({'status': 'error', 'message': 'Settings manager not available'})
    
    try:
        data = request.get_json()
        setting_key = data.get('key')
        setting_value = data.get('value')
        setting_type = data.get('type', 'text')
        is_sensitive = data.get('sensitive', False)
        category = data.get('category', 'general')
        
        if not setting_key:
            return jsonify({'status': 'error', 'message': 'Setting key is required'})
        
        # Validate API key if it's an API key setting
        if category == 'api_keys' and setting_value:
            provider = setting_key.replace('_api_key', '')
            validation = settings_manager.validate_api_key(provider, setting_value)
            if not validation['valid']:
                return jsonify({
                    'status': 'warning',
                    'message': f"API key saved but validation failed: {validation['message']}"
                })
        
        # Update the setting
        success = settings_manager.set_setting(
            key=setting_key,
            value=setting_value,
            setting_type=setting_type,
            is_sensitive=is_sensitive,
            category=category
        )
        
        if success:
            # For API keys, also update environment variable for current session
            if category == 'api_keys':
                os.environ[setting_key.upper()] = setting_value or ''
                
            return jsonify({
                'status': 'success',
                'message': 'Setting updated successfully'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to update setting'
            })
            
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/validate_api_key', methods=['POST'])
def validate_api_key():
    """Validate an API key"""
    if not settings_manager:
        return jsonify({'status': 'error', 'message': 'Settings manager not available'})
    
    try:
        data = request.get_json()
        provider = data.get('provider')
        api_key = data.get('api_key')
        
        if not provider or not api_key:
            return jsonify({'status': 'error', 'message': 'Provider and API key are required'})
        
        result = settings_manager.validate_api_key(provider, api_key)
        return jsonify({
            'status': 'success' if result['valid'] else 'error',
            'valid': result['valid'],
            'message': result['message']
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/export_settings')
def export_settings():
    """Export settings as JSON for backup"""
    if not settings_manager:
        return jsonify({'error': 'Settings manager not available'}), 500
    
    try:
        # Export non-sensitive settings only by default
        include_sensitive = request.args.get('include_sensitive', 'false').lower() == 'true'
        settings_data = settings_manager.export_settings(include_sensitive)
        
        return jsonify(settings_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ==============================
# AUTOMATION MANAGEMENT ROUTES
# ==============================

@app.route('/automation')
def automation_dashboard():
    """Automation management dashboard"""
    if not automation_scheduler:
        flash('Automation scheduler is not available', 'error')
        return redirect('/')
    
    try:
        # Get automation status
        status = automation_scheduler.get_job_status()
        
        return render_template('automation.html', status=status)
    except Exception as e:
        flash(f'Error loading automation dashboard: {str(e)}', 'error')
        return redirect('/')

@app.route('/automation/status')
def get_automation_status():
    """Get current automation status via API"""
    if not automation_scheduler:
        return jsonify({'status': 'error', 'message': 'Automation scheduler not available'})
    
    try:
        status = automation_scheduler.get_job_status()
        return jsonify({'status': 'success', 'data': status})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/automation/pause', methods=['POST'])
def pause_automation():
    """Pause the automation scheduler"""
    if not automation_scheduler:
        return jsonify({'status': 'error', 'message': 'Automation scheduler not available'})
    
    try:
        automation_scheduler.pause_automation()
        return jsonify({'status': 'success', 'message': 'Automation paused'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/automation/resume', methods=['POST'])
def resume_automation():
    """Resume the automation scheduler"""
    if not automation_scheduler:
        return jsonify({'status': 'error', 'message': 'Automation scheduler not available'})
    
    try:
        automation_scheduler.resume_automation()
        return jsonify({'status': 'success', 'message': 'Automation resumed'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/automation/force_check', methods=['POST'])
def force_subscription_check():
    """Force immediate subscription check"""
    if not automation_scheduler:
        return jsonify({'status': 'error', 'message': 'Automation scheduler not available'})
    
    try:
        success = automation_scheduler.force_subscription_check()
        if success:
            return jsonify({'status': 'success', 'message': 'Subscription check completed'})
        else:
            return jsonify({'status': 'error', 'message': 'Subscription check failed'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/automation/jobs')
def get_processing_jobs():
    """Get processing jobs with pagination"""
    if not automation_scheduler:
        return jsonify({'status': 'error', 'message': 'Automation scheduler not available'})
    
    try:
        # Get job queue information
        status = automation_scheduler.get_job_status()
        return jsonify({
            'status': 'success',
            'data': {
                'active_jobs': status.get('active_jobs', []),
                'status_counts': status.get('status_counts', {}),
                'recent_activity': status.get('recent_activity', [])
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    # Check for required environment variables
    if not os.getenv('OPENAI_API_KEY'):
        print("Warning: OPENAI_API_KEY not set in environment variables")
        print("Please create a .env file with your OpenAI API key")
    
    # Run the app
    debug_mode = True  # Enable debug mode to avoid template caching
    port = int(os.getenv('PORT', 5000))  # Back to default port 5000
    
    print(f"Starting YouTube Summarizer Dashboard on http://localhost:{port}")
    print(f"[DEBUG] Actually using port: {port}")
    app.run(debug=debug_mode, host='0.0.0.0', port=port)