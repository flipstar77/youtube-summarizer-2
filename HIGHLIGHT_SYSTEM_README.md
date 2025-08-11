# Video Highlight Extraction System

A comprehensive FFmpeg-based video highlight extraction system that automatically identifies and extracts the most engaging segments from YouTube videos using AI-powered analysis.

## 🌟 Features

### Core Functionality
- **YouTube Video Download**: Download videos in various qualities using yt-dlp
- **Transcript Processing**: Extract and parse SRT subtitle files with precise timing
- **AI-Powered Analysis**: Use OpenAI API to score segments for highlight potential
- **Video Processing**: Extract clips and create compilations using FFmpeg
- **Subtitle Integration**: Burn subtitles into highlight clips
- **Smooth Transitions**: Create professional compilations with transitions

### Advanced Features
- **Batch Processing**: Process multiple videos simultaneously
- **Custom Configurations**: Flexible settings for different use cases
- **Progress Tracking**: Real-time feedback during processing
- **Quality Optimization**: Balance file size and video quality
- **Comprehensive Reporting**: Detailed analysis reports in JSON format
- **Error Handling**: Graceful fallback when services are unavailable

## 📋 Requirements

### System Requirements
- Python 3.7 or higher
- FFmpeg installed and accessible in PATH
- Internet connection for YouTube downloads and AI analysis

### Python Dependencies
```
yt-dlp>=2023.12.30
youtube-transcript-api>=0.6.2
openai>=1.3.0
python-dotenv>=1.0.0
flask>=2.3.0
elevenlabs>=0.2.0
requests>=2.25.0
supabase>=2.3.0
postgrest>=0.16.0
ffmpeg-python>=0.2.0
moviepy>=1.0.3
```

### External Dependencies
- **FFmpeg**: For video processing operations
- **OpenAI API Key**: For advanced highlight analysis (optional - fallback available)

## 🚀 Installation

1. **Clone or download the system files**:
   ```bash
   # Files should be in your project directory:
   # - video_highlight_extractor.py
   # - srt_parser.py
   # - highlight_analyzer.py
   # - ffmpeg_manager.py
   # - requirements.txt
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install FFmpeg**:
   - Windows: Download from https://ffmpeg.org/download.html
   - macOS: `brew install ffmpeg`
   - Linux: `apt-get install ffmpeg` or `yum install ffmpeg`

4. **Set up OpenAI API key** (optional):
   ```bash
   # Create .env file
   echo "OPENAI_API_KEY=your_api_key_here" > .env
   ```

## 📚 Quick Start

### Basic Usage

```python
from video_highlight_extractor import VideoHighlightExtractor

# Initialize with default settings
extractor = VideoHighlightExtractor()

# Extract highlights from a YouTube video
result = extractor.extract_highlights("https://youtube.com/watch?v=example")

if result.success:
    print(f"✅ Extracted {result.highlights_found} highlights")
    print(f"📁 Individual clips: {len(result.individual_clips)}")
    print(f"🎬 Compilation: {result.compilation_path}")
    print(f"📊 Report: {result.analysis_report_path}")
else:
    print(f"❌ Failed: {result.error_message}")
```

### Custom Configuration

```python
from video_highlight_extractor import create_extraction_config, VideoHighlightExtractor

# Create custom configuration
config = create_extraction_config(
    max_highlights=8,                    # Maximum number of highlights
    min_highlight_score=7.0,            # Minimum score threshold
    max_total_duration=300,             # Max 5 minutes of highlights
    video_quality="1080p",              # Video quality preference
    with_subtitles=True,                # Burn subtitles into clips
    with_transitions=True,              # Smooth transitions in compilation
    create_individual_clips=True,       # Create separate clip files
    create_compilation=True             # Create highlight compilation
)

# Initialize with custom config and progress callback
def progress_callback(message, progress):
    print(f"[{progress*100:5.1f}%] {message}")

extractor = VideoHighlightExtractor(
    config=config,
    openai_api_key="your_api_key",     # Optional
    progress_callback=progress_callback
)

result = extractor.extract_highlights("https://youtube.com/watch?v=example")
```

### Batch Processing

```python
# Process multiple videos
urls = [
    "https://youtube.com/watch?v=video1",
    "https://youtube.com/watch?v=video2",
    "https://youtube.com/watch?v=video3"
]

results = extractor.batch_extract_highlights(urls)

for result in results:
    if result.success:
        print(f"✅ {result.video_title}: {result.highlights_found} highlights")
    else:
        print(f"❌ {result.video_title}: {result.error_message}")
```

## 🔧 System Architecture

### Core Components

1. **`video_highlight_extractor.py`** - Main orchestrator
   - Coordinates all system components
   - Manages the complete extraction pipeline
   - Handles configuration and error management

2. **`srt_parser.py`** - Subtitle processing
   - Parses SRT files with precise timing
   - Creates analysis windows for highlight detection
   - Handles subtitle format conversions

3. **`highlight_analyzer.py`** - AI-powered analysis
   - Uses OpenAI API to score content segments
   - Evaluates engagement, information value, uniqueness
   - Provides fallback heuristic scoring

4. **`ffmpeg_manager.py`** - Video processing
   - Handles all FFmpeg operations
   - Extracts clips with precise timing
   - Creates compilations with transitions
   - Optimizes video quality and file sizes

### Integration Points

The system integrates seamlessly with the existing YouTube Summarizer codebase:

- **`video_downloader.py`** - Uses existing download functionality
- **`transcript_extractor.py`** - Leverages existing transcript extraction
- **Database integration** - Can store results in existing database
- **Web interface** - Can be integrated into Flask application

## ⚙️ Configuration Options

### ExtractionConfig Parameters

```python
config = ExtractionConfig(
    # Analysis settings
    window_size_seconds=30,          # Analysis window size
    window_overlap_seconds=5,        # Window overlap for continuity
    min_highlight_score=6.0,         # Minimum score to consider
    max_highlights=10,               # Maximum highlights to extract
    max_total_duration=300.0,        # Maximum total duration (seconds)
    
    # Video processing settings
    video_quality="720p",            # Output video quality
    output_format="mp4",             # Output video format
    with_subtitles=True,            # Include subtitles in clips
    with_transitions=True,           # Add transitions to compilation
    transition_duration=0.5,         # Transition length (seconds)
    
    # File organization
    output_base_dir="highlights",    # Base output directory
    create_individual_clips=True,    # Create individual clip files
    create_compilation=True,         # Create compilation video
    preserve_temp_files=False,       # Keep temporary files
    
    # AI analysis settings
    openai_model="gpt-3.5-turbo",   # OpenAI model to use
    highlight_criteria=None          # Custom highlight criteria
)
```

### Highlight Criteria

```python
highlight_criteria = {
    'engagement_weight': 0.3,        # Weight for engagement scoring
    'information_weight': 0.3,       # Weight for information value
    'uniqueness_weight': 0.2,        # Weight for content uniqueness
    'emotional_weight': 0.2,         # Weight for emotional impact
    'min_duration': 10,              # Minimum clip duration (seconds)
    'max_duration': 120,             # Maximum clip duration (seconds)
    'preferred_topics': [            # Topics to prioritize
        'key_insights', 
        'actionable_tips', 
        'interesting_facts'
    ],
    'avoid_topics': [                # Topics to avoid
        'filler_content', 
        'repetitive_information'
    ]
}
```

## 📊 Output Structure

The system creates the following output structure:

```
highlights/
├── video_{video_id}/
│   ├── clips/
│   │   ├── highlight_01_8.5.mp4     # Individual clips with scores
│   │   ├── highlight_02_8.1.mp4
│   │   └── highlight_03_7.9.mp4
│   ├── highlights_compilation_20250811_143022.mp4  # Final compilation
│   ├── analysis_report_20250811_143022.json        # Detailed analysis
│   └── transcript_video_id.srt                     # Original subtitles
└── extraction_stats.json                           # Overall statistics
```

### Analysis Report Format

```json
{
  "video_info": {
    "title": "Amazing Python Tutorial",
    "uploader": "TechChannel",
    "duration": 1200,
    "view_count": 50000
  },
  "extraction_config": {
    "max_highlights": 10,
    "min_highlight_score": 6.0,
    "video_quality": "720p"
  },
  "analysis_summary": {
    "total_segments_analyzed": 45,
    "highlights_selected": 8,
    "total_highlight_duration": 240.5,
    "average_highlight_score": 7.8,
    "extraction_timestamp": "2025-08-11T14:30:22"
  },
  "selected_highlights": [
    {
      "segment_id": "12",
      "start_time": 145.5,
      "end_time": 175.2,
      "overall_score": 8.5,
      "engagement_score": 9.0,
      "information_score": 8.2,
      "reasoning": "High-value technical content with engaging delivery",
      "key_topics": ["python", "optimization", "performance"],
      "recommended_action": "extract"
    }
  ],
  "statistics": {
    "score_distribution": {
      "extract_count": 8,
      "maybe_count": 15,
      "skip_count": 22
    },
    "common_topics": [
      ["python", 12],
      ["tutorial", 8],
      ["programming", 6]
    ]
  }
}
```

## 🧪 Testing

### Run Tests

```bash
# Simple test suite (works without FFmpeg)
python simple_test.py

# Full test suite (requires FFmpeg)
python test_highlight_extraction.py

# Interactive demonstration
python demo_highlight_system.py
```

### Test Results Interpretation

- **Import Test**: Verifies all modules load correctly
- **SRT Parser Test**: Tests subtitle parsing functionality
- **FFmpeg Test**: Checks FFmpeg installation and video processing
- **Integration Test**: Tests complete system integration
- **YouTube Test**: Tests actual video processing (requires API key)

## 🐛 Troubleshooting

### Common Issues

1. **FFmpeg not found**:
   ```
   Error: FFmpeg not found at ffmpeg
   Solution: Install FFmpeg and ensure it's in your PATH
   ```

2. **OpenAI API errors**:
   ```
   Error: Invalid API key
   Solution: Set valid OPENAI_API_KEY in environment or use fallback scoring
   ```

3. **YouTube download fails**:
   ```
   Error: Video unavailable
   Solution: Check video URL and ensure it's publicly accessible
   ```

4. **Memory issues with large videos**:
   ```
   Solution: Use lower video quality or process shorter segments
   ```

### Performance Optimization

- **Use lower video quality** for faster processing
- **Reduce max_highlights** to limit processing time
- **Enable preserve_temp_files=False** to save disk space
- **Use smaller window_size_seconds** for more precise analysis

## 🔌 Integration Examples

### Flask Web Application

```python
from flask import Flask, request, jsonify
from video_highlight_extractor import VideoHighlightExtractor

app = Flask(__name__)
extractor = VideoHighlightExtractor()

@app.route('/api/extract_highlights', methods=['POST'])
def extract_highlights():
    data = request.json
    url = data.get('url')
    
    if not url:
        return jsonify({'error': 'URL required'}), 400
    
    try:
        result = extractor.extract_highlights(url)
        
        return jsonify({
            'success': result.success,
            'video_title': result.video_title,
            'highlights_found': result.highlights_found,
            'compilation_path': result.compilation_path,
            'individual_clips': result.individual_clips,
            'processing_time': result.processing_time,
            'error_message': result.error_message
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
```

### Celery Background Tasks

```python
from celery import Celery
from video_highlight_extractor import VideoHighlightExtractor

celery = Celery('highlight_extractor')
extractor = VideoHighlightExtractor()

@celery.task(bind=True)
def extract_highlights_task(self, url):
    def progress_callback(message, progress):
        self.update_state(
            state='PROGRESS',
            meta={'message': message, 'progress': progress}
        )
    
    extractor.progress_callback = progress_callback
    result = extractor.extract_highlights(url)
    
    return {
        'success': result.success,
        'video_title': result.video_title,
        'highlights_found': result.highlights_found,
        'compilation_path': result.compilation_path
    }
```

## 📈 Performance Metrics

### Typical Processing Times

- **Short video (5-10 min)**: 2-5 minutes processing
- **Medium video (10-30 min)**: 5-15 minutes processing
- **Long video (30+ min)**: 15-45 minutes processing

### Resource Usage

- **CPU**: Intensive during AI analysis and video processing
- **Memory**: 500MB-2GB depending on video length and quality
- **Disk**: Temporary files can use 2-5x original video size
- **Network**: Downloads original video + API calls for analysis

## 🛠️ Development

### Adding Custom Analyzers

```python
from highlight_analyzer import HighlightAnalyzer

class CustomHighlightAnalyzer(HighlightAnalyzer):
    def _calculate_custom_score(self, text):
        # Your custom scoring logic
        return score
```

### Extending FFmpeg Operations

```python
from ffmpeg_manager import FFmpegManager

class CustomFFmpegManager(FFmpegManager):
    def add_custom_filter(self, input_path, output_path, filter_params):
        # Your custom FFmpeg operations
        pass
```

## 📄 License

This system is designed to integrate with the existing YouTube Summarizer codebase and follows the same licensing terms.

## 🤝 Support

For support and questions:

1. Check the troubleshooting section above
2. Run the test suite to identify issues
3. Review the demonstration script for usage examples
4. Check system logs in `highlight_extraction.log`

## 🔄 Updates and Maintenance

### Regular Maintenance Tasks

- Update yt-dlp regularly: `pip install --upgrade yt-dlp`
- Monitor OpenAI API usage and costs
- Clean up old extraction files periodically
- Update FFmpeg for latest features and security fixes

### System Monitoring

```python
# Get extraction statistics
stats = extractor.get_extraction_stats()
print(f"Videos processed: {stats['videos_processed']}")
print(f"Total highlights: {stats['total_highlights_extracted']}")
print(f"Success rate: {stats['success_rate']}%")
```

---

**Note**: This system is designed to work with publicly available YouTube content and respects YouTube's terms of service. Ensure you have appropriate permissions before processing copyrighted content.