# YouTube Video Summarizer & Content Creator Tools

A modern Flask-based application that provides AI-powered video summarization, content analysis, and creator tools.

## Features

### Core Features
- **Video Summarization**: AI-powered summaries using multiple providers (OpenAI, Claude, Gemini, Grok, etc.)
- **Smart Q&A Chatbot**: Ask questions about your video collection with semantic search
- **Video Highlights**: Automatic extraction of key moments and clips
- **SRT Chaptering**: Upload SRT files and create intelligent chapter markers
- **Channel Subscriptions**: Monitor YouTube channels for new content

### AI Providers Supported
- OpenAI GPT (3.5/4/4o)
- Anthropic Claude
- Google Gemini
- Grok (X.AI)
- Perplexity
- DeepSeek

## Quick Start

### Prerequisites
- Python 3.8+
- FFmpeg (for video processing)
- API keys for desired AI providers

### Installation

1. Clone and setup:
```bash
git clone <your-repo>
cd youtube-summarizer
pip install -r requirements.txt
```

2. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

3. Initialize database:
```bash
python init_system.py
```

4. Run the application:
```bash
python app.py
```

Visit `http://localhost:5000` to access the web interface.

## Configuration

### Required Environment Variables
```
OPENAI_API_KEY=your_openai_key
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key

# Optional AI Providers
CLAUDE_API_KEY=your_claude_key
GEMINI_API_KEY=your_gemini_key
XAI_API_KEY=your_grok_key
PERPLEXITY_API_KEY=your_perplexity_key
DEEPSEEK_API_KEY=your_deepseek_key
```

### Database Options
- **Supabase** (recommended): Full vector search and cloud storage
- **SQLite** (local): Basic functionality, good for testing

## Architecture

### Core Components
- **app.py**: Main Flask application and routes
- **enhanced_summarizer.py**: Multi-provider AI summarization
- **chatbot_qa.py**: Semantic search and Q&A system  
- **video_highlight_extractor.py**: Video analysis and clip extraction
- **srt_chaptering.py**: SRT processing and chapter generation
- **database.py**: Database abstraction layer

### Directory Structure
```
/
├── app.py                      # Main Flask app
├── enhanced_summarizer.py      # AI summarization
├── chatbot_qa.py              # Q&A system
├── video_highlight_extractor.py # Video highlights
├── srt_chaptering.py          # SRT chaptering
├── database.py                # Database layer
├── templates/                 # Web interface
├── static/                    # Assets
└── temp/                      # Temporary files
```

## Usage

### Video Summarization
1. Go to "New Summary" in the navigation
2. Enter a YouTube URL
3. Select AI provider and model
4. Configure summary options
5. Process and view results

### Video Q&A
1. Navigate to "Q&A Chat"
2. Ask questions about your video collection
3. Get AI-powered answers with source references

### Video Highlights
1. Go to "Video Highlights"  
2. Select a video to analyze
3. AI identifies key moments
4. Download individual clips or compilation

### SRT Chaptering
1. Visit "SRT Chapters"
2. Upload your SRT subtitle file
3. Configure chapter settings
4. Get chapters in multiple formats (YouTube, JSON, etc.)

## API Endpoints

### Core Routes
- `GET /` - Dashboard
- `GET /new` - New summary form
- `POST /summarize` - Process video
- `GET /chatbot` - Q&A interface
- `GET /highlights` - Video highlights
- `GET /srt-chaptering` - SRT upload

### API Endpoints
- `POST /api/ask_question` - Q&A queries
- `POST /process_srt` - SRT processing
- `GET /api/summaries` - Get summaries
- `POST /semantic_search` - Vector search

## Development

### Project Focus
This is a streamlined version focusing on core functionality:
- Clean, maintainable codebase
- Modern UI with Bootstrap 5
- Multi-provider AI integration
- Efficient vector search
- Content creator tools

### Key Technologies
- **Backend**: Flask, Python
- **Frontend**: Bootstrap 5, JavaScript
- **Database**: Supabase (PostgreSQL) or SQLite
- **AI**: OpenAI, Claude, Gemini, Grok APIs
- **Video**: FFmpeg, yt-dlp
- **Search**: Vector embeddings, semantic search

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

MIT License - see LICENSE file for details.