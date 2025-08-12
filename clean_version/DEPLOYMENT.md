# Deployment Guide - YouTube Summarizer

## Quick Start (5 minutes)

### 1. Clone & Install
```bash
git clone <your-repo>
cd youtube-summarizer
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env with your API keys (at minimum: OPENAI_API_KEY, SUPABASE_URL, SUPABASE_KEY)
```

### 3. Initialize & Run
```bash
python init_system.py
python app_clean.py
```

Visit `http://localhost:5000` - You're ready to go! 🚀

## Architecture Overview

### Core Files (Essential)
```
youtube-summarizer/
├── app_clean.py              # Clean Flask app (recommended)
├── database_clean.py         # Simplified database layer  
├── enhanced_summarizer.py    # Multi-provider AI summarization
├── chatbot_qa.py            # Q&A and semantic search
├── video_highlight_extractor.py # Video highlights
├── srt_chaptering.py        # SRT chapter creation
├── init_system.py           # System initialization
├── requirements.txt         # Dependencies
└── templates/               # Web interface
    ├── index.html          # Dashboard
    ├── new_summary.html    # Video processing
    ├── chatbot.html       # Q&A interface
    ├── highlights.html    # Video highlights
    └── srt_chaptering.html # SRT processing
```

### Optional Files (Legacy)
```
├── app.py                   # Full-featured app (more complex)
├── database.py             # Original database layer
├── automation_scheduler.py # Background jobs
├── channel_subscriptions.py # RSS monitoring
└── settings_manager.py     # User preferences
```

## Simplified vs Full Version

### Use `app_clean.py` if you want:
- ✅ Simple, maintainable codebase
- ✅ Core features only (summarization, Q&A, highlights, SRT)
- ✅ Easy to understand and modify
- ✅ Faster startup time
- ✅ Fewer dependencies

### Use `app.py` if you need:
- 🔧 Background automation and scheduling
- 🔧 Channel subscription monitoring
- 🔧 Advanced settings management
- 🔧 Complex routing and middleware
- 🔧 Full feature set

## Environment Variables

### Required (Minimum)
```env
OPENAI_API_KEY=sk-...        # For embeddings and GPT models
SUPABASE_URL=https://...     # Database (or use SQLite)
SUPABASE_KEY=eyJ...          # Database access
```

### Optional AI Providers
```env
CLAUDE_API_KEY=sk-ant-...    # Anthropic Claude
GEMINI_API_KEY=AI...         # Google Gemini
XAI_API_KEY=xai-...          # Grok (X.AI)
PERPLEXITY_API_KEY=pplx-...  # Perplexity
DEEPSEEK_API_KEY=sk-...      # DeepSeek
```

### Optional Features
```env
PORT=5000                    # Application port
DEBUG=true                   # Development mode
SECRET_KEY=your-secret       # Session security
```

## Database Options

### Option 1: Supabase (Recommended)
- ✅ Vector search with pgvector
- ✅ Real-time subscriptions  
- ✅ Cloud hosting
- ✅ Web dashboard
- ✅ Automatic backups

### Option 2: SQLite (Local)
- ✅ No setup required
- ✅ Local file storage
- ✅ Good for development
- ❌ No vector search
- ❌ Single-user only

## Feature Overview

### 🎥 Video Summarization
- Multiple AI providers (OpenAI, Claude, Gemini, Grok, etc.)
- Custom prompts and templates
- Key points extraction
- Transcript processing
- Multiple output formats

### 🤖 Q&A Chatbot  
- Semantic search across all videos
- Context-aware responses
- Source attribution
- Real-time search suggestions

### ⚡ Video Highlights
- AI-powered moment detection
- Automatic clip extraction
- Compilation generation
- Download individual clips

### 📝 SRT Chaptering
- Upload subtitle files
- AI-generated chapter markers
- Multiple export formats (YouTube, JSON, Premiere Pro)
- Content creator tools

## Production Deployment

### Using Docker
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN python init_system.py

EXPOSE 5000
CMD ["python", "app_clean.py"]
```

### Using Railway/Render/Heroku
1. Connect your GitHub repository
2. Set environment variables in the dashboard
3. Deploy automatically on push

### Manual Server
```bash
# Install dependencies
pip install -r requirements.txt gunicorn

# Initialize system  
python init_system.py

# Run with Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app_clean:app
```

## Performance Tips

### 1. Use Supabase for Vector Search
- Much faster semantic search
- Better scalability
- Built-in caching

### 2. Configure AI Provider Limits
```python
# In enhanced_summarizer.py
rate_limits = {
    'openai': 60,      # requests per minute
    'claude': 30,
    'gemini': 15
}
```

### 3. Optimize Video Processing
- Use transcript-only mode for highlights
- Cache expensive operations
- Process videos in background

### 4. Enable Caching
```env
REDIS_URL=redis://localhost:6379  # Optional Redis caching
```

## Troubleshooting

### Common Issues

**"No module named 'X'"**
```bash
pip install -r requirements.txt
```

**"Database connection failed"**
- Check Supabase credentials in `.env`
- Verify network connectivity
- Application will fallback to SQLite automatically

**"FFmpeg not found"**
```bash
# Windows (with Chocolatey)
choco install ffmpeg

# Mac (with Homebrew)  
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg
```

**"API key invalid"**
- Verify API keys in `.env`
- Check quota limits
- Test with curl commands

### Performance Issues

**Slow video processing:**
- Use transcript-only mode
- Reduce video quality for highlights
- Process in background

**High memory usage:**
- Limit concurrent video processing
- Clear temp files regularly
- Use streaming for large videos

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review logs in the console output
3. Test with `init_system.py` to verify setup
4. Create an issue in the repository

---

## Quick Migration from Complex Version

If you're currently using the full `app.py`, here's how to migrate:

1. **Backup your data:**
   ```bash
   cp summaries.db summaries_backup.db
   ```

2. **Test the clean version:**
   ```bash
   python app_clean.py
   ```

3. **Verify functionality:**
   - Video summarization works
   - Q&A chatbot responds  
   - Highlights extract properly
   - SRT chaptering functions

4. **Switch when ready:**
   ```bash
   mv app.py app_full.py
   mv app_clean.py app.py
   ```

The clean version maintains all core functionality while being much simpler to understand and maintain.