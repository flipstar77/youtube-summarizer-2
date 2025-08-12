# YouTube Summarizer

A comprehensive Python application that downloads YouTube video transcripts and generates AI-powered summaries with semantic search capabilities. Features automated subscription processing, highlight extraction, and a modern web dashboard.

## ✨ Features

- 📺 **Video Processing**: Extract transcripts from YouTube videos
- 🤖 **AI Summaries**: Generate different types of summaries (brief, detailed, bullet-point, tutorial, professional)
- 🔍 **Semantic Search**: Vector-based similarity search using OpenAI embeddings
- 🎬 **Highlight Extraction**: Automatic video highlight detection and compilation
- 📡 **Subscription Automation**: Monitor RSS feeds and auto-process new videos
- 🎵 **Text-to-Speech**: Generate audio versions with ElevenLabs
- 🖥️ **Web Dashboard**: Beautiful, responsive interface
- 📊 **Analytics**: Statistics and insights dashboard
- 🔄 **Background Jobs**: Automated processing with retry logic

## 🏗️ Architecture

**Primary Database**: Supabase (PostgreSQL + pgvector)
- Vector similarity search with HNSW indexing
- Real-time capabilities
- Hosted and scalable

**Fallback Database**: SQLite (development only)
- Local development when Supabase unavailable
- Limited functionality (no vector search)

## 🚀 Quick Setup

### Option A: Automated Setup (Recommended)
```bash
# Run the setup wizard
python setup.py

# Verify your setup
python init.py

# Start the application
python app.py
```

### Option B: Manual Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Database Setup (Supabase)

1. **Create Supabase Project**: 
   - Go to [supabase.com](https://supabase.com)
   - Create new project
   - Note your project URL and service role key

2. **Apply Database Schema**:
   ```sql
   -- In Supabase SQL Editor, run:
   \i sql/supabase_vector_search.sql
   ```
   This creates tables, vector indexes, and the 2 search functions.

3. **Test Setup**:
   ```bash
   python tests/test_vector_rpc.py
   ```

### Vector Quick Check
Once your schema is applied, verify vector search works:

**Option A: Python Smoke Tests**
```bash
# Set environment variables (Windows PowerShell example)
$env:SUPABASE_URL="https://your-project.supabase.co"
$env:SUPABASE_ANON_KEY="eyJ..."

# Run smoke tests
python tests/test_vector_rpc.py

# Or with pytest (if installed)
pytest -q -k vector_rpc
```

**Option B: SQL Health Check**
Run `sql/health_check.sql` in Supabase SQL Editor - each section should return "ok" or meaningful data.

### Living Checkpoint System
The system includes automated health monitoring:
```bash
# Full system verification (run anytime)
python init.py

# Guided setup for new installations  
python setup.py

# Continuous health monitoring
python init_system.py --monitor
```

### 3. Environment Configuration

Create `.env` file:
```env
# Primary Database (Supabase)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
USE_SUPABASE=true

# AI Services
OPENAI_API_KEY=your_openai_api_key
ELEVENLABS_API_KEY=your_elevenlabs_key

# Optional AI Providers
CLAUDE_API_KEY=your_claude_key
PERPLEXITY_API_KEY=your_perplexity_key
DEEPSEEK_API_KEY=your_deepseek_key
GEMINI_API_KEY=your_gemini_key

# Flask
SECRET_KEY=your_secret_key
FLASK_DEBUG=false
PORT=5000
```

### 4. Start the Application
```bash
python app.py
```

Open: http://localhost:5000

## 🔧 Key Components

### Core Modules
- `app.py` - Flask web application with automation
- `supabase_client.py` - Primary database interface
- `database.py` - SQLite fallback (dev only)
- `enhanced_summarizer.py` - Multi-provider AI summaries
- `vector_embeddings.py` - Semantic search with OpenAI embeddings
- `automation_scheduler.py` - Background job processing

### Database Schema
- **Golden File**: `sql/supabase_vector_search.sql` (single source of truth)
- **Archived Files**: `sql/archive/` (old versions)
- **Vector Functions**: `search_summaries_by_similarity`, `find_similar_summaries`

### Testing
- **Smoke Tests**: `tests/test_vector_rpc.py`
- **CI Pipeline**: `.github/workflows/ci.yml`
- **Linting**: ruff + flake8

## 📋 Usage Guide

### Web Dashboard
1. **Create Summaries**: Paste YouTube URLs, select type and AI provider
2. **Semantic Search**: Find similar content using vector embeddings  
3. **Manage Subscriptions**: Add YouTube channels for automatic processing
4. **Extract Highlights**: Generate video clips of key moments
5. **Automation Control**: Monitor and control background processing

### API Endpoints
- `GET /api/summaries` - List all summaries
- `POST /semantic_search` - Vector similarity search
- `GET /similar/<id>` - Find similar summaries
- `POST /extract_highlights/<id>` - Generate video highlights
- `GET /automation/status` - Check automation status

### Automation Features
- **RSS Monitoring**: Checks subscribed channels every 30 minutes
- **Smart Filtering**: Processing rules based on keywords, duration, etc.
- **Job Queue**: Priority-based processing with retry logic
- **Error Handling**: Exponential backoff and logging

## 🛠️ Development

### Running Tests
```bash
# Vector RPC smoke tests
python tests/test_vector_rpc.py

# Full test suite (if available)
pytest

# Linting
ruff check .
flake8 .
```

### Database Management
```bash
# Apply schema updates
psql -h your-host -U postgres -d your-db -f sql/supabase_vector_search.sql

# Check vector functions
python -c "
from supabase_client import SupabaseDatabase
db = SupabaseDatabase()
result = db.client.rpc('search_summaries_by_similarity', {...})
print(result.data)
"
```

### Architecture Decisions
- **Supabase Primary**: Centralized truth, vector search, real-time
- **Pinned Dependencies**: Stable versions prevent drift
- **Single SQL File**: Eliminates type mismatches
- **Smoke Tests**: Catch RPC issues before production
- **CI Pipeline**: Automated quality checks

## 📁 Project Structure

```
youtube-summarizer/
├── sql/
│   ├── supabase_vector_search.sql  # Golden database schema
│   └── archive/                    # Old SQL files
├── app.py                         # Main Flask application  
├── automation_scheduler.py        # Background job processing
├── enhanced_summarizer.py         # Multi-provider AI summaries
├── vector_embeddings.py           # Semantic search
├── supabase_client.py             # Primary database (Supabase)
├── database.py                    # Fallback database (SQLite dev only)
├── requirements.txt               # Pinned dependencies
├── tests/
│   └── test_vector_rpc.py        # Vector function smoke tests
├── .github/workflows/
│   └── ci.yml                    # CI pipeline
├── templates/                     # HTML templates
└── static/                       # Assets
```

## ⚠️ Important Notes

- **Large Media Files**: Not stored in repo (temp/, outputs/ excluded)
- **API Keys**: Never commit secrets (use .env, test with CI)
- **Vector Search**: Requires Supabase + pgvector extension
- **Background Jobs**: Uses APScheduler for automation
- **Dependencies**: Pinned versions for stability

## 🐛 Troubleshooting

### Vector Search Issues
```bash
# Test RPC functions
python tests/test_vector_rpc.py

# Check if pgvector extension enabled
# In Supabase SQL Editor: SELECT * FROM pg_extension WHERE extname = 'vector';
```

### Database Connection
```bash
# Test Supabase connection
python -c "
from supabase_client import SupabaseDatabase
db = SupabaseDatabase()
print('✅ Connected to Supabase')
"
```

### Common Issues
1. **"Structure mismatch"**: Run smoke tests, check function signatures
2. **Missing embeddings**: Vectors generated on summary creation
3. **Automation not working**: Check APScheduler logs in app output
4. **Large files error**: Ensure temp/ and outputs/ in .gitignore

## 🤝 Contributing

1. **Fork & Branch**: Create feature branches from `main`
2. **Test**: Run smoke tests and linting
3. **CI**: All checks must pass
4. **Schema Changes**: Update `sql/supabase_vector_search.sql` only
5. **Dependencies**: Pin new versions in requirements.txt

## 📄 License

MIT License - See LICENSE file for details.