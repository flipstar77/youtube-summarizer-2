# YouTube Summarizer

A comprehensive Python application that downloads YouTube video transcripts and generates AI-powered summaries. Available as both a command-line tool and a web dashboard.

## Features

- 📺 Extract transcripts from YouTube videos
- 🤖 Generate different types of summaries (brief, detailed, bullet-point) 
- 🌍 Support for multiple languages
- 💾 Save summaries to database and text files
- 🖥️ Beautiful web dashboard interface
- 📱 Responsive design for mobile devices
- 🔍 Search and manage your summaries
- 📊 Dashboard with statistics
- 🎯 REST API endpoints

## Components

### 1. Command Line Interface
Use `youtube_summarizer.py` for quick command-line summaries.

### 2. Web Dashboard
Use `app.py` to run the Flask web application with a full dashboard interface.

## Setup

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Set up environment variables:**
```bash
cp .env.example .env
```
Edit `.env` and add your OpenAI API key and other settings:
```
OPENAI_API_KEY=your_openai_api_key_here
SECRET_KEY=your_flask_secret_key_here
FLASK_DEBUG=False
PORT=5000
```

## Usage

### Web Dashboard (Recommended)

1. **Start the web server:**
```bash
python app.py
```

2. **Open your browser to:**
```
http://localhost:5000
```

3. **Features include:**
   - Create new summaries by pasting YouTube URLs
   - View all your summaries in a organized dashboard
   - Download summaries as text files
   - Delete old summaries
   - Real-time progress tracking
   - Mobile-friendly interface

### Command Line Interface

**Basic usage:**
```bash
python youtube_summarizer.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

**With options:**
```bash
python youtube_summarizer.py "https://www.youtube.com/watch?v=VIDEO_ID" --type brief --language en
```

#### CLI Options:
- `--type`: Summary type (`brief`, `detailed`, `bullet`) - default: `detailed`
- `--language`: Transcript language preference - default: `en`
- `--api-key`: OpenAI API key (if not set in .env file)

## API Endpoints

The web application also provides REST API endpoints:

- `GET /api/summaries` - Get all summaries
- `GET /api/summary/<id>` - Get specific summary
- `DELETE /delete/<id>` - Delete summary

## Project Structure

```
youtube-summarizer/
├── app.py                 # Flask web application
├── youtube_summarizer.py  # Command-line interface
├── transcript_extractor.py # YouTube transcript extraction
├── summarizer.py          # AI summarization logic
├── database.py           # SQLite database operations
├── requirements.txt      # Python dependencies
├── .env.example         # Environment variables template
├── templates/           # HTML templates
│   ├── base.html
│   ├── index.html
│   ├── new_summary.html
│   ├── view_summary.html
│   ├── 404.html
│   └── 500.html
└── static/             # CSS and assets
    └── style.css
```

## Screenshots & Features

### Dashboard
- View all your summaries with video thumbnails
- Statistics showing total summaries and recent activity
- Quick access to create new summaries

### Summary Creation
- Simple form to paste YouTube URLs
- Choose summary type and language
- Real-time progress tracking during processing

### Summary Viewing
- Full summary display with video information
- Copy to clipboard functionality
- Download as text file
- Direct links to original videos

## Requirements

- Python 3.7+
- OpenAI API key
- Internet connection for transcript extraction and API calls
- Modern web browser (for dashboard)

## Troubleshooting

1. **"No transcript found"**: Some videos don't have transcripts available
2. **OpenAI API errors**: Check your API key and account credits
3. **Port already in use**: Change the PORT in your .env file

## Contributing

Feel free to submit issues and enhancement requests!