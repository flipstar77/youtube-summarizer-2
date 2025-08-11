import yt_dlp
import re
from urllib.parse import urlparse, parse_qs


class YouTubeMetadata:
    def __init__(self):
        # Configure yt-dlp to be quiet and only extract info
        self.ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'skip_download': True,
            'extract_flat': False,
        }
    
    def extract_video_id(self, youtube_url):
        """Extract video ID from YouTube URL"""
        parsed_url = urlparse(youtube_url)
        
        if parsed_url.hostname == 'youtu.be':
            return parsed_url.path[1:]
        elif parsed_url.hostname in ('www.youtube.com', 'youtube.com'):
            if parsed_url.path == '/watch':
                return parse_qs(parsed_url.query)['v'][0]
            elif parsed_url.path[:7] == '/embed/':
                return parsed_url.path.split('/')[2]
            elif parsed_url.path[:3] == '/v/':
                return parsed_url.path.split('/')[2]
        
        raise ValueError("Invalid YouTube URL")
    
    def get_video_info(self, youtube_url):
        """Get video metadata using yt-dlp"""
        try:
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                # Extract video info
                info = ydl.extract_info(youtube_url, download=False)
                
                # Clean title - remove special characters that might cause issues
                title = info.get('title', 'Unknown Title')
                title = re.sub(r'[^\w\s\-_\.\(\)\[\]]+', '', title)
                
                # Get uploader/channel name
                uploader = info.get('uploader', info.get('channel', 'Unknown Channel'))
                uploader = re.sub(r'[^\w\s\-_\.]+', '', uploader)
                
                # Get other useful metadata
                duration = info.get('duration', 0)  # Duration in seconds
                upload_date = info.get('upload_date', '')  # YYYYMMDD format
                view_count = info.get('view_count', 0)
                
                # Format upload date
                formatted_date = ''
                if upload_date and len(upload_date) == 8:
                    try:
                        year = upload_date[:4]
                        month = upload_date[4:6]
                        day = upload_date[6:8]
                        formatted_date = f"{year}-{month}-{day}"
                    except:
                        formatted_date = upload_date
                
                # Format duration
                formatted_duration = self._format_duration(duration)
                
                return {
                    'video_id': self.extract_video_id(youtube_url),
                    'title': title,
                    'uploader': uploader,
                    'duration': duration,
                    'duration_formatted': formatted_duration,
                    'upload_date': formatted_date,
                    'view_count': view_count,
                    'thumbnail_url': f"https://img.youtube.com/vi/{self.extract_video_id(youtube_url)}/mqdefault.jpg"
                }
                
        except Exception as e:
            # Fallback to basic info if yt-dlp fails
            video_id = self.extract_video_id(youtube_url)
            return {
                'video_id': video_id,
                'title': f'YouTube Video {video_id}',
                'uploader': 'Unknown Channel',
                'duration': 0,
                'duration_formatted': 'Unknown',
                'upload_date': '',
                'view_count': 0,
                'thumbnail_url': f"https://img.youtube.com/vi/{video_id}/mqdefault.jpg",
                'error': str(e)
            }
    
    def _format_duration(self, seconds):
        """Format duration from seconds to HH:MM:SS or MM:SS"""
        if not seconds or seconds <= 0:
            return "Unknown"
        
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        
        if hours > 0:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes}:{seconds:02d}"