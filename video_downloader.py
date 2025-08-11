import os
import yt_dlp
import tempfile
from datetime import datetime

class VideoDownloader:
    def __init__(self):
        self.download_dir = os.path.join('static', 'downloads')
        os.makedirs(self.download_dir, exist_ok=True)
    
    def get_video_info(self, url):
        """Get video information without downloading"""
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info = ydl.extract_info(url, download=False)
                return {
                    'title': info.get('title', 'Unknown'),
                    'duration': info.get('duration', 0),
                    'view_count': info.get('view_count', 0),
                    'uploader': info.get('uploader', 'Unknown'),
                    'formats': self._extract_format_info(info.get('formats', [])),
                    'video_id': info.get('id', ''),
                    'thumbnail': info.get('thumbnail', ''),
                }
            except Exception as e:
                raise Exception(f"Failed to get video info: {str(e)}")
    
    def _extract_format_info(self, formats):
        """Extract useful format information"""
        format_info = []
        seen_qualities = set()
        
        for f in formats:
            if f.get('vcodec') != 'none' and f.get('acodec') != 'none':  # Has both video and audio
                quality = f.get('height', 0)
                filesize = f.get('filesize', 0) or f.get('filesize_approx', 0)
                
                if quality and quality not in seen_qualities:
                    format_info.append({
                        'format_id': f.get('format_id'),
                        'quality': f"{quality}p",
                        'ext': f.get('ext', 'mp4'),
                        'filesize': filesize,
                        'filesize_mb': round(filesize / (1024 * 1024), 1) if filesize else 'Unknown',
                        'fps': f.get('fps'),
                        'vcodec': f.get('vcodec', '').split('.')[0],
                        'acodec': f.get('acodec', '').split('.')[0],
                    })
                    seen_qualities.add(quality)
        
        # Sort by quality (highest first)
        format_info.sort(key=lambda x: int(x['quality'].replace('p', '')), reverse=True)
        return format_info[:6]  # Limit to top 6 qualities
    
    def download_video(self, url, quality='best', format_type='mp4', with_subtitles=False, srt_content=None):
        """Download video with specified options"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get video info first
        info = self.get_video_info(url)
        safe_title = self._sanitize_filename(info['title'])
        
        output_filename = f"{safe_title}_{timestamp}.%(ext)s"
        output_path = os.path.join(self.download_dir, output_filename)
        
        # Configure yt-dlp options
        ydl_opts = {
            'outtmpl': output_path,
            'format': self._build_format_selector(quality, format_type),
            'writesubtitles': with_subtitles,
            'writeautomaticsub': with_subtitles,
            'subtitleslangs': ['en', 'de', 'fr', 'es'],
            'postprocessors': []
        }
        
        # Add FFmpeg post-processor for format conversion if needed
        if format_type != 'best':
            ydl_opts['postprocessors'].append({
                'key': 'FFmpegVideoConvertor',
                'preferedformat': format_type,
            })
        
        # Add subtitle post-processor if custom SRT is provided
        if srt_content:
            srt_filename = f"{safe_title}_{timestamp}.srt"
            srt_path = os.path.join(self.download_dir, srt_filename)
            with open(srt_path, 'w', encoding='utf-8') as f:
                f.write(srt_content)
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
                
                # Find the downloaded file
                for file in os.listdir(self.download_dir):
                    if file.startswith(safe_title) and file.endswith(f"_{timestamp}.{format_type}"):
                        return os.path.join(self.download_dir, file)
                
                # Fallback: find any file with the timestamp
                for file in os.listdir(self.download_dir):
                    if timestamp in file and not file.endswith('.srt'):
                        return os.path.join(self.download_dir, file)
                
                raise Exception("Downloaded file not found")
                
        except Exception as e:
            raise Exception(f"Download failed: {str(e)}")
    
    def _build_format_selector(self, quality, format_type):
        """Build format selector string for yt-dlp"""
        if quality == 'best':
            return f'best[ext={format_type}]/best'
        elif quality == 'worst':
            return f'worst[ext={format_type}]/worst'
        else:
            # Extract height from quality (e.g., '720p' -> '720')
            height = quality.replace('p', '')
            return f'best[height<={height}][ext={format_type}]/best[height<={height}]/best'
    
    def _sanitize_filename(self, filename):
        """Sanitize filename for file system"""
        # Remove or replace invalid characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        
        # Limit length
        return filename[:50] if len(filename) > 50 else filename
    
    def get_download_formats(self):
        """Get available download formats"""
        return [
            {'value': 'mp4', 'label': 'MP4 (Most Compatible)', 'description': 'Best for general use'},
            {'value': 'webm', 'label': 'WebM (Smaller Size)', 'description': 'Good compression, modern browsers'},
            {'value': 'mkv', 'label': 'MKV (High Quality)', 'description': 'Best quality, larger files'},
            {'value': 'avi', 'label': 'AVI (Legacy)', 'description': 'Older format, wide compatibility'},
        ]
    
    def get_quality_options(self):
        """Get available quality options"""
        return [
            {'value': 'best', 'label': 'Best Available', 'description': 'Highest quality available'},
            {'value': '2160p', 'label': '4K (2160p)', 'description': '4K Ultra HD'},
            {'value': '1440p', 'label': '2K (1440p)', 'description': '2K Quad HD'},
            {'value': '1080p', 'label': 'Full HD (1080p)', 'description': 'Full HD quality'},
            {'value': '720p', 'label': 'HD (720p)', 'description': 'Standard HD'},
            {'value': '480p', 'label': 'SD (480p)', 'description': 'Standard definition'},
            {'value': '360p', 'label': 'Low (360p)', 'description': 'Low quality, small size'},
            {'value': 'worst', 'label': 'Smallest File', 'description': 'Minimum file size'},
        ]
    
    def cleanup_old_downloads(self, max_age_hours=24):
        """Clean up old download files"""
        current_time = datetime.now()
        removed_count = 0
        
        for filename in os.listdir(self.download_dir):
            file_path = os.path.join(self.download_dir, filename)
            if os.path.isfile(file_path):
                # Check file age
                file_time = datetime.fromtimestamp(os.path.getctime(file_path))
                age_hours = (current_time - file_time).total_seconds() / 3600
                
                if age_hours > max_age_hours:
                    try:
                        os.remove(file_path)
                        removed_count += 1
                    except OSError:
                        pass  # Ignore errors when removing files
        
        return removed_count