import re
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound


class TranscriptExtractor:
    def __init__(self):
        pass
    
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
    
    def get_transcript(self, youtube_url, language='en'):
        """Get transcript from YouTube video"""
        try:
            video_id = self.extract_video_id(youtube_url)
            
            # Create API instance and get transcript
            api = YouTubeTranscriptApi()
            transcript_list = api.fetch(video_id)
            
            # Combine all transcript segments into one text
            # The new API returns FetchedTranscriptSnippet objects
            full_text = ' '.join([item.text for item in transcript_list])
            
            # Clean up the text
            full_text = re.sub(r'\[.*?\]', '', full_text)  # Remove bracketed content
            full_text = re.sub(r'\s+', ' ', full_text).strip()  # Clean whitespace
            
            return {
                'video_id': video_id,
                'transcript': full_text,
                'segments': transcript_list
            }
            
        except TranscriptsDisabled:
            raise Exception("Transcripts are disabled for this video")
        except NoTranscriptFound:
            raise Exception("No transcript found for this video")
        except Exception as e:
            raise Exception(f"Error extracting transcript: {str(e)}")