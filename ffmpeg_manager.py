"""
FFmpeg Operations Manager

This module provides a comprehensive wrapper for FFmpeg operations specifically
designed for video highlight extraction, processing, and compilation. It handles
video trimming, concatenation, subtitle embedding, quality optimization, and
various video transformations needed for highlight creation.
"""

import os
import re
import glob
import subprocess
import tempfile
import json
import shutil
from typing import List, Dict, Optional, Tuple, Union, Callable
from pathlib import Path
from dataclasses import dataclass
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VideoInfo:
    """Video information container"""
    duration: float
    width: int
    height: int
    fps: float
    codec: str
    bitrate: int
    file_size: int
    format: str
    
    @classmethod
    def from_ffprobe_output(cls, output: Dict) -> 'VideoInfo':
        """Create VideoInfo from ffprobe JSON output"""
        video_stream = next(
            (stream for stream in output.get('streams', []) 
             if stream.get('codec_type') == 'video'), {}
        )
        format_info = output.get('format', {})
        
        # Parse FPS (can be in different formats)
        fps_str = video_stream.get('r_frame_rate', '0/1')
        if '/' in fps_str:
            num, den = map(int, fps_str.split('/'))
            fps = num / den if den != 0 else 0
        else:
            fps = float(fps_str) if fps_str else 0
        
        return cls(
            duration=float(format_info.get('duration', 0)),
            width=int(video_stream.get('width', 0)),
            height=int(video_stream.get('height', 0)),
            fps=fps,
            codec=video_stream.get('codec_name', ''),
            bitrate=int(format_info.get('bit_rate', 0)),
            file_size=int(format_info.get('size', 0)),
            format=format_info.get('format_name', '')
        )


@dataclass
class FFmpegCommand:
    """FFmpeg command configuration"""
    input_files: List[str]
    output_file: str
    options: List[str]
    filters: List[str]
    progress_callback: Optional[Callable] = None
    
    def build_command(self) -> List[str]:
        """Build the complete FFmpeg command"""
        cmd = ['ffmpeg']
        
        # Add input files
        for input_file in self.input_files:
            cmd.extend(['-i', input_file])
        
        # Add filters if any
        if self.filters:
            filter_complex = ';'.join(self.filters)
            cmd.extend(['-filter_complex', filter_complex])
        
        # Add options
        cmd.extend(self.options)
        
        # Add output file
        cmd.append(self.output_file)
        
        return cmd


class FFmpegManager:
    """Comprehensive FFmpeg operations manager for video highlight extraction"""
    
    def __init__(self, ffmpeg_path: Optional[str] = None, 
                 temp_dir: Optional[str] = None,
                 quality_preset: str = "medium"):
        """
        Initialize FFmpeg manager
        
        Args:
            ffmpeg_path: Path to FFmpeg executable (None for system PATH)
            temp_dir: Temporary directory for intermediate files
            quality_preset: Quality preset for video processing
        """
        self.ffmpeg_path = ffmpeg_path or 'ffmpeg'
        self.ffprobe_path = ffmpeg_path.replace('ffmpeg', 'ffprobe') if ffmpeg_path else 'ffprobe'
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.quality_preset = quality_preset
        
        # Ensure temp directory exists
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Verify FFmpeg is available
        self._verify_ffmpeg()
    
    def _verify_ffmpeg(self):
        """Verify FFmpeg is available and working"""
        try:
            result = subprocess.run([self.ffmpeg_path, '-version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                raise RuntimeError("FFmpeg not working properly")
            logger.info("FFmpeg verified and ready")
        except (subprocess.TimeoutExpired, FileNotFoundError, RuntimeError):
            raise RuntimeError(f"FFmpeg not found at {self.ffmpeg_path}. Please install FFmpeg or specify correct path.")
    
    def get_video_info(self, video_path: str) -> VideoInfo:
        """
        Get comprehensive video information using ffprobe
        
        Args:
            video_path: Path to video file
            
        Returns:
            VideoInfo object with video details
        """
        cmd = [
            self.ffprobe_path,
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            video_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            probe_data = json.loads(result.stdout)
            return VideoInfo.from_ffprobe_output(probe_data)
        except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as e:
            raise RuntimeError(f"Failed to get video info for {video_path}: {str(e)}")
    
    def extract_clip(self, input_path: str, output_path: str, 
                    start_time: float, end_time: float,
                    with_subtitles: bool = False,
                    subtitle_file: Optional[str] = None,
                    progress_callback: Optional[Callable] = None) -> bool:
        """
        Extract a video clip from a larger video
        
        Args:
            input_path: Path to source video
            output_path: Path for output clip
            start_time: Start time in seconds
            end_time: End time in seconds
            with_subtitles: Whether to burn subtitles into video
            subtitle_file: Path to subtitle file (SRT)
            progress_callback: Optional callback for progress updates
            
        Returns:
            True if successful, False otherwise
        """
        duration = end_time - start_time
        
        # Build FFmpeg command
        cmd = [
            self.ffmpeg_path,
            '-ss', str(start_time),
            '-i', input_path,
            '-t', str(duration),
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-preset', self.quality_preset,
            '-avoid_negative_ts', 'make_zero'
        ]
        
        # Add subtitle burning if requested
        if with_subtitles and subtitle_file and os.path.exists(subtitle_file):
            # Escape the subtitle file path for FFmpeg filter
            escaped_subtitle_path = subtitle_file.replace('\\', '\\\\').replace(':', '\\:')
            subtitle_filter = f"subtitles={escaped_subtitle_path}"
            cmd.extend(['-vf', subtitle_filter])
        
        cmd.extend(['-y', output_path])  # Overwrite output file
        
        return self._execute_command(cmd, progress_callback, duration)
    
    def extract_multiple_clips(self, input_path: str, clips: List[Dict],
                              output_dir: str, name_prefix: str = "highlight",
                              with_subtitles: bool = False,
                              subtitle_file: Optional[str] = None,
                              progress_callback: Optional[Callable] = None) -> List[str]:
        """
        Extract multiple clips from a video efficiently
        
        Args:
            input_path: Path to source video
            clips: List of clip dictionaries with 'start', 'end', and optional 'name'
            output_dir: Directory for output clips
            name_prefix: Prefix for output filenames
            with_subtitles: Whether to burn subtitles
            subtitle_file: Path to subtitle file
            progress_callback: Progress callback
            
        Returns:
            List of paths to created clip files
        """
        os.makedirs(output_dir, exist_ok=True)
        output_files = []
        
        total_clips = len(clips)
        logger.info(f"Extracting {total_clips} clips from {input_path}")
        
        for i, clip in enumerate(clips):
            start_time = clip['start']
            end_time = clip['end']
            clip_name = clip.get('name', f"{name_prefix}_{i+1:03d}")
            
            # Sanitize filename
            safe_name = self._sanitize_filename(clip_name)
            output_path = os.path.join(output_dir, f"{safe_name}.mp4")
            
            # Extract clip
            success = self.extract_clip(
                input_path, output_path, start_time, end_time,
                with_subtitles, subtitle_file
            )
            
            if success:
                output_files.append(output_path)
                logger.info(f"Extracted clip {i+1}/{total_clips}: {safe_name}")
            else:
                logger.error(f"Failed to extract clip {i+1}/{total_clips}: {safe_name}")
            
            # Update progress
            if progress_callback:
                progress = (i + 1) / total_clips
                progress_callback(f"Extracted {i+1}/{total_clips} clips", progress)
        
        return output_files
    
    def concatenate_videos(self, video_paths: List[str], output_path: str,
                          with_transitions: bool = False,
                          transition_duration: float = 0.5,
                          add_titles: bool = False,
                          title_duration: float = 2.0,
                          progress_callback: Optional[Callable] = None) -> bool:
        """
        Concatenate multiple video clips into one compilation
        
        Args:
            video_paths: List of video file paths to concatenate
            output_path: Path for final compilation video
            with_transitions: Add smooth transitions between clips
            transition_duration: Duration of transitions in seconds
            add_titles: Add title cards between clips
            title_duration: Duration of title cards
            progress_callback: Progress callback
            
        Returns:
            True if successful
        """
        if not video_paths:
            raise ValueError("No video paths provided")
        
        # Create temporary concat file
        concat_file = os.path.join(self.temp_dir, 'concat_list.txt')
        
        try:
            # Method 1: Simple concatenation without transitions
            if not with_transitions and not add_titles:
                return self._simple_concatenate(video_paths, output_path, concat_file, progress_callback)
            
            # Method 2: Complex concatenation with effects
            return self._complex_concatenate(video_paths, output_path, with_transitions, 
                                           transition_duration, add_titles, title_duration, progress_callback)
        
        finally:
            # Cleanup
            if os.path.exists(concat_file):
                os.remove(concat_file)
    
    def _simple_concatenate(self, video_paths: List[str], output_path: str, 
                           concat_file: str, progress_callback: Optional[Callable] = None) -> bool:
        """Simple concatenation using concat demuxer"""
        # Write concat file
        with open(concat_file, 'w', encoding='utf-8') as f:
            for video_path in video_paths:
                # Escape path for concat file
                escaped_path = video_path.replace('\\', '/').replace("'", "\\'")
                f.write(f"file '{escaped_path}'\n")
        
        # Build command
        cmd = [
            self.ffmpeg_path,
            '-f', 'concat',
            '-safe', '0',
            '-i', concat_file,
            '-c', 'copy',  # Copy streams without re-encoding for speed
            '-y', output_path
        ]
        
        total_duration = sum(self.get_video_info(path).duration for path in video_paths)
        return self._execute_command(cmd, progress_callback, total_duration)
    
    def _complex_concatenate(self, video_paths: List[str], output_path: str,
                           with_transitions: bool, transition_duration: float,
                           add_titles: bool, title_duration: float,
                           progress_callback: Optional[Callable] = None) -> bool:
        """Complex concatenation with transitions and effects"""
        
        # Build filter complex for transitions
        filters = []
        input_labels = [f"[{i}:v][{i}:a]" for i in range(len(video_paths))]
        
        if with_transitions:
            # Add crossfade transitions between videos
            video_chain = input_labels[0]
            audio_chain = f"[{0}:a]"
            
            for i in range(1, len(video_paths)):
                video_chain = f"{video_chain}[{i}:v]xfade=transition=fade:duration={transition_duration}:offset=0[v{i}]"
                audio_chain = f"{audio_chain}[{i}:a]acrossfade=d={transition_duration}[a{i}]"
            
            filters.extend([video_chain, audio_chain])
        
        # Build command
        cmd = [self.ffmpeg_path]
        
        # Add input files
        for video_path in video_paths:
            cmd.extend(['-i', video_path])
        
        # Add filter complex if needed
        if filters:
            cmd.extend(['-filter_complex', ';'.join(filters)])
        
        # Output options
        cmd.extend([
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-preset', self.quality_preset,
            '-y', output_path
        ])
        
        total_duration = sum(self.get_video_info(path).duration for path in video_paths)
        return self._execute_command(cmd, progress_callback, total_duration)
    
    def add_subtitle_overlay(self, input_path: str, output_path: str,
                           subtitle_file: str, style: Optional[Dict] = None,
                           progress_callback: Optional[Callable] = None) -> bool:
        """
        Add subtitle overlay to video
        
        Args:
            input_path: Input video path
            output_path: Output video path
            subtitle_file: SRT subtitle file path
            style: Subtitle styling options
            progress_callback: Progress callback
            
        Returns:
            True if successful
        """
        if not os.path.exists(subtitle_file):
            raise FileNotFoundError(f"Subtitle file not found: {subtitle_file}")
        
        # Default subtitle style
        default_style = {
            'fontsize': 24,
            'fontcolor': 'white',
            'bordercolor': 'black',
            'borderw': 2,
            'fontname': 'Arial'
        }
        
        if style:
            default_style.update(style)
        
        # Build subtitle filter
        escaped_subtitle_path = subtitle_file.replace('\\', '\\\\').replace(':', '\\:')
        subtitle_filter = f"subtitles={escaped_subtitle_path}"
        
        # Add style options
        style_options = []
        for key, value in default_style.items():
            if key in ['fontsize', 'borderw']:
                style_options.append(f"{key}={value}")
            else:
                style_options.append(f"{key}={value}")
        
        if style_options:
            subtitle_filter += f":{':'.join(style_options)}"
        
        # Build command
        cmd = [
            self.ffmpeg_path,
            '-i', input_path,
            '-vf', subtitle_filter,
            '-c:a', 'copy',  # Copy audio without re-encoding
            '-y', output_path
        ]
        
        video_info = self.get_video_info(input_path)
        return self._execute_command(cmd, progress_callback, video_info.duration)
    
    def create_highlight_compilation(self, clips_data: List[Dict], output_path: str,
                                   intro_text: Optional[str] = None,
                                   outro_text: Optional[str] = None,
                                   background_music: Optional[str] = None,
                                   progress_callback: Optional[Callable] = None) -> bool:
        """
        Create a complete highlight compilation with intro, outro, and music
        
        Args:
            clips_data: List of clip dictionaries with paths and metadata
            output_path: Final compilation output path
            intro_text: Optional intro text to display
            outro_text: Optional outro text to display
            background_music: Optional background music file
            progress_callback: Progress callback
            
        Returns:
            True if successful
        """
        temp_files = []
        
        try:
            video_segments = []
            
            # Add intro if specified
            if intro_text:
                intro_file = self._create_text_clip(intro_text, 3.0, "intro")
                video_segments.append(intro_file)
                temp_files.append(intro_file)
            
            # Add all highlight clips
            for clip_data in clips_data:
                if os.path.exists(clip_data['path']):
                    video_segments.append(clip_data['path'])
            
            # Add outro if specified
            if outro_text:
                outro_file = self._create_text_clip(outro_text, 3.0, "outro")
                video_segments.append(outro_file)
                temp_files.append(outro_file)
            
            # Concatenate all segments
            temp_compilation = os.path.join(self.temp_dir, 'temp_compilation.mp4')
            temp_files.append(temp_compilation)
            
            success = self.concatenate_videos(video_segments, temp_compilation, 
                                           with_transitions=True, progress_callback=progress_callback)
            
            if not success:
                return False
            
            # Add background music if specified
            if background_music and os.path.exists(background_music):
                return self._add_background_music(temp_compilation, output_path, background_music, progress_callback)
            else:
                # Just move the temp file to final location
                shutil.move(temp_compilation, output_path)
                return True
        
        finally:
            # Cleanup temp files
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except OSError:
                        pass
    
    def _create_text_clip(self, text: str, duration: float, clip_type: str) -> str:
        """Create a text overlay clip for intro/outro"""
        output_file = os.path.join(self.temp_dir, f"{clip_type}_{int(duration)}.mp4")
        
        # Create a simple colored background with text
        cmd = [
            self.ffmpeg_path,
            '-f', 'lavfi',
            '-i', f'color=c=black:size=1920x1080:duration={duration}',
            '-vf', f'drawtext=text=\'{text}\':fontcolor=white:fontsize=48:x=(w-text_w)/2:y=(h-text_h)/2',
            '-c:v', 'libx264',
            '-t', str(duration),
            '-y', output_file
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return output_file
        else:
            raise RuntimeError(f"Failed to create text clip: {result.stderr}")
    
    def _add_background_music(self, video_path: str, output_path: str, music_path: str,
                            progress_callback: Optional[Callable] = None) -> bool:
        """Add background music to video"""
        video_info = self.get_video_info(video_path)
        
        cmd = [
            self.ffmpeg_path,
            '-i', video_path,
            '-i', music_path,
            '-filter_complex', '[0:a][1:a]amix=inputs=2:duration=first:dropout_transition=3[a]',
            '-map', '0:v:0',
            '-map', '[a]',
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-y', output_path
        ]
        
        return self._execute_command(cmd, progress_callback, video_info.duration)
    
    def optimize_video_size(self, input_path: str, output_path: str,
                           target_size_mb: Optional[int] = None,
                           max_width: int = 1920,
                           progress_callback: Optional[Callable] = None) -> bool:
        """
        Optimize video for smaller file size while maintaining quality
        
        Args:
            input_path: Input video path
            output_path: Output video path
            target_size_mb: Target file size in MB (optional)
            max_width: Maximum video width
            progress_callback: Progress callback
            
        Returns:
            True if successful
        """
        video_info = self.get_video_info(input_path)
        
        # Calculate optimal bitrate if target size is specified
        if target_size_mb:
            target_size_bits = target_size_mb * 8 * 1024 * 1024
            target_bitrate = int(target_size_bits / video_info.duration * 0.8)  # 80% for video, 20% for audio
        else:
            target_bitrate = None
        
        # Build command
        cmd = [
            self.ffmpeg_path,
            '-i', input_path,
            '-c:v', 'libx264',
            '-preset', 'slow',  # Better compression
            '-crf', '23',  # Good quality/size balance
        ]
        
        # Add bitrate constraint if specified
        if target_bitrate:
            cmd.extend(['-b:v', f'{target_bitrate}k', '-maxrate', f'{target_bitrate * 1.5}k', '-bufsize', f'{target_bitrate * 2}k'])
        
        # Scale video if too large
        if video_info.width > max_width:
            scale_filter = f'scale={max_width}:-2'
            cmd.extend(['-vf', scale_filter])
        
        cmd.extend([
            '-c:a', 'aac',
            '-b:a', '128k',
            '-y', output_path
        ])
        
        return self._execute_command(cmd, progress_callback, video_info.duration)
    
    def _execute_command(self, cmd: List[str], progress_callback: Optional[Callable] = None, 
                        total_duration: Optional[float] = None) -> bool:
        """
        Execute FFmpeg command with progress tracking
        
        Args:
            cmd: FFmpeg command as list
            progress_callback: Optional progress callback
            total_duration: Total expected duration for progress calculation
            
        Returns:
            True if command executed successfully
        """
        logger.info(f"Executing FFmpeg command: {' '.join(cmd[:5])}...")
        
        try:
            if progress_callback and total_duration:
                # Run with progress tracking
                return self._execute_with_progress(cmd, progress_callback, total_duration)
            else:
                # Simple execution
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                return True
        
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg command failed: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Error executing FFmpeg command: {str(e)}")
            return False
    
    def _execute_with_progress(self, cmd: List[str], progress_callback: Callable, 
                              total_duration: float) -> bool:
        """Execute command with progress tracking"""
        # Add progress reporting to command
        progress_cmd = cmd + ['-progress', 'pipe:1']
        
        process = subprocess.Popen(
            progress_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            universal_newlines=True
        )
        
        try:
            current_time = 0.0
            
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                
                if output.startswith('out_time='):
                    time_str = output.split('=')[1].strip()
                    try:
                        # Parse time format (HH:MM:SS.microseconds)
                        if ':' in time_str:
                            parts = time_str.split(':')
                            if len(parts) == 3:
                                hours = int(parts[0])
                                minutes = int(parts[1])
                                seconds = float(parts[2])
                                current_time = hours * 3600 + minutes * 60 + seconds
                        else:
                            current_time = float(time_str)
                        
                        # Calculate progress
                        if total_duration > 0:
                            progress = min(current_time / total_duration, 1.0)
                            progress_callback(f"Processing: {current_time:.1f}s/{total_duration:.1f}s", progress)
                    
                    except (ValueError, IndexError):
                        continue
            
            # Wait for process to complete
            process.wait()
            
            if process.returncode == 0:
                progress_callback("Processing complete", 1.0)
                return True
            else:
                stderr_output = process.stderr.read()
                logger.error(f"FFmpeg failed: {stderr_output}")
                return False
        
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for file system"""
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        
        # Remove multiple underscores and limit length
        filename = re.sub(r'_+', '_', filename)
        return filename[:100] if len(filename) > 100 else filename
    
    def cleanup_temp_files(self):
        """Clean up temporary files created during processing"""
        try:
            temp_pattern = os.path.join(self.temp_dir, 'ffmpeg_*')
            for temp_file in glob.glob(temp_pattern):
                try:
                    os.remove(temp_file)
                except OSError:
                    pass
            logger.info("Cleaned up temporary files")
        except Exception as e:
            logger.warning(f"Could not clean up temp files: {str(e)}")


# Utility functions
def seconds_to_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS.mmm timestamp format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def timestamp_to_seconds(timestamp: str) -> float:
    """Convert HH:MM:SS.mmm timestamp to seconds"""
    parts = timestamp.split(':')
    if len(parts) == 3:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds
    return 0.0


if __name__ == "__main__":
    # Example usage
    ffmpeg = FFmpegManager()
    
    # Test video info
    test_video = "D:\\mcp\\static\\downloads\\Supabase Postgres Vector DB Crash Course_20250810_145558.mp4"
    if os.path.exists(test_video):
        info = ffmpeg.get_video_info(test_video)
        print(f"Video Info: {info.duration}s, {info.width}x{info.height}, {info.fps}fps")
        
        # Test clip extraction
        output_clip = "D:\\mcp\\static\\downloads\\test_clip.mp4"
        success = ffmpeg.extract_clip(test_video, output_clip, 10, 30)
        print(f"Clip extraction: {'Success' if success else 'Failed'}")
    else:
        print("Test video not found")