"""
Caption parser and normalizer
Handles VTT/SRT parsing and normalization to CaptionChunk format
"""
import re
from typing import List
from .captions_types import CaptionChunk

TIMECODE_SRT = re.compile(r"(\d+):(\d{2}):(\d{2}),(\d{3})")
TIMECODE_VTT = re.compile(r"(\d+):(\d{2}):(\d{2})\.(\d{3})")

def parse_srt(s: str, lang: str, source: str) -> List[CaptionChunk]:
    """
    Robust SRT parser
    Expected format: idx\nHH:MM:SS,mmm --> HH:MM:SS,mmm\ntext\n\n
    """
    chunks: List[CaptionChunk] = []
    blocks = re.split(r"\r?\n\r?\n", s.strip(), flags=re.M)
    
    for b in blocks:
        lines = [ln for ln in b.splitlines() if ln.strip()]
        if len(lines) < 2:
            continue
        
        # Second line is timecode
        tc = lines[1]
        try:
            start_str, end_str = [t.strip() for t in tc.split("-->")]
            start_ms = _srt_to_ms(start_str)
            end_ms = _srt_to_ms(end_str)
        except Exception:
            continue
        
        text = " ".join(lines[2:]).strip()
        if text:
            chunks.append(CaptionChunk(start_ms, end_ms, text, lang, source))  # type: ignore
    
    return chunks

def parse_vtt(s: str, lang: str, source: str) -> List[CaptionChunk]:
    """
    Simple VTT parser - ignores styles/WEBVTT header
    """
    s = re.sub(r"^WEBVTT.*?\n+", "", s, flags=re.S|re.I)
    return _parse_vtt_like_blocks(s, lang, source)

def _parse_vtt_like_blocks(s: str, lang: str, source: str) -> List[CaptionChunk]:
    chunks: List[CaptionChunk] = []
    blocks = re.split(r"\r?\n\r?\n", s.strip(), flags=re.M)
    
    for b in blocks:
        lines = [ln for ln in b.splitlines() if ln.strip()]
        
        # Find first line with timecode
        tc_line = None
        for ln in lines:
            if "-->" in ln:
                tc_line = ln
                break
        if not tc_line:
            continue
        
        try:
            start_str, end_str = [t.strip() for t in tc_line.split("-->")]
            start_ms = _vtt_to_ms(start_str)
            end_ms = _vtt_to_ms(end_str)
        except Exception:
            continue
        
        # Remaining lines are text
        text_lines = []
        take = False
        for ln in lines:
            if take:
                text_lines.append(ln)
            if ln is tc_line:
                take = True
        
        text = " ".join(text_lines).strip()
        if text:
            chunks.append(CaptionChunk(start_ms, end_ms, text, lang, source))  # type: ignore
    
    return chunks

def _srt_to_ms(tc: str) -> int:
    """Convert SRT timecode to milliseconds"""
    m = TIMECODE_SRT.search(tc)
    if not m: 
        raise ValueError("bad srt timecode")
    h, m_, s, ms = map(int, m.groups())
    return ((h*60 + m_)*60 + s)*1000 + ms

def _vtt_to_ms(tc: str) -> int:
    """Convert VTT timecode to milliseconds"""
    m = TIMECODE_VTT.search(tc)
    if not m: 
        raise ValueError("bad vtt timecode")
    h, m_, s, ms = map(int, m.groups())
    return ((h*60 + m_)*60 + s)*1000 + ms