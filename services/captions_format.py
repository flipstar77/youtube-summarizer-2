"""
SRT formatter with proper split/merge, timing and readability rules
"""
from typing import List
from .captions_types import CaptionChunk

# Parameters (adjustable)
MAX_LINE_CHARS   = 44         # ~42–50 gut lesbar
MAX_LINES        = 2
MIN_DURATION_MS  = 900        # <1s -> mergen
MAX_DURATION_MS  = 4000       # >4s -> splitten
MAX_CPS          = 20         # chars per second

def format_chunks_for_srt(chunks: List[CaptionChunk]) -> List[CaptionChunk]:
    """
    Normalize chunks for SRT:
    - merge if too short
    - split if too long/too much text
    - max 2 lines
    - monotonic timing
    """
    chunks = sorted(chunks, key=lambda c: (c.start_ms, c.end_ms))
    chunks = _merge_short(chunks)
    chunks = _split_long(chunks)
    chunks = _enforce_cps(chunks)
    return chunks

def to_srt(chunks: List[CaptionChunk]) -> str:
    """
    Render formatted chunks as SRT text
    """
    out = []
    for i, c in enumerate(chunks, 1):
        out.append(str(i))
        out.append(f"{_ms_to_srt(c.start_ms)} --> {_ms_to_srt(c.end_ms)}")
        out.extend(_wrap_text(c.text))
        out.append("")  # blank line
    return "\n".join(out)

# --------- Helpers ----------

def _merge_short(chunks: List[CaptionChunk]) -> List[CaptionChunk]:
    """Merge chunks that are too short"""
    res = []
    for c in chunks:
        if res and (c.end_ms - c.start_ms) < MIN_DURATION_MS:
            prev = res[-1]
            # merge if they follow each other closely
            if c.start_ms - prev.end_ms <= 200:
                merged = CaptionChunk(
                    start_ms=prev.start_ms,
                    end_ms=max(prev.end_ms, c.end_ms),
                    text=(prev.text + " " + c.text).strip(),
                    lang=prev.lang,
                    source=prev.source,
                )
                res[-1] = merged
                continue
        res.append(c)
    return res

def _split_long(chunks: List[CaptionChunk]) -> List[CaptionChunk]:
    """Split chunks that are too long"""
    res = []
    for c in chunks:
        dur = c.end_ms - c.start_ms
        if dur <= MAX_DURATION_MS and len(c.text) <= MAX_LINE_CHARS*MAX_LINES:
            res.append(c)
            continue
        
        # split heuristically at sentence marks or spaces
        parts = _smart_split(c.text, MAX_LINE_CHARS*MAX_LINES)
        if len(parts) == 1:
            res.append(c)
            continue
        
        # distribute duration proportionally by text length
        total_chars = sum(len(p) for p in parts)
        start = c.start_ms
        for p in parts:
            frac = max(1, int(dur * (len(p)/total_chars)))
            res.append(CaptionChunk(start, start+frac, p.strip(), c.lang, c.source))
            start += frac
        
        # small correction: last end exactly on c.end_ms
        if res:
            res[-1] = CaptionChunk(
                start_ms=res[-1].start_ms,
                end_ms=c.end_ms,
                text=res[-1].text,
                lang=res[-1].lang,
                source=res[-1].source
            )
    return res

def _enforce_cps(chunks: List[CaptionChunk]) -> List[CaptionChunk]:
    """Enforce maximum characters per second reading speed"""
    res = []
    for c in chunks:
        dur = max(1, c.end_ms - c.start_ms)
        cps = len(c.text) / (dur/1000)
        if cps <= MAX_CPS:
            res.append(c)
            continue
        
        # too fast: split into smaller portions
        parts = _smart_split(c.text, max(1, int(MAX_CPS * (dur/1000))))
        if len(parts) == 1:
            res.append(c)
            continue
        
        total_chars = sum(len(p) for p in parts)
        start = c.start_ms
        for p in parts:
            frac = max(400, int(dur * (len(p)/total_chars)))  # min 0.4s
            res.append(CaptionChunk(start, start+frac, p.strip(), c.lang, c.source))
            start += frac
        
        if res:
            res[-1] = CaptionChunk(
                start_ms=res[-1].start_ms,
                end_ms=c.end_ms,
                text=res[-1].text,
                lang=res[-1].lang,
                source=res[-1].source
            )
    return res

def _ms_to_srt(ms: int) -> str:
    """Convert milliseconds to SRT timestamp format"""
    h = ms // 3600000
    ms -= h*3600000
    m = ms // 60000
    ms -= m*60000
    s = ms // 1000
    ms -= s*1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def _wrap_text(text: str) -> List[str]:
    """Wrap text to maximum line width"""
    words = text.split()
    lines, cur = [], ""
    for w in words:
        if len(cur)+1+len(w) <= MAX_LINE_CHARS:
            cur = (cur + " " + w).strip()
        else:
            lines.append(cur)
            cur = w
    if cur: 
        lines.append(cur)
    return lines[:MAX_LINES] if len(lines) > MAX_LINES else lines

def _smart_split(text: str, max_chars: int) -> List[str]:
    """Smart text splitting at sentence boundaries"""
    # first try sentence marks
    for sep in [". ", "? ", "! ", "; "]:
        parts = _split_keep(text, sep)
        if all(len(p) <= max_chars for p in parts):
            return parts
    
    # fallback: hard word wraps
    words = text.split()
    parts, cur = [], ""
    for w in words:
        if len(cur)+1+len(w) <= max_chars:
            cur = (cur + " " + w).strip()
        else:
            parts.append(cur)
            cur = w
    if cur: 
        parts.append(cur)
    return parts

def _split_keep(text: str, sep: str) -> List[str]:
    """Split text keeping the separator"""
    out, buf = [], ""
    for token in text.split(sep):
        if buf:
            out.append((buf + sep).strip())
            buf = ""
        buf = token
    if buf: 
        out.append(buf.strip())
    return out