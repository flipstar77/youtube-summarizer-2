"""
PDF Processing Module for YouTube Summarizer
Handles PDF upload, text extraction, and integration with vector database
"""

import os
import hashlib
from typing import Dict, Any, Optional, List
from datetime import datetime
import PyPDF2
import pdfplumber
from pathlib import Path
import re

class PDFProcessor:
    def __init__(self, upload_dir: str = "pdf_uploads", db=None, vectorizer=None):
        """
        Initialize PDF processor
        
        Args:
            upload_dir: Directory to store uploaded PDFs
            db: Database instance (Supabase or SQLite)
            vectorizer: Vector embedding service
        """
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(exist_ok=True)
        self.db = db
        self.vectorizer = vectorizer
        
    def generate_document_id(self, file_path: str) -> str:
        """Generate unique ID for PDF document"""
        with open(file_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()[:8]
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        return f"pdf_{file_hash}_{timestamp}"
    
    def extract_text_pypdf2(self, pdf_path: str) -> Dict[str, Any]:
        """Extract text using PyPDF2 (faster, works for most PDFs)"""
        try:
            text_content = []
            metadata = {}
            
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Extract metadata
                if pdf_reader.metadata:
                    metadata = {
                        'title': pdf_reader.metadata.get('/Title', ''),
                        'author': pdf_reader.metadata.get('/Author', ''),
                        'subject': pdf_reader.metadata.get('/Subject', ''),
                        'creator': pdf_reader.metadata.get('/Creator', ''),
                        'creation_date': str(pdf_reader.metadata.get('/CreationDate', ''))
                    }
                
                # Extract text from each page
                total_pages = len(pdf_reader.pages)
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():
                            text_content.append({
                                'page': page_num,
                                'text': page_text.strip()
                            })
                    except Exception as e:
                        print(f"[WARNING] Could not extract text from page {page_num}: {str(e)}")
                        continue
            
            return {
                'success': True,
                'method': 'PyPDF2',
                'metadata': metadata,
                'total_pages': total_pages,
                'pages': text_content,
                'full_text': '\n\n'.join([p['text'] for p in text_content])
            }
            
        except Exception as e:
            print(f"[ERROR] PyPDF2 extraction failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'method': 'PyPDF2'
            }
    
    def extract_text_pdfplumber(self, pdf_path: str) -> Dict[str, Any]:
        """Extract text using pdfplumber (better for complex layouts, tables)"""
        try:
            text_content = []
            tables_found = []
            
            with pdfplumber.open(pdf_path) as pdf:
                # Extract metadata
                metadata = pdf.metadata or {}
                total_pages = len(pdf.pages)
                
                for page_num, page in enumerate(pdf.pages, 1):
                    try:
                        # Extract text
                        page_text = page.extract_text()
                        if page_text:
                            text_content.append({
                                'page': page_num,
                                'text': page_text.strip()
                            })
                        
                        # Extract tables if present
                        tables = page.extract_tables()
                        if tables:
                            for table_idx, table in enumerate(tables):
                                tables_found.append({
                                    'page': page_num,
                                    'table_index': table_idx,
                                    'data': table
                                })
                                # Convert table to text representation
                                table_text = self._table_to_text(table)
                                if table_text:
                                    text_content.append({
                                        'page': page_num,
                                        'text': f"[TABLE]\n{table_text}\n[/TABLE]",
                                        'is_table': True
                                    })
                    
                    except Exception as e:
                        print(f"[WARNING] Could not extract from page {page_num}: {str(e)}")
                        continue
            
            return {
                'success': True,
                'method': 'pdfplumber',
                'metadata': metadata,
                'total_pages': total_pages,
                'pages': text_content,
                'tables': tables_found,
                'full_text': '\n\n'.join([p['text'] for p in text_content])
            }
            
        except Exception as e:
            print(f"[ERROR] pdfplumber extraction failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'method': 'pdfplumber'
            }
    
    def _table_to_text(self, table: List[List]) -> str:
        """Convert table data to readable text format"""
        if not table:
            return ""
        
        text_lines = []
        for row in table:
            # Filter out None values and join cells
            cleaned_row = [str(cell) if cell is not None else "" for cell in row]
            if any(cleaned_row):  # Skip empty rows
                text_lines.append(" | ".join(cleaned_row))
        
        return "\n".join(text_lines)
    
    def extract_text(self, pdf_path: str, method: str = 'auto') -> Dict[str, Any]:
        """
        Extract text from PDF using specified method
        
        Args:
            pdf_path: Path to PDF file
            method: 'pypdf2', 'pdfplumber', or 'auto' (tries both)
        
        Returns:
            Dictionary with extraction results
        """
        if method == 'pypdf2':
            return self.extract_text_pypdf2(pdf_path)
        elif method == 'pdfplumber':
            return self.extract_text_pdfplumber(pdf_path)
        else:  # auto - try PyPDF2 first (faster), fallback to pdfplumber
            result = self.extract_text_pypdf2(pdf_path)
            if not result['success'] or not result.get('full_text', '').strip():
                print("[INFO] PyPDF2 failed or returned empty, trying pdfplumber...")
                result = self.extract_text_pdfplumber(pdf_path)
            return result
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove page numbers (common patterns)
        text = re.sub(r'\n\d+\n', '\n', text)
        # Fix common OCR issues
        text = text.replace('ﬁ', 'fi').replace('ﬂ', 'fl')
        # Remove non-printable characters
        text = ''.join(char for char in text if char.isprintable() or char in '\n\t')
        return text.strip()
    
    def chunk_text_intelligent(self, text: str, max_chunk_size: int = 2000) -> List[Dict[str, Any]]:
        """
        Intelligently chunk text based on paragraphs and sections
        
        Returns:
            List of chunks with metadata
        """
        chunks = []
        
        # Split by double newlines (paragraphs) first
        paragraphs = text.split('\n\n')
        
        current_chunk = []
        current_size = 0
        chunk_index = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            para_size = len(para)
            
            # If single paragraph is too large, split by sentences
            if para_size > max_chunk_size:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                for sentence in sentences:
                    if current_size + len(sentence) > max_chunk_size and current_chunk:
                        chunks.append({
                            'index': chunk_index,
                            'text': ' '.join(current_chunk),
                            'size': current_size
                        })
                        chunk_index += 1
                        current_chunk = [sentence]
                        current_size = len(sentence)
                    else:
                        current_chunk.append(sentence)
                        current_size += len(sentence) + 1
            
            # Normal paragraph processing
            elif current_size + para_size > max_chunk_size and current_chunk:
                chunks.append({
                    'index': chunk_index,
                    'text': '\n\n'.join(current_chunk),
                    'size': current_size
                })
                chunk_index += 1
                current_chunk = [para]
                current_size = para_size
            else:
                current_chunk.append(para)
                current_size += para_size + 2  # +2 for \n\n
        
        # Add remaining chunk
        if current_chunk:
            chunks.append({
                'index': chunk_index,
                'text': '\n\n'.join(current_chunk),
                'size': current_size
            })
        
        return chunks
    
    def process_pdf(self, file_path: str, title: Optional[str] = None, 
                    source_type: str = 'pdf_upload') -> Dict[str, Any]:
        """
        Complete PDF processing pipeline
        
        Args:
            file_path: Path to uploaded PDF
            title: Optional title override
            source_type: Type of source for categorization
        
        Returns:
            Processing results including document ID and status
        """
        try:
            # Generate document ID
            doc_id = self.generate_document_id(file_path)
            
            # Extract text
            print(f"[INFO] Extracting text from PDF: {file_path}")
            extraction_result = self.extract_text(file_path)
            
            if not extraction_result['success']:
                return {
                    'success': False,
                    'error': f"Text extraction failed: {extraction_result.get('error', 'Unknown error')}"
                }
            
            # Clean text
            full_text = self.clean_text(extraction_result['full_text'])
            
            if not full_text:
                return {
                    'success': False,
                    'error': 'No text content found in PDF'
                }
            
            # Use title from metadata or parameter
            if not title:
                title = extraction_result.get('metadata', {}).get('title', '')
            if not title:
                title = Path(file_path).stem.replace('_', ' ').title()
            
            # Prepare metadata
            metadata = {
                'source_type': source_type,
                'extraction_method': extraction_result['method'],
                'total_pages': extraction_result['total_pages'],
                'has_tables': len(extraction_result.get('tables', [])) > 0,
                'original_metadata': extraction_result.get('metadata', {}),
                'file_path': str(file_path),
                'processed_at': datetime.now().isoformat()
            }
            
            # Save to database
            if self.db:
                print(f"[INFO] Saving PDF to database: {title}")
                summary_id = self._save_to_database(
                    doc_id=doc_id,
                    title=title,
                    content=full_text,
                    metadata=metadata
                )
                
                # Create vector embeddings if available
                if self.vectorizer and summary_id:
                    print(f"[INFO] Creating vector embeddings for PDF chunks")
                    chunks = self.chunk_text_intelligent(full_text)
                    embeddings_created = self._create_embeddings(
                        summary_id=summary_id,
                        chunks=chunks,
                        title=title
                    )
                    metadata['embeddings_created'] = embeddings_created
            else:
                summary_id = None
            
            return {
                'success': True,
                'document_id': doc_id,
                'summary_id': summary_id,
                'title': title,
                'text_length': len(full_text),
                'total_pages': extraction_result['total_pages'],
                'chunks': len(self.chunk_text_intelligent(full_text)),
                'metadata': metadata
            }
            
        except Exception as e:
            print(f"[ERROR] PDF processing failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _save_to_database(self, doc_id: str, title: str, content: str, 
                         metadata: Dict[str, Any]) -> Optional[int]:
        """Save PDF content to database"""
        try:
            # For Supabase
            if hasattr(self.db, 'client'):
                result = self.db.client.table('summaries').insert({
                    'video_id': doc_id,  # Reusing video_id field for document ID
                    'url': f"https://youtu.be/PDF_{doc_id}",  # Use YouTube-like URL format to pass constraint
                    'title': title,
                    'summary_type': 'detailed',
                    'summary': content[:5000],  # Store first 5000 chars as summary
                    'transcript_length': len(content),
                    'audio_file': None,
                    'voice_id': None
                }).execute()
                
                if result.data and len(result.data) > 0:
                    return result.data[0]['id']
            
            # For SQLite fallback
            else:
                return self.db.save_summary(
                    video_id=doc_id,
                    url=f"pdf://{metadata['file_path']}",
                    title=title,
                    summary_type='detailed',
                    summary=content[:5000],
                    transcript_length=len(content)
                )
                
        except Exception as e:
            print(f"[ERROR] Database save failed: {str(e)}")
            return None
    
    def _create_embeddings(self, summary_id: int, chunks: List[Dict[str, Any]], 
                          title: str) -> int:
        """Create vector embeddings for PDF chunks"""
        embeddings_created = 0
        
        try:
            for chunk in chunks:
                # Combine title with chunk for better context
                chunk_text = f"{title}\n\n{chunk['text']}"
                
                # Create embedding using existing vectorizer
                if self.vectorizer:
                    try:
                        self.vectorizer.create_and_store_embedding(
                            summary_id=summary_id,
                            text=chunk_text,
                            metadata={
                                'chunk_index': chunk['index'],
                                'chunk_size': chunk['size'],
                                'source_type': 'pdf'
                            }
                        )
                        embeddings_created += 1
                    except Exception as e:
                        print(f"[WARNING] Failed to create embedding for chunk {chunk['index']}: {str(e)}")
                        continue
                        
        except Exception as e:
            print(f"[ERROR] Embedding creation failed: {str(e)}")
        
        return embeddings_created