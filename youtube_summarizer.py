#!/usr/bin/env python3

import sys
import argparse
from transcript_extractor import TranscriptExtractor
from summarizer import TextSummarizer


def main():
    parser = argparse.ArgumentParser(description='YouTube Video Summarizer')
    parser.add_argument('url', help='YouTube video URL to summarize')
    parser.add_argument('--type', choices=['brief', 'detailed', 'bullet'], 
                       default='detailed', help='Summary type (default: detailed)')
    parser.add_argument('--language', default='en', 
                       help='Transcript language preference (default: en)')
    parser.add_argument('--api-key', help='OpenAI API key (optional if set in .env)')
    
    args = parser.parse_args()
    
    try:
        print(f"Processing YouTube video: {args.url}")
        print("=" * 50)
        
        # Extract transcript
        print("📄 Extracting transcript...")
        extractor = TranscriptExtractor()
        transcript_data = extractor.get_transcript(args.url, args.language)
        
        print(f"✅ Transcript extracted ({len(transcript_data['transcript'])} characters)")
        
        # Summarize
        print(f"🤖 Generating {args.type} summary...")
        summarizer = TextSummarizer(args.api_key)
        summary = summarizer.summarize(transcript_data['transcript'], args.type)
        
        print("✅ Summary generated!")
        print("\n" + "=" * 50)
        print("SUMMARY")
        print("=" * 50)
        print(summary)
        print("=" * 50)
        
        # Save to file
        video_id = transcript_data['video_id']
        output_file = f"summary_{video_id}_{args.type}.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"YouTube Video Summary\n")
            f.write(f"URL: {args.url}\n")
            f.write(f"Video ID: {video_id}\n")
            f.write(f"Summary Type: {args.type}\n")
            f.write(f"{'=' * 50}\n\n")
            f.write(summary)
        
        print(f"📁 Summary saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()