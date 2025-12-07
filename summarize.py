# ===== FILE: summarize.py =====
#!/usr/bin/env python3
"""
YouTube Video Summarizer - CLI
Offline AI-powered transcription and summarization
"""

import argparse
import logging
import time
from pathlib import Path
from modules.utils import setup_logging, load_config, get_device, format_time, ensure_dir
from modules.downloader import YouTubeDownloader
from modules.transcriber import SpeechTranscriber
from modules.summarizer import TextSummarizer
from modules.diarizer import SpeakerDiarizer

def main():
    parser = argparse.ArgumentParser(
        description="Summarize YouTube videos using offline AI models"
    )
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("-o", "--output", help="Output file path")
    parser.add_argument("-d", "--diarize", action="store_true", 
                       help="Enable speaker diarization")
    parser.add_argument("-v", "--verbose", action="store_true",
                       help="Verbose output")
    parser.add_argument("--max-length", type=int, default=150,
                       help="Maximum summary length (words)")
    parser.add_argument("--config", default="config.yaml",
                       help="Config file path")
    
    args = parser.parse_args()
    
    # Setup
    logger = setup_logging(args.verbose)
    config = load_config(args.config)
    device = get_device(config)
    
    # Override config with CLI args
    if args.diarize:
        config['processing']['enable_diarization'] = True
    if args.max_length:
        config['processing']['max_summary_length'] = args.max_length
    
    start_time = time.time()
    
    logger.info("=" * 50)
    logger.info("YouTube Video Summarizer")
    logger.info("=" * 50)
    logger.info(f"URL: {args.url}")
    logger.info(f"Device: {device}")
    logger.info(f"Diarization: {config['processing']['enable_diarization']}")
    
    try:
        # Step 1: Download audio
        logger.info("\n[1/4] Downloading audio...")
        downloader = YouTubeDownloader(config['output']['output_dir'])
        download_result = downloader.download(args.url)
        audio_path = download_result['audio_path']
        metadata = download_result['metadata']
        
        logger.info(f"Title: {metadata['title']}")
        logger.info(f"Duration: {format_time(metadata['duration'])}")
        
        # Step 2: Transcribe
        logger.info("\n[2/4] Transcribing audio...")
        transcriber = SpeechTranscriber(
            model_size=config['models']['whisper_size'],
            device=device,
            use_fp16=config['performance']['use_fp16']
        )
        transcript_result = transcriber.transcribe(audio_path)
        transcript = transcript_result['text']
        segments = transcript_result['segments']
        
        logger.info(f"Transcript: {len(transcript)} characters")
        logger.info(f"Language: {transcript_result['language']}")
        
        # Step 3: Diarization (optional)
        if config['processing']['enable_diarization']:
            logger.info("\n[3/4] Performing speaker diarization...")
            try:
                diarizer = SpeakerDiarizer(device=device)
                diarization = diarizer.diarize(audio_path)
                segments_with_speakers = diarizer.merge_with_transcript(segments, diarization)
                
                # Format transcript with speakers
                speaker_transcript = []
                current_speaker = None
                for seg in segments_with_speakers:
                    if seg['speaker'] != current_speaker:
                        current_speaker = seg['speaker']
                        speaker_transcript.append(f"\n{current_speaker}:")
                    speaker_transcript.append(seg['text'])
                
                transcript_for_summary = ' '.join(speaker_transcript)
                logger.info(f"Identified {len(set(s['speaker'] for s in segments_with_speakers))} speakers")
            except Exception as e:
                logger.warning(f"Diarization failed: {e}. Continuing without speaker labels.")
                transcript_for_summary = transcript
        else:
            logger.info("\n[3/4] Skipping diarization...")
            transcript_for_summary = transcript
        
        # Step 4: Summarize
        logger.info("\n[4/4] Generating summary...")
        summarizer = TextSummarizer(
            model_name=config['models']['summarizer'],
            device=device,
            use_fp16=config['performance']['use_fp16']
        )
        summary = summarizer.summarize(
            transcript_for_summary,
            max_length=config['processing']['max_summary_length'],
            min_length=config['processing']['min_summary_length'],
            chunk_size=config['processing']['chunk_size']
        )
        
        # Output results
        elapsed = time.time() - start_time
        
        logger.info("\n" + "=" * 50)
        logger.info("SUMMARY")
        logger.info("=" * 50)
        print(f"\n{summary}\n")
        
        logger.info("=" * 50)
        logger.info(f"Processing time: {format_time(elapsed)}")
        logger.info("=" * 50)
        
        # Save to file if requested
        if args.output:
            output_path = Path(args.output)
            ensure_dir(output_path.parent)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"Title: {metadata['title']}\n")
                f.write(f"URL: {args.url}\n")
                f.write(f"Duration: {format_time(metadata['duration'])}\n")
                f.write(f"\nSUMMARY:\n{summary}\n")
                
                if config['output']['save_transcript']:
                    f.write(f"\n\nFULL TRANSCRIPT:\n{transcript}\n")
            
            logger.info(f"Saved to: {output_path}")
        
        # Cleanup
        if not config['output']['save_audio']:
            downloader.cleanup(audio_path)
        
    except Exception as e:
        logger.error(f"\nError: {str(e)}", exc_info=args.verbose)
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())