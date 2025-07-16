import librosa
import soundfile as sf
import numpy as np
from pathlib import Path


# util to split wav file into segments based on silence

# Configuration - edit these variables as needed
INPUT_AUDIO_PATH = "data/case-1/audio.wav"  # Path to your input WAV file
OUTPUT_DIR = "segments"  # Directory to save output segments
MIN_SILENCE_DURATION = 0.5  # Minimum silence duration in seconds
SILENCE_THRESHOLD = -40.0  # Silence threshold in dB
MIN_SEGMENT_DURATION = 0.5  # Minimum segment duration in seconds

def split_audio_by_silence(audio_path, output_dir, min_silence_duration=0.5, 
                          silence_threshold=-40.0, min_segment_duration=0.5):
    """
    Split audio file into segments based on silence detection.
    """
    # Load audio
    audio, sample_rate = librosa.load(audio_path, sr=None)
    audio_duration = librosa.get_duration(y=audio, sr=sample_rate)
    
    print(f"Loaded audio: {audio_path}")
    print(f"Duration: {audio_duration:.2f} seconds")
    
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    
    # Calculate RMS energy
    frame_length = int(0.025 * sample_rate)  # 25ms frames
    hop_length = int(0.010 * sample_rate)    # 10ms hop
    
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    rms_db = 20 * np.log10(rms + 1e-10)
    
    # Find silence frames
    silence_frames = rms_db < silence_threshold
    
    # Convert frame indices to time
    frame_times = librosa.frames_to_time(np.arange(len(silence_frames)), 
                                       sr=sample_rate, hop_length=hop_length)
    
    # Find continuous silence periods
    silence_periods = []
    start_time = None
    
    for i, is_silent in enumerate(silence_frames):
        if is_silent and start_time is None:
            start_time = frame_times[i]
        elif not is_silent and start_time is not None:
            end_time = frame_times[i]
            duration = end_time - start_time
            if duration >= min_silence_duration:
                silence_periods.append((start_time, end_time))
            start_time = None
    
    # Handle case where audio ends with silence
    if start_time is not None:
        end_time = frame_times[-1]
        duration = end_time - start_time
        if duration >= min_silence_duration:
            silence_periods.append((start_time, end_time))
    
    print(f"Found {len(silence_periods)} silence periods")
    
    # Find split points (middle of each silence period)
    split_points = [0.0]  # Start of audio
    for start_silence, end_silence in silence_periods:
        split_point = (start_silence + end_silence) / 2
        split_points.append(split_point)
    split_points.append(audio_duration)  # End of audio
    
    # Filter out segments that are too short
    filtered_splits = [split_points[0]]
    for i in range(1, len(split_points)):
        segment_duration = split_points[i] - split_points[i-1]
        if segment_duration >= min_segment_duration:
            filtered_splits.append(split_points[i])
    
    print(f"Split points: {[f'{p:.2f}s' for p in filtered_splits]}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Split and save segments
    base_name = Path(audio_path).stem
    segment_files = []
    
    for i in range(len(filtered_splits) - 1):
        start_time = filtered_splits[i]
        end_time = filtered_splits[i + 1]
        
        # Extract segment
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        segment = audio[start_sample:end_sample]
        
        # Save segment
        segment_filename = f"{start_time:.2f}_{end_time:.2f}.wav"
        segment_path = output_path / segment_filename
        
        sf.write(str(segment_path), segment, sample_rate)
        segment_files.append(str(segment_path))
        
        segment_duration = end_time - start_time
        print(f"Segment {i+1}: {start_time:.2f}s - {end_time:.2f}s ({segment_duration:.2f}s)")
    
    print(f"Created {len(segment_files)} segments in {output_path}")
    return segment_files

if __name__ == "__main__":
    # Run the splitting
    segment_files = split_audio_by_silence(
        INPUT_AUDIO_PATH, 
        OUTPUT_DIR,
        MIN_SILENCE_DURATION,
        SILENCE_THRESHOLD,
        MIN_SEGMENT_DURATION
    )
