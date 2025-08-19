# Data Directory

This directory contains input video files for analysis.

## Usage

Place your video files here:
- `input.mp4` - Main input video for analysis
- `input_720.mp4` - 720p version for faster processing

## Download Test Video

Use the provided script to download the sample video:

```bash
python scripts/download_video.py --url "https://www.youtube.com/watch?v=k9gRgg_tW24" --out data/input.mp4
```

## Supported Formats

- MP4 (recommended)
- AVI
- MOV
- Any format supported by OpenCV

## Notes

- Keep videos under 100MB for better performance
- 720p resolution provides good balance of quality and speed
- Videos are automatically excluded from git tracking
