# Download YouTube video to local file using yt_dlp
# Usage: python scripts/download_video.py --url <URL> --out data/input.mp4
import argparse
from pathlib import Path
import sys

try:
    from yt_dlp import YoutubeDL
except Exception as e:
    print("ERROR: yt_dlp not installed. Install with: pip install yt-dlp", file=sys.stderr)
    raise


def download(url: str, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ydl_opts = {
        'format': 'best[ext=mp4]/best',
        'outtmpl': str(out_path),
        'quiet': True,
        'noprogress': True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    download(args.url, Path(args.out))
    print(f"Saved to {args.out}")
