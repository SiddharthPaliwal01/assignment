# ⚽ Soccer Ball Touch Analysis

A computer vision project that analyzes soccer juggling videos to detect and count ball-player interactions using YOLOv8 object detection and pose estimation.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange.svg)

## 🎯 Features

- **Ball Detection**: Real-time soccer ball tracking using YOLOv8
- **Pose Estimation**: Player ankle position detection for touch analysis
- **Touch Counting**: Automated left/right leg touch detection and counting
- **Ball Spin Analysis**: Optical flow-based spin direction estimation
- **Performance Optimization**: GTX 1650 Ti optimized configurations
- **Interactive Analysis**: Jupyter notebook with step-by-step analysis

## 📊 Results

The system successfully analyzes soccer juggling videos and provides:
- **Touch Detection**: Precise ball-foot contact identification
- **Touch Statistics**: Left vs right leg touch counts
- **Annotated Video**: Visual output with detected touches highlighted
- **Data Export**: CSV and JSON format results

### Sample Results
- **Total Touches**: 152 detected touches
- **Left Leg**: 73 touches
- **Right Leg**: 79 touches
- **Processing Speed**: ~2-4 FPS on GTX 1650 Ti

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ RAM

### Installation

1. **Clone the repository**
```bash
git clone [https://github.com/SiddharthPaliwal01/assignment.git]
cd soccer-touch-analysis
```

2. **Create virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download test video** (optional)
```bash
python scripts/download_video.py --url "https://www.youtube.com/watch?v=k9gRgg_tW24" --out data/input.mp4
```

### Usage

#### Option 1: Command Line Analysis
```bash
python -m src.app --video data/input.mp4 --out outputs/annotated.mp4
```

#### Option 2: Jupyter Notebook (Recommended)
```bash
jupyter notebook Soccer_Ball_Touch_Analysis.ipynb
```

#### Option 3: Standalone Script
```bash
python run_notebook_analysis.py
```

## 📁 Project Structure

```
soccer-touch-analysis/
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
├── Soccer_Ball_Touch_Analysis.ipynb   # Interactive Jupyter notebook
├── run_notebook_analysis.py           # Standalone analysis script
├── src/
│   └── app.py                         # Main application logic
├── scripts/
│   └── download_video.py              # Video download utility
├── data/                              # Input videos (create manually)
└── outputs/                           # Analysis results (auto-created)
    ├── annotated.mp4                  # Annotated video output
    ├── touch_events.csv               # Detailed touch data
    └── summary.json                   # Analysis summary
```

## ⚙️ Configuration

### Standard Configuration
- **Input Resolution**: 1280px
- **Detection Confidence**: 0.15
- **Touch Distance**: 60px
- **Speed Threshold**: 1.5 px/frame

### GTX 1650 Ti Optimized
- **Input Resolution**: 640px (60-70% speedup)
- **Detection Confidence**: 0.25
- **Memory Optimization**: torch.no_grad()
- **Processing Time**: ~6-8 minutes for 2061 frames

## 🔧 Technical Details

### Computer Vision Pipeline
1. **Object Detection**: YOLOv8n for ball and player detection
2. **Pose Estimation**: YOLOv8n-pose for ankle keypoint extraction
3. **Object Tracking**: IoU-based tracking for temporal consistency
4. **Touch Detection**: Proximity-based ball-ankle interaction analysis
5. **Spin Analysis**: Optical flow for ball rotation estimation

### Key Algorithms
- **Ball Tracking**: Simple IoU tracker with disappearance handling
- **Touch Detection**: Distance thresholding with debounce logic
- **Spin Estimation**: Farneback optical flow on ball ROI
- **Performance Monitoring**: Real-time FPS and progress tracking

## 📈 Performance Optimization

### Hardware Requirements
- **Minimum**: GTX 1650 Ti (4GB VRAM) + 16GB RAM
- **Recommended**: RTX 3060+ (8GB+ VRAM) + 32GB RAM
- **Processing Speed**: 2-4 FPS (optimized) to 1-2 FPS (standard)

### Optimization Strategies
1. **Resolution Scaling**: Reduce input resolution for faster processing
2. **Model Selection**: Use nano models (yolov8n) for speed
3. **Batch Processing**: Process frames in smaller batches
4. **Memory Management**: Use torch.no_grad() for inference

## 📊 Output Formats

### CSV Export (touch_events.csv)
```csv
frame,player_id,leg,ball_pos_x,ball_pos_y,ankle_pos_x,ankle_pos_y,distance,ball_speed
145,0,left,640,360,642,385,25.2,3.4
```

### JSON Summary (summary.json)
```json
{
  "total_touches": 152,
  "left_leg_touches": 73,
  "right_leg_touches": 79,
  "avg_ball_speed": 2.8,
  "processing_time_seconds": 485
}
```

## 🤖 Machine Learning Models

The project uses pre-trained YOLOv8 models:
- **yolov8n.pt**: Object detection (ball, person)
- **yolov8n-pose.pt**: Human pose estimation (17 keypoints)

Models are automatically downloaded on first run.

## 🐛 Troubleshooting

### Common Issues

**GPU Memory Error**
```bash
# Use optimized configuration
python run_notebook_analysis.py --config fast
```

**NumPy Compatibility**
```bash
# Downgrade NumPy if needed
pip install "numpy<2.0"
```

**CUDA Not Found**
```bash
# Install CPU-only version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## 🔬 Research Applications

- **Sports Analytics**: Player performance analysis
- **Biomechanics**: Movement pattern analysis
- **Computer Vision**: Real-time object interaction detection
- **Machine Learning**: Training data generation for sports AI

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- **Ultralytics**: YOLOv8 implementation
- **OpenCV**: Computer vision library
- **PyTorch**: Deep learning framework
- **YouTube Video**: Soccer juggling demonstration

## 📧 Contact

For questions or collaboration opportunities, please open an issue on GitHub.

---

**⭐ Star this repository if you found it helpful!**
