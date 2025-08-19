import numpy as np
from src.app import TouchCounter, Config

def test_touch_debounce():
    cfg = Config(video_path="dummy.mp4")
    tc = TouchCounter(cfg)
    ball = (100, 100)
    ankles = {"left_ankle": (100, 105), "right_ankle": (300, 300)}
    # First frame establishes last center, no event
    e = tc.update(ball, ankles, 0, 0.0)
    assert e is None
    # Next frame with move + near ankle should count
    e = tc.update((101, 101), ankles, 1, 0.0)
    assert e is not None
    # Immediately after, should be in cooldown
    e = tc.update((102, 102), ankles, 2, 0.0)
    assert e is None
