import urllib.request
import json
import urllib.error

data = {
    "participant_id": "COG0001",
    "question_id": "Q1",
    "task_number": 1,
    "elapsed_second": 1,
    "camera": {
        "participant_id": "COG0001",
        "question_id": "Q1",
        "task_number": 1,
        "elapsed_second": 1,
        "first_frame_id": 1,
        "last_frame_id": 1,
        "frame_count": 1,
        "paired_frame_count": 1,
        "first_captured_at": "2026-09-05T00:00:00Z",
        "last_captured_at": "2026-09-05T00:00:00Z",
        "eye_sample_count": 0,
        "eye_face_detected_ratio": 0.0,
        "mean_ear": 0.0,
        "min_ear": 0.0,
        "max_ear": 0.0,
        "mean_perclos": 0.0,
        "max_perclos": 0.0,
        "dominant_gaze": "unknown",
        "gaze_change_count": 0,
        "saccade_count": 0,
        "blink_events": 0,
        "blink_count_end": 0,
        "mean_blink_latency_ms": 0.0,
        "mean_head_pitch": 0.0,
        "mean_head_yaw": 0.0,
        "dominant_fatigue": "unknown",
        "mean_tracking_confidence": 0.0,
        "facial_sample_count": 0,
        "facial_face_detected_ratio": 0.0,
        "dominant_emotion": "unknown",
        "mean_emotion_confidence": 0.0,
        "emotion_change_count": 0,
        "angry_ratio": 0.0,
        "disgust_ratio": 0.0,
        "fearful_ratio": 0.0,
        "happy_ratio": 0.0,
        "neutral_ratio": 0.0,
        "sad_ratio": 0.0,
        "surprised_ratio": 0.0
    },
    "behavior": {
        "participant_id": "COG0001",
        "question_id": "Q1",
        "task_number": 1,
        "elapsed_second": 1,
        "captured_at": "2026-09-05T00:00:00Z",
        "keypress_count": 0,
        "backspace_count": 0,
        "typing_active": False,
        "thinking_pause_seconds": 0.0,
        "mouse_move_count": 0,
        "cursor_distance_px": 0.0,
        "scroll_up_count": 0,
        "scroll_down_count": 0,
        "scroll_count": 0,
        "mouse_active": False,
        "prompt_sent": False,
        "prompt_length_words": 0,
        "prompt_count_so_far": 0,
        "time_since_last_prompt_seconds": 0.0,
        "prompt_sentiment": "none"
    }
}

req = urllib.request.Request(
    'http://127.0.0.1:8002/api/multimodal-tracking/second', 
    data=json.dumps(data).encode(), 
    headers={'Content-Type': 'application/json', 'Authorization': 'Bearer 123'}
)

try:
    with urllib.request.urlopen(req) as response:
        print(response.read().decode())
except urllib.error.HTTPError as e:
    print(e.code)
    print(e.read().decode())
