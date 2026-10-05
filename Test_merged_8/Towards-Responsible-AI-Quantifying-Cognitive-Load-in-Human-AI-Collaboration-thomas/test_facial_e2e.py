"""End-to-end test of the FacialExpressionAnalyzer pipeline."""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

print("=" * 60)
print("STEP 1: Check if ONNX model exists")
print("=" * 60)
from pathlib import Path
onnx_path = Path(__file__).resolve().parent / "backend" / "models" / "facial_expression_recognition_mobilefacenet_2022july.onnx"
print(f"  Path: {onnx_path}")
print(f"  Exists: {onnx_path.exists()}")
if onnx_path.exists():
    print(f"  Size: {onnx_path.stat().st_size} bytes")

lm_path = Path(__file__).resolve().parent / "backend" / "face_landmarker.task"
print(f"\n  FaceLandmarker path: {lm_path}")
print(f"  Exists: {lm_path.exists()}")

print("\n" + "=" * 60)
print("STEP 2: Try importing FacialExpressionAnalyzer")
print("=" * 60)
try:
    from backend.vision_analysis import FacialExpressionAnalyzer
    print("  SUCCESS: FacialExpressionAnalyzer imported")
except Exception as e:
    print(f"  FAILED: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("STEP 3: Try instantiating FacialExpressionAnalyzer")
print("=" * 60)
try:
    analyzer = FacialExpressionAnalyzer()
    print("  SUCCESS: FacialExpressionAnalyzer created")
    print(f"  _face_mesh is None: {analyzer._face_mesh is None}")
    print(f"  _session type: {type(analyzer._session)}")
except Exception as e:
    print(f"  FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("STEP 4: Test analyze_frame with black frame (no face)")
print("=" * 60)
import numpy as np
black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
try:
    result = analyzer.analyze_frame(black_frame)
    print(f"  Result: {result}")
    print(f"  face_detected: {result.get('face_detected')}")
    print(f"  reason: {result.get('reason')}")
except Exception as e:
    print(f"  FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("STEP 5: Test with a synthetic face-like image")
print("=" * 60)
import cv2
# Create a simple frame with a bright oval to simulate a face
face_frame = np.ones((480, 640, 3), dtype=np.uint8) * 200
# Draw a face-like oval
cv2.ellipse(face_frame, (320, 240), (100, 130), 0, 0, 360, (220, 190, 170), -1)
# Draw eyes
cv2.circle(face_frame, (290, 210), 15, (50, 50, 50), -1)
cv2.circle(face_frame, (350, 210), 15, (50, 50, 50), -1)
# Draw mouth
cv2.ellipse(face_frame, (320, 280), (30, 15), 0, 0, 180, (50, 50, 50), 2)
try:
    result = analyzer.analyze_frame(face_frame)
    print(f"  face_detected: {result.get('face_detected')}")
    print(f"  emotion: {result.get('emotion')}")
    print(f"  model_confidence: {result.get('model_confidence')}")
    print(f"  reason: {result.get('reason')}")
except Exception as e:
    print(f"  FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("STEP 6: Test FacialExpressionService")
print("=" * 60)
try:
    from backend.vision_analysis import facial_expression_service
    print(f"  facial_expression_service type: {type(facial_expression_service)}")
    print(f"  Has _analyzer? {hasattr(facial_expression_service, '_analyzer') if hasattr(facial_expression_service, '_analyzer') else 'N/A'}")
except Exception as e:
    print(f"  FAILED to import facial_expression_service: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("STEP 7: Test process_frame with base64 image")
print("=" * 60)
try:
    import base64
    _, buffer = cv2.imencode('.jpg', face_frame)
    b64 = base64.b64encode(buffer).decode('utf-8')
    data_url = f"data:image/jpeg;base64,{b64}"
    
    result = facial_expression_service.process_frame(
        participant_id="test_user",
        question_id="q1",
        task_number=1,
        frame_id=1,
        elapsed_ms=100,
        elapsed_second=1,
        image_data_url=data_url,
    )
    print(f"  Result keys: {list(result.keys())}")
    print(f"  frame_result: {result.get('frame_result')}")
    print(f"  storage_events: {result.get('storage_events')}")
except Exception as e:
    print(f"  FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
