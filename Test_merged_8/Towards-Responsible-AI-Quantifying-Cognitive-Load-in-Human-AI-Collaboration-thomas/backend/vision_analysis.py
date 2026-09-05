from __future__ import annotations

import base64
import hashlib
import json
import math
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any


LEFT_EAR_POINTS = (362, 385, 387, 263, 373, 380)
RIGHT_EAR_POINTS = (33, 160, 158, 133, 153, 144)
LEFT_IRIS = (474, 475, 476, 477)
RIGHT_IRIS = (469, 470, 471, 472)
HEAD_POSE_LANDMARKS = (1, 152, 226, 446, 57, 287)


@dataclass
class VisionState:
    calibrated_left: Any = None
    calibrated_right: Any = None
    calibration_left: dict[int, list[Any]] = field(default_factory=dict)
    calibration_right: dict[int, list[Any]] = field(default_factory=dict)
    ear_history: deque[float] = field(default_factory=lambda: deque(maxlen=600))
    calibration_ears: list[float] = field(default_factory=list)
    ear_threshold: float = 0.21
    blink_count: int = 0
    eyes_closed: bool = False
    closed_frames: int = 0
    closed_started_at: float = 0.0
    last_blink_latency_ms: float = 0.0
    closure_history: deque[tuple[float, bool]] = field(default_factory=deque)
    valid_sample_count: int = 0
    last_gaze: Any = None
    last_gaze_at: float = 0.0
    directions: deque[str] = field(default_factory=lambda: deque(maxlen=5))


class VisionAnalyzer:
    """Web-frame adapter for the reusable eye-tracking algorithms."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._states: dict[str, VisionState] = {}
        self._face_mesh = None

    @staticmethod
    def _decode(data_url: str):
        import cv2
        import numpy as np

        encoded = data_url.split(",", 1)[-1]
        raw = base64.b64decode(encoded, validate=True)
        frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Invalid camera frame")
        return frame

    def _mesh(self):
        if self._face_mesh is None:
            try:
                import mediapipe as mp
                solutions = getattr(mp, "solutions", None)
                if solutions is not None and hasattr(solutions, "face_mesh"):
                    self._face_mesh = solutions.face_mesh.FaceMesh(
                        max_num_faces=5,
                        refine_landmarks=True,
                        min_detection_confidence=0.6,
                        min_tracking_confidence=0.6,
                    )
            except Exception:
                self._face_mesh = None
        return self._face_mesh

    def forget_participant(self, participant_id: str) -> None:
        """Release calibration and rolling eye state after a session ends."""
        with self._lock:
            self._states.pop(participant_id, None)

    @staticmethod
    def _point(landmarks, index: int, width: int, height: int):
        import numpy as np

        item = landmarks[index]
        return np.array((item.x * width, item.y * height), dtype=float)

    @classmethod
    def _ear(cls, landmarks, points, width: int, height: int) -> float:
        import numpy as np

        p1, p2, p3, p4, p5, p6 = [cls._point(landmarks, i, width, height) for i in points]
        return float((np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2 * np.linalg.norm(p1 - p4) + 1e-6))

    @classmethod
    def _iris(cls, landmarks, points, width: int, height: int):
        import numpy as np

        return np.mean([cls._point(landmarks, i, width, height) for i in points], axis=0)

    @staticmethod
    def _direction(vector) -> str:
        dx, dy = vector
        horizontal = "Left" if dx < -0.018 else "Right" if dx > 0.018 else "Centre"
        vertical = "Up" if dy < -0.018 else "Down" if dy > 0.018 else "Centre"
        if horizontal == "Centre":
            return vertical
        if vertical == "Centre":
            return horizontal
        return f"{vertical}-{horizontal}"

    @classmethod
    def _head_pose(cls, landmarks, width: int, height: int) -> tuple[float, float]:
        import cv2
        import numpy as np

        model_points = np.array(
            ((0, 0, 0), (0, -330, -65), (-225, 170, -135),
             (225, 170, -135), (-150, -150, -125), (150, -150, -125)),
            dtype=np.float64,
        )
        image_points = np.array(
            [cls._point(landmarks, index, width, height) for index in HEAD_POSE_LANDMARKS],
            dtype=np.float64,
        )
        camera = np.array(
            ((width, 0, width / 2), (0, width, height / 2), (0, 0, 1)),
            dtype=np.float64,
        )
        solved, rotation_vector, _ = cv2.solvePnP(
            model_points,
            image_points,
            camera,
            np.zeros((4, 1)),
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not solved:
            return 0.0, 0.0
        rotation, _ = cv2.Rodrigues(rotation_vector)
        sy = math.sqrt(rotation[0, 0] ** 2 + rotation[1, 0] ** 2)
        pitch = math.degrees(math.atan2(-rotation[2, 0], sy))
        yaw = math.degrees(math.atan2(rotation[1, 0], rotation[0, 0]))
        return pitch, yaw

    @staticmethod
    def _update_blink_state(state: VisionState, ear: float, now: float) -> bool:
        """Update one participant's blink state and return whether a blink ended.

        Eye frames are sampled asynchronously, so requiring a fixed number of
        closed frames makes normal short blinks impossible to detect when the
        sampling interval is longer than the blink. Duration-based detection
        works with both fast and slower camera/API sampling. A small hysteresis
        prevents EAR noise near the threshold from repeatedly opening an eye.
        """
        reopen_threshold = state.ear_threshold + 0.015
        if not state.eyes_closed:
            if ear < state.ear_threshold:
                state.eyes_closed = True
                state.closed_started_at = now
                state.closed_frames = 1
            return False

        state.closed_frames += 1
        if ear < reopen_threshold:
            return False

        closure_duration = now - state.closed_started_at
        blinked = 0.04 <= closure_duration <= 0.8
        if blinked:
            state.blink_count += 1
            state.last_blink_latency_ms = round(closure_duration * 1000, 1)
        state.eyes_closed = False
        state.closed_frames = 0
        state.closed_started_at = 0.0
        return blinked

    def analyze(self, participant_id: str, data_url: str, calibration_point: int | None = None) -> dict[str, Any]:
        import cv2
        import numpy as np

        mesh = self._mesh()
        if mesh is None:
            return {
                "face_detected": False,
                "pupils_detected": False,
                "calibrating": False,
                "calibration_complete": False,
                "tracking_confidence": 0.0,
            }
        frame = self._decode(data_url)
        height, width = frame.shape[:2]
        with self._inference_lock:
            results = mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not results or not getattr(results, "multi_face_landmarks", None):
            return {
                "face_detected": False,
                "pupils_detected": False,
                "calibrating": False,
                "calibration_complete": False,
                "tracking_confidence": 0.0,
            }

        face_landmark_sets = [face.landmark for face in results.multi_face_landmarks[:5]]
        landmarks = face_landmark_sets[0]
        faces_detected = len(face_landmark_sets)
        left_iris = self._iris(landmarks, LEFT_IRIS, width, height)
        right_iris = self._iris(landmarks, RIGHT_IRIS, width, height)
        ear = (self._ear(landmarks, LEFT_EAR_POINTS, width, height) + self._ear(landmarks, RIGHT_EAR_POINTS, width, height)) / 2
        try:
            head_pitch, head_yaw = self._head_pose(landmarks, width, height)
        except Exception:
            head_pitch, head_yaw = 0.0, 0.0
        now = time.time()

        with self._lock:
            state = self._states.setdefault(participant_id, VisionState())
            state.valid_sample_count += 1
            if state.calibrated_left is None:
                state.calibration_ears.append(ear)
                if calibration_point is None:
                    left_samples = state.calibration_left.setdefault(-1, [])
                    right_samples = state.calibration_right.setdefault(-1, [])
                    if len(left_samples) < 10:
                        left_samples.append(left_iris)
                        right_samples.append(right_iris)
                    calibration = min(100, len(left_samples) * 10)
                    if len(left_samples) >= 10:
                        state.calibrated_left = np.median(left_samples, axis=0)
                        state.calibrated_right = np.median(right_samples, axis=0)
                        state.ear_threshold = min(0.25, max(0.16, float(np.median(state.calibration_ears)) * 0.72))
                    return {
                        "face_detected": True,
                        "faces_detected": faces_detected,
                        "face_signatures": self._face_signatures(face_landmark_sets),
                        "pupils_detected": True,
                        "calibrating": state.calibrated_left is None,
                        "calibration_complete": state.calibrated_left is not None,
                        "calibration_percent": calibration,
                        "left_pupil_x": round(float(left_iris[0]), 3),
                        "left_pupil_y": round(float(left_iris[1]), 3),
                        "right_pupil_x": round(float(right_iris[0]), 3),
                        "right_pupil_y": round(float(right_iris[1]), 3),
                    }
                if not 0 <= calibration_point < 16:
                    raise ValueError("calibration_point must be between 0 and 15")
                left_samples = state.calibration_left.setdefault(calibration_point, [])
                right_samples = state.calibration_right.setdefault(calibration_point, [])
                if len(left_samples) < 2:
                    left_samples.append(left_iris)
                    right_samples.append(right_iris)
                completed_points = sum(len(samples) >= 2 for samples in state.calibration_left.values())
                calibration = round(completed_points / 16 * 100)
                point_complete = len(left_samples) >= 2
                calibration_complete = completed_points == 16
                if calibration_complete:
                    central_points = (5, 6, 9, 10)
                    state.calibrated_left = np.mean([sample for point in central_points for sample in state.calibration_left[point]], axis=0)
                    state.calibrated_right = np.mean([sample for point in central_points for sample in state.calibration_right[point]], axis=0)
                    state.ear_threshold = min(0.25, max(0.16, float(np.median(state.calibration_ears)) * 0.72))
                return {
                    "face_detected": True,
                    "faces_detected": faces_detected,
                    "face_signatures": self._face_signatures(face_landmark_sets),
                    "pupils_detected": True,
                    "calibrating": not calibration_complete,
                    "calibration_point": calibration_point,
                    "calibration_point_complete": point_complete,
                    "calibration_complete": calibration_complete,
                    "calibration_percent": calibration,
                    "left_pupil_x": round(float(left_iris[0]), 3),
                    "left_pupil_y": round(float(left_iris[1]), 3),
                    "right_pupil_x": round(float(right_iris[0]), 3),
                    "right_pupil_y": round(float(right_iris[1]), 3),
                }

            interocular_distance = max(1.0, float(np.linalg.norm(left_iris - right_iris)))
            gaze = (((left_iris - state.calibrated_left) + (right_iris - state.calibrated_right)) / 2) / interocular_distance
            direction = self._direction(gaze)
            state.directions.append(direction)
            direction = Counter(state.directions).most_common(1)[0][0]
            speed = 0.0
            if state.last_gaze is not None and now > state.last_gaze_at:
                speed = float(np.linalg.norm(gaze - state.last_gaze) / (now - state.last_gaze_at))
            state.last_gaze, state.last_gaze_at = gaze, now
            state.ear_history.append(ear)
            blinked = self._update_blink_state(state, ear, now)
            closed = state.eyes_closed
            state.closure_history.append((now, closed))
            while state.closure_history and now - state.closure_history[0][0] > 60:
                state.closure_history.popleft()
            perclos = sum(item[1] for item in state.closure_history) / len(state.closure_history)
            fatigue_score = min(1.0, max(0.0, (0.30 - ear) / 0.30) * 0.5 + min(1.0, perclos / 0.3) * 0.5)
            pupil_symmetry = max(0.0, 1.0 - abs(left_iris[1] - right_iris[1]) / max(1.0, interocular_distance * 0.2))
            pose_confidence = max(0.0, 1.0 - (abs(head_yaw) / 35 + abs(head_pitch) / 35) / 2)
            tracking_confidence = min(1.0, pupil_symmetry * 0.65 + pose_confidence * 0.35)

        output: dict[str, Any] = {
            "face_detected": True,
            "faces_detected": faces_detected,
            "face_signatures": self._face_signatures(face_landmark_sets),
            "pupils_detected": True,
            "calibrating": False,
            "direction": direction,
            "saccade": "fast" if speed > 0.35 else "focused",
            "blinked": blinked,
            "blink_count": state.blink_count,
            "blink_latency_ms": state.last_blink_latency_ms if blinked else 0.0,
            "ear": round(ear, 4),
            "ear_threshold": round(state.ear_threshold, 4),
            "perclos": round(perclos, 4),
            "fatigue_score": round(fatigue_score, 4),
            "fatigue": "high" if fatigue_score >= 0.6 else "moderate" if fatigue_score >= 0.3 else "low",
            "head_pitch": round(head_pitch, 2),
            "head_yaw": round(head_yaw, 2),
            "left_pupil_x": round(float(left_iris[0]), 3),
            "left_pupil_y": round(float(left_iris[1]), 3),
            "right_pupil_x": round(float(right_iris[0]), 3),
            "right_pupil_y": round(float(right_iris[1]), 3),
            "tracking_confidence": round(tracking_confidence, 4),
            "valid_sample_count": state.valid_sample_count,
        }
        return output

    @staticmethod
    def _face_signatures(face_landmark_sets) -> list[str]:
        """Return short anonymized signatures for up to five detected faces.

        The signatures are derived from normalized, quantized landmarks rather
        than storing camera images. They are useful for counting/enrollment and
        are not intended as biometric authentication credentials.
        """
        import numpy as np

        signatures: list[str] = []
        anchor_indices = (1, 33, 61, 152, 263, 291, 10, 234, 454, 168)
        for landmarks in face_landmark_sets[:5]:
            points = np.array([[landmarks[index].x, landmarks[index].y, landmarks[index].z] for index in anchor_indices], dtype=float)
            minimum = points[:, :2].min(axis=0)
            maximum = points[:, :2].max(axis=0)
            scale = np.maximum(maximum - minimum, 1e-6)
            normalized = points.copy()
            normalized[:, :2] = (normalized[:, :2] - minimum) / scale
            normalized[:, 2] = normalized[:, 2] / max(float(scale.max()), 1e-6)
            quantized = np.round(normalized, 3).tolist()
            signatures.append(hashlib.sha256(json.dumps(quantized, separators=(",", ":")).encode("utf-8")).hexdigest()[:32])
        return signatures


vision_analyzer = VisionAnalyzer()


# ----------------------------------------------------------------------
# Merged from facial_expression.py
# ----------------------------------------------------------------------


import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_FILENAME = (
    "facial_expression_recognition_mobilefacenet_2022july.onnx"
)

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / MODEL_FILENAME


# The order must match the exact output order of the OpenCV model.
EMOTION_LABELS = (
    "angry",
    "disgust",
    "fearful",
    "happy",
    "neutral",
    "sad",
    "surprised",
)


# Standard five-point facial template used for 112 × 112 aligned faces.
#
# Point order:
# 1. Image-left eye
# 2. Image-right eye
# 3. Nose tip
# 4. Image-left mouth corner
# 5. Image-right mouth corner
FACE_ALIGNMENT_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


class FacialExpressionAnalyzer:
    """
    Detect, align and classify visible facial expressions.

    The component is independent of eye tracking. It receives a BGR image
    and returns one facial-expression result.
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        minimum_detection_confidence: float = 0.60,
        minimum_tracking_confidence: float = 0.60,
    ) -> None:
        self.model_path = Path(model_path).resolve()

        if not self.model_path.exists():
            raise FileNotFoundError(
                "Facial-expression model was not found at:\n"
                f"{self.model_path}\n\n"
                "Run setup_facial_model.py before starting the application."
            )

        if not 0.0 <= minimum_detection_confidence <= 1.0:
            raise ValueError(
                "minimum_detection_confidence must be between 0 and 1."
            )

        if not 0.0 <= minimum_tracking_confidence <= 1.0:
            raise ValueError(
                "minimum_tracking_confidence must be between 0 and 1."
            )

        self._session = self._create_onnx_session()

        self._input = self._session.get_inputs()[0]
        self._input_name = self._input.name

        self._output_names = [
            output.name for output in self._session.get_outputs()
        ]

        self._validate_model_structure()

        # MediaPipe detects and tracks the face landmarks.
        solutions = getattr(mp, "solutions", None)
        if solutions is not None and hasattr(solutions, "face_mesh"):
            self._face_mesh = solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=minimum_detection_confidence,
                min_tracking_confidence=minimum_tracking_confidence,
            )
        else:
            self._face_mesh = None

    def _create_onnx_session(self) -> ort.InferenceSession:
        """
        Load the model using the CPU execution provider.
        """

        options = ort.SessionOptions()

        # Restricting the model to one inference thread helps prevent the
        # facial module from taking every CPU core away from eye tracking.
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1

        options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        return ort.InferenceSession(
            str(self.model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )

    def _validate_model_structure(self) -> None:
        """
        Confirm that the loaded model has a usable image input.
        """

        input_shape = self._input.shape

        if len(input_shape) != 4:
            raise RuntimeError(
                "Expected a four-dimensional ONNX input such as "
                f"[batch, channels, height, width], but received {input_shape}."
            )

        if not self._output_names:
            raise RuntimeError(
                "The ONNX model does not contain any output nodes."
            )

    @staticmethod
    def _average_landmarks(
        landmarks: Any,
        landmark_indices: tuple[int, ...],
        image_width: int,
        image_height: int,
    ) -> np.ndarray:
        """
        Calculate the average pixel position of selected landmarks.
        """

        points = []

        for index in landmark_indices:
            landmark = landmarks[index]

            points.append(
                [
                    landmark.x * image_width,
                    landmark.y * image_height,
                ]
            )

        return np.mean(
            np.asarray(points, dtype=np.float32),
            axis=0,
        )

    def _extract_alignment_points(
        self,
        landmarks: Any,
        image_width: int,
        image_height: int,
    ) -> np.ndarray:
        """
        Convert MediaPipe landmarks into five alignment points.
        """

        # These pairs represent the eye areas.
        image_left_eye = self._average_landmarks(
            landmarks,
            (33, 133),
            image_width,
            image_height,
        )

        image_right_eye = self._average_landmarks(
            landmarks,
            (362, 263),
            image_width,
            image_height,
        )

        nose_tip = self._average_landmarks(
            landmarks,
            (1,),
            image_width,
            image_height,
        )

        image_left_mouth = self._average_landmarks(
            landmarks,
            (61,),
            image_width,
            image_height,
        )

        image_right_mouth = self._average_landmarks(
            landmarks,
            (291,),
            image_width,
            image_height,
        )

        return np.asarray(
            [
                image_left_eye,
                image_right_eye,
                nose_tip,
                image_left_mouth,
                image_right_mouth,
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _points_are_valid(
        points: np.ndarray,
        image_width: int,
        image_height: int,
    ) -> bool:
        """
        Reject invalid or extreme landmark coordinates.
        """

        if points.shape != (5, 2):
            return False

        if not np.isfinite(points).all():
            return False

        # A small margin is allowed because landmarks can occasionally appear
        # slightly outside the image when the face is near the camera border.
        horizontal_margin = image_width * 0.10
        vertical_margin = image_height * 0.10

        x_valid = np.logical_and(
            points[:, 0] >= -horizontal_margin,
            points[:, 0] <= image_width + horizontal_margin,
        )

        y_valid = np.logical_and(
            points[:, 1] >= -vertical_margin,
            points[:, 1] <= image_height + vertical_margin,
        )

        return bool(np.all(x_valid) and np.all(y_valid))

    @staticmethod
    def _align_face(
        frame_bgr: np.ndarray,
        source_points: np.ndarray,
    ) -> np.ndarray | None:
        """
        Transform the detected face into the model's 112 × 112 template.
        """

        transformation, _ = cv2.estimateAffinePartial2D(
            source_points,
            FACE_ALIGNMENT_TEMPLATE,
            method=cv2.LMEDS,
        )

        if transformation is None:
            return None

        aligned_face = cv2.warpAffine(
            frame_bgr,
            transformation,
            (112, 112),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        return aligned_face

    @staticmethod
    def _prepare_model_input(
        aligned_face_bgr: np.ndarray,
    ) -> np.ndarray:
        """
        Apply the preprocessing expected by the OpenCV FER model.

        Processing:
        BGR → RGB
        uint8 → float32
        [0, 255] → [0, 1]
        normalize using mean=0.5 and standard deviation=0.5
        HWC → CHW
        add batch dimension
        """

        if aligned_face_bgr.shape[:2] != (112, 112):
            aligned_face_bgr = cv2.resize(
                aligned_face_bgr,
                (112, 112),
                interpolation=cv2.INTER_LINEAR,
            )

        face_rgb = cv2.cvtColor(
            aligned_face_bgr,
            cv2.COLOR_BGR2RGB,
        )

        normalized = face_rgb.astype(np.float32) / 255.0

        normalized = (normalized - 0.5) / 0.5

        # Convert:
        # height × width × channels
        # to
        # channels × height × width
        input_tensor = np.transpose(
            normalized,
            (2, 0, 1),
        )

        input_tensor = np.expand_dims(
            input_tensor,
            axis=0,
        )

        return np.ascontiguousarray(
            input_tensor,
            dtype=np.float32,
        )

    @staticmethod
    def _convert_scores_to_probabilities(
        raw_scores: np.ndarray,
    ) -> np.ndarray:
        """
        Convert model output scores to normalized relative scores.

        The resulting values add up to 1. These should be treated as model
        scores, not guaranteed calibrated psychological probabilities.
        """

        scores = np.asarray(
            raw_scores,
            dtype=np.float32,
        ).reshape(-1)

        if scores.size != len(EMOTION_LABELS):
            raise RuntimeError(
                "Expected seven facial-expression scores, "
                f"but the model returned {scores.size}."
            )

        # If the output already looks like probabilities, normalize it safely.
        if np.all(scores >= 0.0) and np.isclose(
            np.sum(scores),
            1.0,
            atol=1e-3,
        ):
            total = float(np.sum(scores))

            if total <= 0.0:
                return np.full(
                    len(EMOTION_LABELS),
                    1.0 / len(EMOTION_LABELS),
                    dtype=np.float32,
                )

            return scores / total

        # Otherwise treat the values as logits and apply a stable softmax.
        shifted_scores = scores - np.max(scores)
        exponentials = np.exp(shifted_scores)
        denominator = float(np.sum(exponentials))

        if denominator <= 0.0 or not np.isfinite(denominator):
            raise RuntimeError(
                "The model produced invalid expression scores."
            )

        return exponentials / denominator

    def analyze_frame(
        self,
        frame_bgr: np.ndarray,
    ) -> dict[str, Any]:
        """
        Analyze one BGR webcam frame.

        Returns:
            {
                "face_detected": bool,
                "emotion": str,
                "model_confidence": float,
                "scores": dict,
                "inference_ms": float,
                "processing_ms": float
            }
        """

        processing_started = time.perf_counter()

        if frame_bgr is None:
            return self._invalid_result(
                reason="frame_is_none",
                processing_started=processing_started,
            )

        if not isinstance(frame_bgr, np.ndarray):
            return self._invalid_result(
                reason="frame_is_not_numpy_array",
                processing_started=processing_started,
            )

        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            return self._invalid_result(
                reason="invalid_frame_shape",
                processing_started=processing_started,
            )

        if frame_bgr.size == 0:
            return self._invalid_result(
                reason="empty_frame",
                processing_started=processing_started,
            )

        image_height, image_width = frame_bgr.shape[:2]

        frame_rgb = cv2.cvtColor(
            frame_bgr,
            cv2.COLOR_BGR2RGB,
        )

        detection_result = self._face_mesh.process(frame_rgb)

        if not detection_result.multi_face_landmarks:
            return self._invalid_result(
                reason="no_face",
                processing_started=processing_started,
            )

        face_landmarks = (
            detection_result.multi_face_landmarks[0].landmark
        )

        alignment_points = self._extract_alignment_points(
            face_landmarks,
            image_width,
            image_height,
        )

        if not self._points_are_valid(
            alignment_points,
            image_width,
            image_height,
        ):
            return self._invalid_result(
                reason="invalid_landmarks",
                processing_started=processing_started,
            )

        aligned_face = self._align_face(
            frame_bgr,
            alignment_points,
        )

        if aligned_face is None:
            return self._invalid_result(
                reason="alignment_failed",
                processing_started=processing_started,
            )

        input_tensor = self._prepare_model_input(
            aligned_face,
        )

        inference_started = time.perf_counter()

        outputs = self._session.run(
            self._output_names,
            {
                self._input_name: input_tensor,
            },
        )

        inference_ms = (
            time.perf_counter() - inference_started
        ) * 1000.0

        if not outputs:
            raise RuntimeError(
                "The ONNX model returned no output."
            )

        probabilities = self._convert_scores_to_probabilities(
            outputs[0],
        )

        dominant_index = int(np.argmax(probabilities))

        expression_scores = {
            emotion: round(float(probabilities[index]), 6)
            for index, emotion in enumerate(EMOTION_LABELS)
        }

        processing_ms = (
            time.perf_counter() - processing_started
        ) * 1000.0

        return {
            "face_detected": True,
            "emotion": EMOTION_LABELS[dominant_index],
            "model_confidence": round(
                float(probabilities[dominant_index]),
                6,
            ),
            "scores": expression_scores,
            "inference_ms": round(inference_ms, 3),
            "processing_ms": round(processing_ms, 3),
            "reason": "success",
        }

    @staticmethod
    def _invalid_result(
        reason: str,
        processing_started: float,
    ) -> dict[str, Any]:
        """
        Produce a consistent result when a valid face is unavailable.
        """

        processing_ms = (
            time.perf_counter() - processing_started
        ) * 1000.0

        return {
            "face_detected": False,
            "emotion": "unknown",
            "model_confidence": 0.0,
            "scores": {
                emotion: 0.0
                for emotion in EMOTION_LABELS
            },
            "inference_ms": 0.0,
            "processing_ms": round(processing_ms, 3),
            "reason": reason,
        }

    def close(self) -> None:
        """
        Release MediaPipe resources.
        """

        if self._face_mesh is not None:
            self._face_mesh.close()

    def __enter__(self) -> "FacialExpressionAnalyzer":
        return self

    def __exit__(
        self,
        exception_type: Any,
        exception_value: Any,
        traceback: Any,
    ) -> None:
        self.close()


def test_image(
    image_path: str | Path,
    model_path: str | Path,
) -> int:
    """
    Run one local image through the complete facial-expression pipeline.
    """

    image_path = Path(image_path).resolve()

    if not image_path.exists():
        print(f"Test image was not found: {image_path}")
        return 1

    frame = cv2.imread(str(image_path))

    if frame is None:
        print(
            "OpenCV could not read the image. "
            "Use a JPG, JPEG or PNG file."
        )
        return 1

    try:
        with FacialExpressionAnalyzer(
            model_path=model_path,
        ) as analyzer:
            result = analyzer.analyze_frame(frame)

    except Exception as error:
        print("\nFACIAL-EXPRESSION TEST FAILED")
        print(f"Error: {error}")
        return 1

    print("\nFacial-expression result:")
    print(
        json.dumps(
            result,
            indent=2,
        )
    )

    if not result["face_detected"]:
        print(
            "\nNo usable face was detected. "
            "Try a clear, front-facing and well-lit image."
        )
        return 1

    print("\nSTEP 2 COMPLETED SUCCESSFULLY")
    print(
        "The face was detected, aligned and classified "
        "using the ONNX model."
    )

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Test the CogniTrack facial-expression analyzer "
            "using one image."
        )
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to a clear test image containing one face.",
    )

    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help="Path to the facial-expression ONNX model.",
    )

    arguments = parser.parse_args()

    return test_image(
        image_path=arguments.image,
        model_path=arguments.model,
    )


if __name__ == "__main__":
    raise SystemExit(main())

# ----------------------------------------------------------------------
# Merged from facial_expression_processor.py
# ----------------------------------------------------------------------


from pathlib import Path
from typing import Any

import numpy as np


class FacialExpressionProcessor:
    """
    Connect frame-level facial-expression recognition with
    one-second majority aggregation.

    This class does not control the webcam frame rate.

    The caller is responsible for selecting approximately
    5–6 frames per second and passing them to this processor.
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        expected_frames_per_second: int = 6,
        minimum_valid_frames: int = 3,
        confidence_gap_threshold: float = 0.15,
    ) -> None:
        if expected_frames_per_second < 1:
            raise ValueError(
                "expected_frames_per_second must be at least 1."
            )

        if minimum_valid_frames < 1:
            raise ValueError(
                "minimum_valid_frames must be at least 1."
            )

        if minimum_valid_frames > expected_frames_per_second:
            raise ValueError(
                "minimum_valid_frames cannot be greater than "
                "expected_frames_per_second."
            )

        if confidence_gap_threshold < 0:
            raise ValueError(
                "confidence_gap_threshold cannot be negative."
            )

        self.expected_frames_per_second = (
            expected_frames_per_second
        )

        self.minimum_valid_frames = minimum_valid_frames

        self.confidence_gap_threshold = (
            confidence_gap_threshold
        )

        self._analyzer = FacialExpressionAnalyzer(
            model_path=model_path
        )

        self._current_second: int | None = None

        self._current_aggregator: (
            SecondEmotionAggregator | None
        ) = None

        self._previous_emotion: str | None = None

        self._closed = False

    @property
    def previous_emotion(self) -> str | None:
        """
        Return the most recently confirmed per-second emotion.
        """

        return self._previous_emotion

    @property
    def current_second(self) -> int | None:
        """
        Return the elapsed second currently being collected.
        """

        return self._current_second
    def process_frame_only(
    self,
    frame_bgr: np.ndarray,
    ) -> dict[str, Any]:
        """
        Analyze exactly one selected webcam frame.

        No majority voting.
        No per-second aggregation.
        No database persistence.

        The returned result belongs only to this camera frame.
        """

        self._ensure_open()

        return self._analyzer.analyze_frame(
            frame_bgr
        )



    def process_selected_frame(
        self,
        frame_bgr: np.ndarray,
        elapsed_second: int,
    ) -> dict[str, Any]:
        """
        Process one selected webcam frame.

        Args:
            frame_bgr:
                OpenCV BGR image.

            elapsed_second:
                One-based elapsed second.

                Example:
                    1 = first second
                    2 = second second
                    10 = tenth second

        Returns:
            {
                "frame_result": {...},
                "completed_second": {...} | None
            }

        completed_second is returned when the incoming frame belongs
        to a newer second than the previously collected frames.
        """

        self._ensure_open()

        if elapsed_second < 1:
            raise ValueError(
                "elapsed_second must start from 1."
            )

        completed_second: dict[str, Any] | None = None

        if self._current_second is None:
            self._start_second(elapsed_second)

        elif elapsed_second < self._current_second:
            raise ValueError(
                "Frames must be supplied in chronological order. "
                f"Current second is {self._current_second}, "
                f"but received second {elapsed_second}."
            )

        elif elapsed_second > self._current_second:
            completed_second = self._finalize_current_second()
            self._start_second(elapsed_second)

        frame_result = self._analyzer.analyze_frame(
            frame_bgr
        )

        if self._current_aggregator is None:
            raise RuntimeError(
                "The per-second aggregator was not initialized."
            )

        self._current_aggregator.add_result(
            frame_result
        )

        return {
            "frame_result": frame_result,
            "completed_second": completed_second,
        }

    def _start_second(
        self,
        elapsed_second: int,
    ) -> None:
        """
        Create a fresh aggregator for a new elapsed second.
        """

        self._current_second = elapsed_second

        self._current_aggregator = (
            SecondEmotionAggregator(
                elapsed_second=elapsed_second,
                expected_frames=(
                    self.expected_frames_per_second
                ),
                confidence_gap_threshold=(
                    self.confidence_gap_threshold
                ),
            )
        )

    def _finalize_current_second(
        self,
    ) -> dict[str, Any] | None:
        """
        Finalize the current second and update temporal state.
        """

        if (
            self._current_aggregator is None
            or self._current_second is None
        ):
            return None

        previous_before_decision = self._previous_emotion

        second_result = (
            self._current_aggregator.finalize(
                previous_emotion=(
                    previous_before_decision
                ),
                minimum_valid_frames=(
                    self.minimum_valid_frames
                ),
            )
        )

        selected_emotion = str(
            second_result.get(
                "emotion",
                "unknown",
            )
        ).strip().lower()

        expression_is_valid = (
            selected_emotion in VALID_EMOTIONS
        )

        emotion_changed = (
            expression_is_valid
            and previous_before_decision is not None
            and selected_emotion
            != previous_before_decision
        )

        second_result["previous_emotion"] = (
            previous_before_decision
        )

        second_result["emotion_changed"] = (
            emotion_changed
        )

        if expression_is_valid:
            self._previous_emotion = (
                selected_emotion
            )

        return second_result

    def flush(
        self,
    ) -> dict[str, Any] | None:
        """
        Finalize the currently collected second.

        Use this when:
        - testing stops,
        - a question ends,
        - a session ends,
        - or the processor is reset.
        """

        self._ensure_open()

        completed_second = (
            self._finalize_current_second()
        )

        self._current_second = None
        self._current_aggregator = None

        return completed_second

    def reset(
        self,
    ) -> dict[str, Any] | None:
        """
        Finalize the current second and clear temporal state.

        This will later be used when moving to a new question.
        """

        completed_second = self.flush()

        self._previous_emotion = None

        return completed_second

    def close(
        self,
    ) -> None:
        """
        Release MediaPipe and model-related resources.
        """

        if self._closed:
            return

        self._analyzer.close()

        self._current_second = None
        self._current_aggregator = None
        self._previous_emotion = None
        self._closed = True

    def _ensure_open(
        self,
    ) -> None:
        """
        Prevent use after the processor has been closed.
        """

        if self._closed:
            raise RuntimeError(
                "FacialExpressionProcessor is closed."
            )

    def __enter__(
        self,
    ) -> "FacialExpressionProcessor":
        return self

    def __exit__(
        self,
        exception_type: Any,
        exception_value: Any,
        traceback: Any,
    ) -> None:
        self.close()

# ----------------------------------------------------------------------
# Merged from facial_expression_aggregator.py
# ----------------------------------------------------------------------


from collections import Counter
from dataclasses import dataclass, field
from typing import Any


VALID_EMOTIONS = {
    "angry",
    "disgust",
    "fearful",
    "happy",
    "neutral",
    "sad",
    "surprised",
}


@dataclass
class SecondEmotionAggregator:
    """
    Collect facial-expression predictions from selected frames
    belonging to one second and calculate one final expression.

    Tie-handling priority:
    1. Clear majority
    2. Last two valid frames
    3. Meaningful confidence-total difference
    4. Previous second's confirmed emotion
    5. Unknown
    """

    elapsed_second: int
    expected_frames: int = 6
    confidence_gap_threshold: float = 0.15

    _emotions: list[str] = field(default_factory=list)
    _model_confidences: list[float] = field(default_factory=list)
    _total_frames: int = 0
    _invalid_reasons: Counter[str] = field(
        default_factory=Counter
    )

    def __post_init__(self) -> None:
        if self.elapsed_second < 0:
            raise ValueError(
                "elapsed_second cannot be negative."
            )

        if self.expected_frames < 1:
            raise ValueError(
                "expected_frames must be at least 1."
            )

        if self.confidence_gap_threshold < 0:
            raise ValueError(
                "confidence_gap_threshold cannot be negative."
            )

    def add_result(
        self,
        result: dict[str, Any],
    ) -> None:
        """
        Add the result from one selected webcam frame.

        Valid frames participate in expression voting.

        No-face and invalid frames are counted but do not vote.
        """

        self._total_frames += 1

        face_detected = bool(
            result.get("face_detected", False)
        )

        emotion = str(
            result.get("emotion", "unknown")
        ).strip().lower()

        if not face_detected or emotion not in VALID_EMOTIONS:
            reason = str(
                result.get("reason", "invalid_frame")
            ).strip()

            if not reason:
                reason = "invalid_frame"

            self._invalid_reasons[reason] += 1
            return

        confidence = self._safe_confidence(
            result.get("model_confidence", 0.0)
        )

        self._emotions.append(emotion)
        self._model_confidences.append(confidence)

    @staticmethod
    def _safe_confidence(value: Any) -> float:
        """
        Convert model confidence to a safe value between 0 and 1.
        """

        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return 0.0

        return max(
            0.0,
            min(1.0, confidence),
        )

    @property
    def valid_frames(self) -> int:
        """
        Number of frames that produced a valid expression.
        """

        return len(self._emotions)

    @property
    def total_frames(self) -> int:
        """
        Total number of selected frames received.
        """

        return self._total_frames

    def finalize(
        self,
        previous_emotion: str | None = None,
        minimum_valid_frames: int = 3,
    ) -> dict[str, Any]:
        """
        Calculate the final expression for this second.

        Args:
            previous_emotion:
                Confirmed expression from the previous second.

            minimum_valid_frames:
                Minimum number of valid facial predictions required
                to assign an expression to the current second.
        """

        if minimum_valid_frames < 1:
            raise ValueError(
                "minimum_valid_frames must be at least 1."
            )

        normalized_previous = self._normalize_previous_emotion(
            previous_emotion
        )

        vote_counts = Counter(self._emotions)

        if self.valid_frames < minimum_valid_frames:
            return {
                "elapsed_second": self.elapsed_second,
                "emotion": "unknown",
                "majority_confidence": 0.0,
                "average_model_confidence": 0.0,
                "valid_frames": self.valid_frames,
                "total_frames": self.total_frames,
                "expected_frames": self.expected_frames,
                "vote_counts": dict(vote_counts),
                "invalid_reasons": dict(
                    self._invalid_reasons
                ),
                "result_status": (
                    "insufficient_valid_frames"
                ),
                "tie_break_reason": None,
            }

        (
            selected_emotion,
            selected_votes,
            tie_break_reason,
        ) = self._select_emotion(
            vote_counts=vote_counts,
            previous_emotion=normalized_previous,
        )

        majority_confidence = (
            selected_votes / self.valid_frames
            if selected_emotion != "unknown"
            else 0.0
        )

        selected_confidences = [
            confidence
            for emotion, confidence in zip(
                self._emotions,
                self._model_confidences,
            )
            if emotion == selected_emotion
        ]

        average_model_confidence = (
            sum(selected_confidences)
            / len(selected_confidences)
            if selected_confidences
            else 0.0
        )

        return {
            "elapsed_second": self.elapsed_second,
            "emotion": selected_emotion,
            "majority_confidence": round(
                majority_confidence,
                4,
            ),
            "average_model_confidence": round(
                average_model_confidence,
                6,
            ),
            "valid_frames": self.valid_frames,
            "total_frames": self.total_frames,
            "expected_frames": self.expected_frames,
            "vote_counts": dict(vote_counts),
            "invalid_reasons": dict(
                self._invalid_reasons
            ),
            "result_status": (
                "success"
                if selected_emotion != "unknown"
                else "ambiguous_tie"
            ),
            "tie_break_reason": tie_break_reason,
        }

    @staticmethod
    def _normalize_previous_emotion(
        previous_emotion: str | None,
    ) -> str | None:
        """
        Validate and normalize the previous second's expression.
        """

        if not isinstance(previous_emotion, str):
            return None

        normalized = previous_emotion.strip().lower()

        if normalized not in VALID_EMOTIONS:
            return None

        return normalized

    def _select_emotion(
        self,
        vote_counts: Counter[str],
        previous_emotion: str | None,
    ) -> tuple[str, int, str]:
        """
        Select the final expression for one second.

        Priority:
        1. Clear majority
        2. Last two valid frames
        3. Meaningful confidence-total difference
        4. Previous second's confirmed expression
        5. Unknown
        """

        highest_vote_count = max(
            vote_counts.values()
        )

        tied_emotions = [
            emotion
            for emotion, votes in vote_counts.items()
            if votes == highest_vote_count
        ]

        # 1. A clear majority exists.
        if len(tied_emotions) == 1:
            return (
                tied_emotions[0],
                highest_vote_count,
                "clear_majority",
            )

        # 2. In a tie, use the last two valid frames
        # only when both frames agree.
        if len(self._emotions) >= 2:
            second_last_emotion = self._emotions[-2]
            last_emotion = self._emotions[-1]

            if (
                second_last_emotion == last_emotion
                and last_emotion in tied_emotions
            ):
                return (
                    last_emotion,
                    highest_vote_count,
                    "last_two_valid_frames",
                )

        # 3. Calculate confidence totals for tied expressions.
        confidence_totals = {
            emotion: sum(
                confidence
                for frame_emotion, confidence in zip(
                    self._emotions,
                    self._model_confidences,
                )
                if frame_emotion == emotion
            )
            for emotion in tied_emotions
        }

        ranked_emotions = sorted(
            confidence_totals.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        highest_emotion, highest_confidence = (
            ranked_emotions[0]
        )

        second_highest_confidence = (
            ranked_emotions[1][1]
        )

        confidence_gap = (
            highest_confidence
            - second_highest_confidence
        )

        # Select the confidence winner only when
        # the difference is meaningful.
        if (
            confidence_gap
            >= self.confidence_gap_threshold
        ):
            return (
                highest_emotion,
                highest_vote_count,
                "confidence_total",
            )

        # 4. Confidence difference is small.
        # Retain the previous expression for stability.
        if previous_emotion in tied_emotions:
            return (
                previous_emotion,
                highest_vote_count,
                "previous_second_emotion",
            )

        # 5. No reliable decision can be made.
        return (
            "unknown",
            highest_vote_count,
            "unresolved_tie",
        )

# ----------------------------------------------------------------------
# Merged from facial_expression_service.py
# ----------------------------------------------------------------------


import base64
import binascii
import threading
from dataclasses import dataclass, field
from typing import Any

try:
    import cv2
    import numpy as np
    from facial_expression_processor import FacialExpressionProcessor

    FACIAL_IMPORT_ERROR: ImportError | None = None
except ImportError as exc:  # Keep the rest of the API available if FER is unavailable.
    cv2 = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]
    FacialExpressionProcessor = Any  # type: ignore[assignment,misc]
    FACIAL_IMPORT_ERROR = exc


@dataclass
class _ParticipantFacialState:
    """Temporal facial-expression state for one participant."""

    question_id: str
    task_number: int
    processor: Any
    segment_start_second: int | None = None
    question_closed: bool = False
    lock: Any = field(default_factory=threading.Lock, repr=False)


class FacialExpressionService:
    """Participant-aware facial-expression processing service.

    The browser sends approximately six selected frames per second from the
    same webcam stream used by eye tracking. This service keeps those frames
    separated by participant and question, runs ONNX inference, aggregates the
    frame predictions into one result per elapsed second, and emits sparse
    storage events instead of storing every frame.
    """

    def __init__(
        self,
        expected_frames_per_second: int = 6,
        minimum_valid_frames: int = 3,
        confidence_gap_threshold: float = 0.15,
    ) -> None:
        self.expected_frames_per_second = expected_frames_per_second
        self.minimum_valid_frames = minimum_valid_frames
        self.confidence_gap_threshold = confidence_gap_threshold
        self._states: dict[str, _ParticipantFacialState] = {}
        self._registry_lock = threading.RLock()

    def _new_processor(self) -> Any:
        if FACIAL_IMPORT_ERROR is not None:
            raise ImportError(
                "Facial-expression dependencies are not installed"
            ) from FACIAL_IMPORT_ERROR

        return FacialExpressionProcessor(
            expected_frames_per_second=self.expected_frames_per_second,
            minimum_valid_frames=self.minimum_valid_frames,
            confidence_gap_threshold=self.confidence_gap_threshold,
        )

    @staticmethod
    def _decode(data_url: str) -> Any:
        if cv2 is None or np is None:
            raise ImportError("Facial-expression dependencies are not installed")

        if not isinstance(data_url, str) or not data_url.strip():
            raise ValueError("The facial-expression image is empty")

        encoded = data_url.split(",", 1)[-1]
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("The facial-expression image is not valid base64") from exc

        frame = cv2.imdecode(
            np.frombuffer(raw, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        if frame is None:
            raise ValueError("Invalid facial-expression camera frame")
        return frame

    def _get_or_create_state(
        self,
        participant_id: str,
        question_id: str,
        task_number: int,
    ) -> _ParticipantFacialState:
        with self._registry_lock:
            state = self._states.get(participant_id)
            if state is None:
                state = _ParticipantFacialState(
                    question_id=question_id,
                    task_number=task_number,
                    processor=self._new_processor(),
                )
                self._states[participant_id] = state
            return state

    @staticmethod
    def _attach_question(
        result: dict[str, Any] | None,
        question_id: str,
        task_number: int,
    ) -> dict[str, Any] | None:
        if result is None:
            return None
        return {
            "question_id": question_id,
            "task_number": task_number,
            **result,
        }

    @staticmethod
    def _ignored_frame(reason: str) -> dict[str, Any]:
        return {
            "face_detected": False,
            "emotion": "unknown",
            "model_confidence": 0.0,
            "scores": {},
            "inference_ms": 0.0,
            "processing_ms": 0.0,
            "reason": reason,
        }

    @staticmethod
    def _build_storage_event(
        state: _ParticipantFacialState,
        second_result: dict[str, Any] | None,
        terminal_reason: str | None = None,
    ) -> dict[str, Any] | None:
        """Convert one completed second into a sparse storage event."""

        if not second_result:
            return None

        result_status = str(second_result.get("result_status", "")).strip().lower()
        emotion = str(second_result.get("emotion", "unknown")).strip().lower() or "unknown"

        elapsed_second = int(second_result.get("elapsed_second", 0) or 0)
        if elapsed_second < 1:
            return None

        previous_value = second_result.get("previous_emotion")
        previous_emotion = (
            str(previous_value).strip().lower() if previous_value else ""
        )
        emotion_changed = bool(second_result.get("emotion_changed", False))

        previous_segment_start_second: int | None = None
        previous_segment_end_second: int | None = None

        if state.segment_start_second is None:
            state.segment_start_second = elapsed_second

        if emotion_changed:
            previous_segment_start_second = state.segment_start_second
            previous_segment_end_second = max(1, elapsed_second - 1)
            state.segment_start_second = elapsed_second

        if terminal_reason:
            record_reason: str | None = terminal_reason
        elif not previous_emotion:
            record_reason = "initial"
        elif emotion_changed:
            record_reason = "emotion_change"
        elif elapsed_second % 10 == 0:
            record_reason = "interval_checkpoint"
        else:
            record_reason = None

        if record_reason is None:
            if result_status != "success" or emotion == "unknown":
                record_reason = result_status or "unknown_emotion"
            else:
                record_reason = "second_complete"

        return {
            "question_id": state.question_id,
            "task_number": state.task_number,
            "elapsed_second": elapsed_second,
            "emotion": emotion,
            "confidence": float(second_result.get("majority_confidence", 0.0) or 0.0),
            "average_model_confidence": float(
                second_result.get("average_model_confidence", 0.0) or 0.0
            ),
            "valid_frames": int(second_result.get("valid_frames", 0) or 0),
            "total_frames": int(second_result.get("total_frames", 0) or 0),
            "expected_frames": int(
                second_result.get("expected_frames", 6) or 6
            ),
            "record_reason": record_reason,
            "previous_emotion": previous_emotion,
            "segment_start_second": state.segment_start_second,
            "segment_end_second": elapsed_second,
            "previous_segment_start_second": (
                previous_segment_start_second
                if previous_segment_start_second is not None
                else ""
            ),
            "previous_segment_end_second": (
                previous_segment_end_second
                if previous_segment_end_second is not None
                else ""
            ),
            "emotion_changed": emotion_changed,
            "tie_break_reason": str(second_result.get("tie_break_reason") or ""),
            "result_status": result_status,
        }

    def _finalize_state(
        self,
        state: _ParticipantFacialState,
        terminal_reason: str,
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        """Finalize and reset the active question for a participant."""

        completed = state.processor.reset()
        attached = self._attach_question(
            completed,
            state.question_id,
            state.task_number,
        )
        event = self._build_storage_event(
            state=state,
            second_result=attached,
            terminal_reason=terminal_reason,
        )
        state.segment_start_second = None
        state.question_closed = True
        return attached, [event] if event else []

    def process_frame(
        self,
        participant_id: str,
        question_id: str,
        task_number: int,
        frame_id: int,
        elapsed_ms: int,
        elapsed_second: int,
        image_data_url: str,
    ) -> dict[str, Any]:
        """Process one selected webcam frame for a participant question."""

        participant_id = participant_id.strip()
        question_id = question_id.strip()
        if not participant_id:
            raise ValueError("participant_id is required")
        if not question_id:
            raise ValueError("question_id is required")
        if task_number < 1:
            raise ValueError("task_number must start from 1")
        if frame_id < 1:
            raise ValueError(
                "frame_id must start from 1"
        )

        if elapsed_ms < 0:
            raise ValueError(
                "elapsed_ms cannot be negative"
        )
        if elapsed_second < 1:
            raise ValueError("elapsed_second must start from 1")

        frame = self._decode(image_data_url)
        state = self._get_or_create_state(
            participant_id=participant_id,
            question_id=question_id,
            task_number=task_number,
        )

        with state.lock:
            storage_events: list[dict[str, Any]] = []
            completed_previous_question: dict[str, Any] | None = None

            if state.question_id != question_id:
                if not state.question_closed:
                    completed_previous_question, old_events = self._finalize_state(
                        state,
                        terminal_reason="question_end",
                    )
                    storage_events.extend(old_events)

                state.question_id = question_id
                state.task_number = task_number
                state.segment_start_second = None
                state.question_closed = False

            elif state.question_closed:
                # A request that was already in flight when the answer was
                # submitted must not reopen the completed question.
                return {
                    "active_question_id": question_id,
                    "frame_result": self._ignored_frame("question_closed"),
                    "completed_second": None,
                    "completed_previous_question": None,
                    "storage_events": [],
                }

            state.task_number = task_number
            # ---------------------------------------------------------
            # FRAME-LEVEL FACIAL ANALYSIS
            # ---------------------------------------------------------

            processed = state.processor.process_selected_frame(
                frame_bgr=frame,
                elapsed_second=elapsed_second,
            )
            frame_result = processed.get("frame_result") or {}
            completed_second = processed.get("completed_second")
            if completed_second:
                completed_second = self._attach_question(
                    completed_second,
                    state.question_id,
                    state.task_number,
                )
                event = self._build_storage_event(
                    state=state,
                    second_result=completed_second,
                )
                if event:
                    storage_events.append(event)

            return {
                "active_question_id": question_id,

                # Synchronization metadata.
                "frame_id": frame_id,
                "elapsed_ms": elapsed_ms,
                "elapsed_second": elapsed_second,

                # Raw prediction for THIS frame.
                "frame_result": frame_result,

                "completed_second": completed_second,

                "completed_previous_question": (
                    completed_previous_question
                ),

                "storage_events": storage_events,
            }
            

            

    def flush_question(
        self,
        participant_id: str,
        question_id: str,
    ) -> dict[str, Any] | None:
        """Finalize the current second when an answer is submitted."""

        with self._registry_lock:
            state = self._states.get(participant_id)
        if state is None:
            return None

        with state.lock:
            if state.question_id != question_id or state.question_closed:
                return None

            completed, storage_events = self._finalize_state(
                state,
                terminal_reason="question_end",
            )
            return {
                "question_id": state.question_id,
                "task_number": state.task_number,
                "completed_second": completed,
                "storage_events": storage_events,
            }

    def forget_participant(
        self,
        participant_id: str,
    ) -> dict[str, Any] | None:
        """Finalize the active question, close resources, and remove state."""

        with self._registry_lock:
            state = self._states.pop(participant_id, None)
        if state is None:
            return None

        with state.lock:
            completed: dict[str, Any] | None = None
            storage_events: list[dict[str, Any]] = []
            try:
                if not state.question_closed:
                    completed, storage_events = self._finalize_state(
                        state,
                        terminal_reason="session_end",
                    )
            finally:
                state.processor.close()

            return {
                "question_id": state.question_id,
                "task_number": state.task_number,
                "completed_second": completed,
                "storage_events": storage_events,
            }


facial_expression_service = FacialExpressionService(
    expected_frames_per_second=6,
    minimum_valid_frames=3,
    confidence_gap_threshold=0.15,
)


# ----------------------------------------------------------------------
# Merged from eye_tracking.py
# ----------------------------------------------------------------------

"""Independent real-time pupil and eye-tracking pipeline.

It uses MediaPipe's refined iris landmarks.
"""




class EyeTrackingAnalyzer(VisionAnalyzer):
    """Eye-only adapter reused by the browser/API eye stream.

    ``VisionAnalyzer`` already implements the calibrated iris, EAR, blink,
    PERCLOS, gaze, saccade, and head-pose algorithm from the supplied pupil
    tracker.
    """

    def analyze_eye_frame(self, participant_id: str, data_url: str, calibration_point: int | None = None):
        return self.analyze(participant_id, data_url, calibration_point)


eye_tracking_analyzer = EyeTrackingAnalyzer()
