print('start')
import cv2, numpy as np
print('import')
from backend.vision_analysis import FacialExpressionAnalyzer
print('imported class')
analyzer = FacialExpressionAnalyzer()
print('initialized')
frame = np.zeros((480, 640, 3), dtype=np.uint8)
print('analyzing')
res = analyzer.analyze_frame(frame)
print(res)
