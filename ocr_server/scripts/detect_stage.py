"""①차량탐지 + ③번호판탐지 (torch/ultralytics 전용, paddle import 절대 금지).

torch와 paddle은 같은 프로세스에서 같이 못 씀(CUDA DLL 충돌 확인됨: cublas64_12.dll /
cudnn_cnn64_9.dll 이름이 겹쳐서 어느 쪽을 먼저 import해도 나머지가 깨짐). 그래서 탐지(torch)와
인식(paddle, recognize_stage.py)을 완전히 별도 프로세스로 분리했다. 이 파일은 절대로
paddle/ppocr을 import하면 안 된다.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
VEHICLE_MODEL_PATH = ROOT / "yolo11n.pt"
PLATE_MODEL_PATH = ROOT / "runs" / "detect" / "plate_detector_v8s-2" / "weights" / "best.pt"
# CPU용 - best.pt를 OpenVINO로 변환한 모델(입력 512x960 고정). Intel CPU에서 PyTorch보다 훨씬 빠름.
# 만드는 법(torch 환경, 한 번만): YOLO("runs/detect/plate_detector_v8s-2/weights/best.pt").export(format="openvino", imgsz=[512, 960])
# 입력 크기를 자유(dynamic=True)로 하면 실행 중 수 초씩 멈추는 현상이 있어 고정 크기로 변환함
PLATE_OPENVINO_PATH = PLATE_MODEL_PATH.parent / "best_openvino_model"

# COCO 클래스 인덱스: 2=car, 5=bus, 7=truck
VEHICLE_CLASS_IDS = {2, 5, 7}

VEHICLE_CONF = 0.4
PLATE_CONF = 0.3
PLATE_CROP_MARGIN = 4  # px, 번호판 박스 경계가 타이트할 경우 대비 여유

# 한국 번호판 형식: 숫자2~3 + 한글1 + 숫자4 (신형/구형), 앞에 지역명(1~2글자)이 붙는 구형도 허용.
_PLATE_FORMAT_RE = re.compile(r"^([가-힣]{1,2})?\d{2,3}[가-힣]\d{4}$")


def is_valid_plate_format(text: str) -> bool:
    return bool(_PLATE_FORMAT_RE.match(text))


def imread_unicode(path) -> np.ndarray | None:
    """cv2.imread는 Windows에서 한글 등 비ASCII 경로를 못 읽는 버그가 있어 우회한다."""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path, image: np.ndarray, ext: str = ".jpg") -> bool:
    """cv2.imwrite도 한글 등 비ASCII 경로에서 True를 반환하면서 실제로는 파일을
    안 만드는 경우가 확인됨(imwrite 자체가 신뢰 불가). cv2.imencode(메모리 인코딩)
    + 파이썬 기본 파일쓰기로 우회한다."""
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        return False
    buf.tofile(str(path))
    return os.path.exists(str(path))


class VehicleDetector:
    def __init__(
        self,
        model_path: Path = VEHICLE_MODEL_PATH,
        conf: float = VEHICLE_CONF,
        device: str | None = None,
    ):
        self.model = YOLO(str(model_path))
        self.conf = conf
        self.device = device  # None이면 ultralytics 기본(가능하면 GPU) 동작

    def detect(self, image_bgr: np.ndarray) -> list[tuple[tuple[int, int, int, int], int]]:
        result = self.model.predict(
            image_bgr, conf=self.conf, device=self.device, verbose=False
        )[0]
        boxes = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in VEHICLE_CLASS_IDS:
                continue
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            boxes.append(((x1, y1, x2, y2), cls_id))
        return boxes


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


class PlateDetector:
    # ③ 학습은 imgsz=1280(원거리 CCTV 스케일)으로 했는데, 검증용 실사진처럼 가까이서
    # 찍혀 번호판이 크게 나온 경우 1280에서는 전혀 탐지가 안 되는 현상을 확인함(학습
    # 데이터가 전부 초소형 번호판이라 큰 물체를 담당하는 탐지 레이어가 훈련이 안 됨).
    # 재학습 없이 여러 imgsz로 나눠서 추론(이미지 피라미드)하면, 그중 어느 스케일에서는
    # 번호판이 학습 때 본 크기와 비슷해져서 탐지가 되는 걸 확인해서 이 방식으로 우회함
    # (300장 샘플 검증 결과 탐지율 98.3%, runs/detect/plate_detector-2 55epoch 체크포인트 기준).
    IMGSZ_CANDIDATES = (1280, 640, 480, 384, 320)
    DEDUP_IOU_THRESHOLD = 0.5

    def __init__(
        self,
        model_path: Path = PLATE_MODEL_PATH,
        conf: float = PLATE_CONF,
        imgsz_candidates: tuple[int, ...] = IMGSZ_CANDIDATES,
        device: str | None = None,
    ):
        self.model = YOLO(str(model_path))
        self.conf = conf
        self.imgsz_candidates = imgsz_candidates
        self.device = device  # None이면 ultralytics 기본(가능하면 GPU) 동작

    def detect(self, image_bgr: np.ndarray) -> list[tuple[float, tuple[int, int, int, int]]]:
        """반환값: [(confidence, (x1,y1,x2,y2)), ...] confidence 내림차순 정렬."""
        candidates: list[tuple[float, tuple[int, int, int, int]]] = []
        for imgsz in self.imgsz_candidates:
            result = self.model.predict(
                image_bgr, conf=self.conf, imgsz=imgsz, device=self.device, verbose=False
            )[0]
            for box in result.boxes:
                xyxy = tuple(int(v) for v in box.xyxy[0].tolist())
                candidates.append((float(box.conf[0]), xyxy))

        # 여러 스케일에서 같은 번호판이 중복 검출될 수 있어, confidence 높은 순으로
        # 정렬한 뒤 이미 채택된 박스와 많이 겹치는(IoU 높은) 건 걸러낸다 (cross-scale NMS).
        candidates.sort(key=lambda c: -c[0])
        kept: list[tuple[float, tuple[int, int, int, int]]] = []
        for score, box in candidates:
            if any(_iou(box, k[1]) > self.DEDUP_IOU_THRESHOLD for k in kept):
                continue
            kept.append((score, box))
        return kept


def crop_with_margin(
    image_bgr: np.ndarray,
    box: tuple[int, int, int, int],
    margin: int = PLATE_CROP_MARGIN,
) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = box
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(w, x2 + margin)
    y2 = min(h, y2 + margin)
    return image_bgr[y1:y2, x1:x2]


def match_vehicle(plate_box, vehicle_boxes):
    px1, py1, px2, py2 = plate_box
    pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
    for (vx1, vy1, vx2, vy2), cls_id in vehicle_boxes:
        if vx1 <= pcx <= vx2 and vy1 <= pcy <= vy2:
            return (vx1, vy1, vx2, vy2), cls_id
    return None, None
