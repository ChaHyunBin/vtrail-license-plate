"""CCTV 실시간 번호판 탐지 스트리밍 서버 (port 8010)

기능: CCTV 영상을 재생하면서 번호판을 탐지(YOLO)하고, 읽은 번호를 수배차량 목록과 대조해
      발견된 차량을 알려줌. 영상은 MJPEG로 브라우저에 보냄.

연결:
- client/cctv.vue    : /stream(영상), /matches(발견 목록), /control(일시정지·이동)을 카메라별로 호출.
                       대조 목록(targets)도 여기서 받음
- live_ocr_worker.py : 번호판 글자 인식 담당(별도 프로세스, 이 서버를 실행하면 ocr_env로 자동 실행됨).
                       _live_pending/에 크롭을 넣으면 _live_results.csv에 결과를 씀
- fa_server          : DB 저장은 이 서버가 안 함. cctv.vue가 발견 결과를 각자 PC의
                       fa_server(POST /cctv/detection)로 보내서 저장

실행: py -3.11 scripts/live_stream_server.py (글자 인식 워커도 같이 켜짐, Ctrl+C로 둘 다 종료)
"""
import importlib.util
import re
import subprocess
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw, ImageFont

from detect_stage import (
    PLATE_OPENVINO_PATH, PlateDetector, crop_with_margin, imwrite_unicode, is_valid_plate_format,
)

ROOT = Path(__file__).resolve().parent.parent
PENDING_DIR = ROOT / "output" / "_live_pending"      # OCR 요청(번호판 크롭) -> live_ocr_worker가 가져감
RESULTS_PATH = ROOT / "output" / "_live_results.csv"  # OCR 결과 <- live_ocr_worker가 씀
CAPTURES_DIR = ROOT / "output" / "captures"
CAPTURES_DIR.mkdir(parents=True, exist_ok=True)

DISPLAY_SCALE = 1280 / 1920       # 1920x1080 영상을 1280x720으로 줄여서 처리/송출
DETECT_EVERY_N_FRAMES = 5         # 번호판 탐지 주기(프레임)
HOLD_SECONDS = 0.5                # 이 시간 동안 안 보이면 추적 중인 박스 삭제
# 박스를 화면에 그리고 따라가게 하는 최대 프레임(마지막 탐지 기준). 탐지가 별도 스레드라 결과가 2~3프레임 늦게
# 오므로 탐지 주기의 2배(10)로는 다음 결과 직전에 잠깐 사라져 깜빡임 -> 3배(0.5초)
BOX_SHOW_FRAMES = DETECT_EVERY_N_FRAMES * 3
MATCH_DIST_RATIO = 1.5            # 새 박스를 기존 차량과 같은 차로 볼 최대 중심 거리(번호판 너비 배수)
MATCH_DX_RATIO = 0.5              # 그중 좌우로는 이만큼만 허용(차는 주로 위아래로 움직임. 옆 반사띠 등 오연결 방지)
FOLLOW_MIN_SCORE = 0.6            # 탐지 사이 프레임에서 번호판 모양으로 박스를 옮길 최소 유사도(0~1)
REOCR_COOLDOWN_SECONDS = 0.15     # 같은 박스를 다시 OCR 요청하는 최소 간격(0.15 = 탐지할 때마다, 약 0.17초)
# 최근 HISTORY_SIZE번을 모두 같은 번호로 읽은(확정된) 박스는 이 간격으로만 다시 읽음.
# 이미 아는 번호판을 계속 읽느라 새 번호판 OCR이 밀리는 것 방지(카메라 2대일 때 특히)
CONFIRMED_COOLDOWN_SECONDS = 1.0
KEEP_CHARS_RE = re.compile(r"[^0-9가-힣]")

# 발견 확정 조건: 같은 차량을 읽은 최근 3번 중 2번 이상 같은 번호일 때만 확정.
# 한 번의 오인식(3<->8, 보<->부 등)이 수배 번호와 우연히 같아져 저장/알림되는 걸 막음
HISTORY_SIZE = 3
CONFIRM_MIN_HITS = 2

# 카메라 설치 위치(노드) - 발견 결과에 담겨 fa_server가 detection_log.node_id로 저장, 지도에 표시됨.
# 팀 DB가 바뀌면 fa_server VehicleScoringService.predict_next_node(직전, 마지막 노드) 후보 중에서 고를 것
DEMO_NODE_ID = "1180032700"       # 앞 카메라 - 여의상류IC
BACK_NODE_ID = "1140008502"       # 뒤 카메라 - 한국통신목동지점

# 카메라별 설정: 재생 영상 / 탐지 영역(ROI) / 설치 노드 / 재생 속도(speed, 없으면 1.0). /stream?camera= 로 선택
# roi_poly: 도로 원근에 맞춘 사다리꼴(1280x720 기준, 좌상-우상-우하-좌하). None이면 화면 전체 탐지
CAMERA_CONFIG = {
    "front": {
        "video": ROOT / "output" / "video_watch" / "front_clip25s.mp4",
        "roi_poly": [(430, 195), (1090, 195), (1270, 720), (270, 720)],
        "node_id": DEMO_NODE_ID,
    },
    "back": {
        # 원본 back.mp4 2:37.75~3:02.75, 30fps(용량 100MB 이하). 12초쯤 시연용 수배차량 통과, 차량이 위로 멀어지는 방향
        "video": ROOT / "output" / "video_watch" / "back_clip25s.mp4",
        "roi_poly": [(300, 150), (1080, 150), (1280, 720), (60, 720)],
        "node_id": BACK_NODE_ID,
        "speed": 0.9,  # 재생 속도(1.0 = 원래 속도). 차가 빠르게 멀어져서 살짝 느리게
    },
}
DEFAULT_CAMERA = "front"

_FONT = ImageFont.truetype(r"C:\Windows\Fonts\malgunbd.ttf", 26)
JPEG_QUALITY = 80  # 송출 화질(기본 95). 80이면 크기 절반·변환 시간 약 30%↓, 눈으로는 차이 거의 없음

# 번호 글자(한글) 그림 캐시 - (글자, 색)마다 한 번만 그려 두고 프레임에 붙임.
# 매 프레임 전체를 PIL로 바꿔 그리면 약 6ms 걸려서, 작은 글자 그림만 재사용
_LABEL_CACHE: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}


def label_sprite(text: str, color_bgr: tuple) -> tuple[np.ndarray, np.ndarray]:
    """(글자 BGR 그림, 투명도 0~1) - 처음 쓰는 (글자, 색)만 PIL로 그림"""
    key = (text, color_bgr)
    if key not in _LABEL_CACHE:
        if len(_LABEL_CACHE) > 512:  # 점수가 붙은 글자는 종류가 많아서 너무 쌓이면 비움
            _LABEL_CACHE.clear()
        _, _, x1, y1 = _FONT.getbbox(text)
        img = Image.new("RGBA", (x1 + 2, y1 + 2), (0, 0, 0, 0))
        ImageDraw.Draw(img).text((0, 0), text, font=_FONT, fill=(color_bgr[2], color_bgr[1], color_bgr[0], 255))
        rgba = np.array(img)
        _LABEL_CACHE[key] = (rgba[:, :, 2::-1].astype(np.float32), rgba[:, :, 3:4].astype(np.float32) / 255)
    return _LABEL_CACHE[key]


def draw_label(frame: np.ndarray, text: str, org: tuple, color_bgr: tuple):
    """프레임의 org(왼쪽 위) 위치에 글자 그림을 투명도대로 합성(화면 밖으로 나간 부분은 잘라냄)"""
    bgr, alpha = label_sprite(text, color_bgr)
    x, y = org
    h = min(bgr.shape[0], frame.shape[0] - y)
    w = min(bgr.shape[1], frame.shape[1] - x)
    if h <= 0 or w <= 0:
        return
    roi = frame[y:y + h, x:x + w]
    a = alpha[:h, :w]
    roi[:] = (roi * (1 - a) + bgr[:h, :w] * a).astype(np.uint8)

app = FastAPI()

# cctv.vue(localhost:9000)에서 /matches 등을 axios로 호출할 수 있게 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:9000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/captures", StaticFiles(directory=str(CAPTURES_DIR)), name="captures")


# 번호판 비교용 - 숫자/한글만 남김("12가 3456" -> "12가3456"). cctv.vue normalizePlate와 같은 기준
def normalize(text: str) -> str:
    return KEEP_CHARS_RE.sub("", text)


def iou(a, b):
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


def center_dist(a, b):
    acx, acy = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    bcx, bcy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5


# 새로 탐지된 박스가 이미 추적 중인 차량(박스)과 같은 차량인지 찾음(겹침/중심거리 기준)
def find_match(new_box, tracked, taken):
    # taken: 이번 탐지에서 이미 다른 박스와 이어진 차량 - 한 차량에 박스 두 개가 붙지 않게 제외
    candidates = {bid: t for bid, t in tracked.items() if bid not in taken}

    # 비교 위치: 지금 그리는 위치(box, follow_boxes가 옮김) + 마지막으로 탐지된 위치(det_box).
    # 탐지 결과는 탐지 스레드 때문에 2~3프레임 전 위치라 지금 위치하고만 비교하면 같은 차인데 어긋나서 박스가 갈라짐
    def places(t):
        return [t["box"], t["det_box"]] if t.get("det_box") else [t["box"]]

    best_id, best_score = None, 0.0
    for bid, t in candidates.items():
        score = max(iou(new_box, b) for b in places(t))
        if score >= 0.35 and score > best_score:
            best_id, best_score = bid, score
    if best_id is not None:
        return best_id
    # 안 겹치면 가장 가까운 박스 - 빠르게 멀어지는 차(뒤 카메라)도 같은 차로 이어지게 너비의 1.5배까지 허용
    nw = max(new_box[2] - new_box[0], new_box[3] - new_box[1])
    # 위치별(box/det_box) (중심 거리, 좌우 차이) 중 가장 가까운 것 - 좌우로 먼 박스(옆 반사띠 등)는 같은 차로 안 봄
    def nearest(t):
        return min((center_dist(new_box, b), abs((new_box[0] + new_box[2]) - (b[0] + b[2])) / 2) for b in places(t))

    near = [(*nearest(t), bid) for bid, t in candidates.items()]
    near = [(d, bid) for d, dx, bid in near if d < nw * MATCH_DIST_RATIO and dx < nw * MATCH_DX_RATIO]
    return min(near)[1] if near else None


def plate_template(frame, box):
    """탐지된 번호판 부분을 흑백 이미지로 잘라 둠 - follow_boxes가 다음 프레임에서 같은 모양을 찾는 기준."""
    x1, y1, x2, y2 = box
    patch = frame[max(0, y1):y2, max(0, x1):x2]
    return cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.size else None


def follow_boxes(frame, tracked, frame_id):
    """탐지(5프레임마다) 사이 프레임에서 박스를 번호판 위치에 붙여 움직임.
    직전 박스 주변 좁은 범위에서 탐지 때 저장한 번호판 모양과 가장 비슷한 곳을 찾아 박스를 옮김.
    화면에 그리는 위치만 바뀌고, 인식(OCR)·발견 확정에는 영향 없음."""
    H, W = frame.shape[:2]
    for t in tracked.values():
        tmpl = t.get("tmpl")
        if tmpl is None or frame_id - t["last_seen_frame"] > BOX_SHOW_FRAMES:
            continue
        th, tw = tmpl.shape[:2]
        x1, y1, x2, y2 = t["box"]
        mx, my = int(tw * 0.5) + 8, int(th * 1.5) + 12  # 차는 위아래로 많이 움직여서 세로 범위를 넓게
        sx1, sy1, sx2, sy2 = max(0, x1 - mx), max(0, y1 - my), min(W, x2 + mx), min(H, y2 + my)
        if sy2 - sy1 < th or sx2 - sx1 < tw:
            continue
        region = cv2.cvtColor(frame[sy1:sy2, sx1:sx2], cv2.COLOR_BGR2GRAY)  # 주변 영역만 흑백 변환(가볍게)
        _, score, _, (lx, ly) = cv2.minMaxLoc(cv2.matchTemplate(region, tmpl, cv2.TM_CCOEFF_NORMED))
        if score >= FOLLOW_MIN_SCORE:
            t["box"] = (sx1 + lx, sy1 + ly, sx1 + lx + tw, sy1 + ly + th)


# 재생(세션)별 발견 목록 - /matches가 cctv.vue에 돌려줌. 세션별로 나눠 동시 재생 시 섞이지 않음
SESSION_MATCHES: dict[str, dict] = {}  # session -> {"created": 시각, "matches": [발견, ...]}
SESSION_TTL_SECONDS = 3600  # 1시간 지난 세션은 새 재생 시작 때 정리

# 번호판 -> 최근 발견. 예전 방식 cctv.vue(/last_match 호출)를 쓰는 PC 호환용
LAST_MATCH: dict[str, dict] = {}

# 재생(세션)별 조작 상태 - cctv.vue 버튼이 /control로 바꾸고, generate_stream이 매 프레임 확인
# paused: 일시정지 여부 / seek: 이동할 초(+앞, -뒤, 여러 번 누르면 누적)
SESSION_CONTROL: dict[str, dict] = {}
HOLD_FRAME_INTERVAL = 0.5  # 일시정지/영상 끝에서 같은 장면을 다시 보내는 간격(연결 유지용)

# 번호판 탐지 모델 - 서버 시작 때 한 번만 로딩 + 예열해서 모든 재생이 같이 씀.
# 재생할 때마다 새로 로딩하면 모델 로딩/첫 추론 준비 시간만큼 첫 화면이 늦게 뜸.
# 동시 재생(카메라 2대 등)이 한 모델을 같이 쓰므로 탐지는 lock으로 한 번에 하나씩
# OpenVINO 모델(detect_stage.PLATE_OPENVINO_PATH)이 있고 openvino가 설치돼 있으면 그걸 씀(입력 512x960 고정).
# CPU 측정(앞 카메라 ROI): PyTorch 640 약 70ms / OpenVINO 512x960 약 32ms, 960이 640보다 번호판을 약 25% 더 잡음
if PLATE_OPENVINO_PATH.exists() and importlib.util.find_spec("openvino"):
    PLATE_DETECTOR = PlateDetector(model_path=PLATE_OPENVINO_PATH, imgsz_candidates=([512, 960],))
    WARMUP_RUNS = 12  # OpenVINO는 처음 10번쯤 느림(내부 준비) -> 서버 시작 때 미리 돌려둠
else:
    PLATE_DETECTOR = PlateDetector(imgsz_candidates=(640,))
    WARMUP_RUNS = 1
for _ in range(WARMUP_RUNS):
    PLATE_DETECTOR.detect(np.zeros((720, 1280, 3), dtype=np.uint8))
PLATE_DETECTOR_LOCK = threading.Lock()


class DetectThread:
    """한 재생(스트림)의 번호판 탐지를 영상 루프와 따로 돌림.
    CPU에서는 탐지 1번이 약 70ms(프레임 2장 분량)라 영상 루프 안에서 돌리면 탐지하는 프레임마다 영상이 멈칫함.
    영상 루프는 쉬고 있을 때만 프레임을 넘기고(submit) 끝난 결과를 가져가기만(take_result) 해서 영상은 계속 흐름."""

    def __init__(self, roi):
        self.roi = roi
        self.cond = threading.Condition()
        self.request = None   # 탐지할 프레임 (generation, frame_id, frame)
        self.result = None    # 끝난 탐지 (generation, frame_id, frame, plate_boxes)
        self.busy = False
        self.stopped = False
        threading.Thread(target=self._run, daemon=True).start()

    def submit(self, generation, frame_id, frame) -> bool:
        """탐지 스레드가 쉬고 있으면 이 프레임을 넘김. 탐지 중이면 넘기지 않음(False)."""
        with self.cond:
            if self.busy or self.request is not None:
                return False
            # 영상 루프가 이 프레임에 박스를 그리므로 복사본을 넘김(박스 좌표/크롭은 이 프레임 기준)
            self.request = (generation, frame_id, frame.copy())
            self.cond.notify()
            return True

    def take_result(self):
        with self.cond:
            result, self.result = self.result, None
            return result

    def stop(self):
        with self.cond:
            self.stopped = True
            self.cond.notify()

    def _run(self):
        rx1, ry1, rx2, ry2 = self.roi
        while True:
            with self.cond:
                while self.request is None and not self.stopped:
                    self.cond.wait()
                if self.stopped:
                    return
                generation, frame_id, frame = self.request
                self.request = None
                self.busy = True
            with PLATE_DETECTOR_LOCK:
                plate_boxes = PLATE_DETECTOR.detect(frame[ry1:ry2, rx1:rx2])
            with self.cond:
                self.result = (generation, frame_id, frame, plate_boxes)
                self.busy = False


def publish_match(session: str, vehicle_number: str, confidence: float, node_id: str, hits: int):
    """발견 확정된 수배차량을 세션 발견 목록에 추가(DB 저장은 안 함).
    cctv.vue가 /matches로 받아 fa_server에 저장 요청. detection_id로 같은 발견을 한 번만 보냄.
    위치 이름(node_name)은 fa_server가 자기 DB에서 채움."""
    match = {
        "detection_id": uuid.uuid4().hex,
        "vehicle_number": vehicle_number,
        "confidence": confidence,
        "node_id": node_id,
        "detected_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hits": hits,
        "history_size": HISTORY_SIZE,
    }
    SESSION_MATCHES.setdefault(session, {"created": time.time(), "matches": []})["matches"].append(match)
    LAST_MATCH[normalize(vehicle_number)] = match


# 오래된 세션 발견 목록 삭제(메모리 정리) - /stream 호출 때마다 실행
def prune_sessions():
    now = time.time()
    for s in [s for s, v in SESSION_MATCHES.items() if now - v["created"] > SESSION_TTL_SECONDS]:
        del SESSION_MATCHES[s]


def generate_stream(targets: set[str], session: str, camera: dict):
    """영상 한 번 재생 = 스트림 하나. 프레임마다 [OCR 결과 반영 -> 번호판 탐지/OCR 요청 -> 박스 그리기]
    후 JPEG로 내보냄. targets(시청자의 공유 수배 목록)에 있는 번호가 확정되면 publish_match."""
    video_path, roi_poly, node_id = camera["video"], camera["roi_poly"], camera["node_id"]
    # 박스 id 앞에 세션 구분자 - OCR 결과 파일을 모든 스트림이 같이 쓰므로 동시 재생 시 결과 섞임 방지
    box_prefix = re.sub(r"[^0-9a-zA-Z]", "", session)[:8] or uuid.uuid4().hex[:8]

    PENDING_DIR.mkdir(parents=True, exist_ok=True)
    results_offset = RESULTS_PATH.stat().st_size if RESULTS_PATH.exists() else 0
    results_leftover = ""

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = 1.0 / (fps * camera.get("speed", 1.0))  # 프레임 사이 실제 대기 시간

    # roi_poly 없으면(아직 안 튜닝된 카메라) 리사이즈된 프레임 전체를 탐지 영역으로 씀
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * DISPLAY_SCALE)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * DISPLAY_SCALE)
    if roi_poly is None:
        roi_poly = [(0, 0), (frame_w, 0), (frame_w, frame_h), (0, frame_h)]
    roi = (
        min(p[0] for p in roi_poly), min(p[1] for p in roi_poly),
        max(p[0] for p in roi_poly), max(p[1] for p in roi_poly),
    )
    roi_poly_np = np.array(roi_poly, dtype=np.float32)

    detect_thread = DetectThread(roi)
    generation = 0  # 이동(seek)할 때마다 증가 - 이동 전 장면의 늦게 끝난 탐지 결과는 버림
    last_submit = -DETECT_EVERY_N_FRAMES  # 마지막으로 탐지에 넘긴 프레임

    tracked = {}
    next_box_id = 0
    published: set[str] = set()  # 이 스트림(한 번의 재생)에서 이미 올린 차량은 다시 안 올림

    frame_id = 0
    play_start = time.time()
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    last_chunk = None     # 마지막으로 보낸 장면 - 일시정지/영상 끝에서 이걸 계속 보냄
    ended = False         # 영상 끝 - 스트림을 끊지 않고 마지막 장면에서 대기(5초 뒤로 가능하게)
    holding = False       # 직전까지 멈춰 있었는지 - 다시 재생할 때 재생 시계를 맞추는 용도
    show_one = False      # 멈춘 상태에서 이동하면 이동한 장면 한 장만 새로 그림
    SESSION_CONTROL[session] = {"paused": False, "seek": 0.0}

    try:
        while True:
            # 0) 조작 반영 - 이동(±초) / 일시정지 / 영상 끝 대기
            ctl = SESSION_CONTROL.get(session, {})
            if ctl.get("seek"):
                cur = max(frame_id - 1, 0)
                target_frame = int(min(max(cur + ctl["seek"] * fps, 0), max(total_frames - 1, 0)))
                ctl["seek"] = 0.0
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                frame_id = target_frame
                tracked.clear()  # 이동 전 장면의 박스는 버림(늦게 온 OCR 결과도 박스가 없어 무시됨)
                generation += 1
                last_submit = frame_id - DETECT_EVERY_N_FRAMES
                ended = False
                show_one = True
            if (ctl.get("paused") or ended) and last_chunk is not None and not show_one:
                holding = True
                yield last_chunk
                time.sleep(HOLD_FRAME_INTERVAL)
                continue
            if holding or show_one:
                play_start = time.time() - frame_id * frame_interval  # 멈춘/이동한 위치부터 원래 속도로
                holding = False
            show_one = False

            ok, frame = cap.read()
            if not ok:
                ended = True
                if last_chunk is None:
                    break
                continue
            if DISPLAY_SCALE != 1.0:
                frame = cv2.resize(frame, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)

            # 1) 새 OCR 결과 반영 - 박스별 최근 읽기 기록으로 번호 결정, 수배 목록과 대조
            try:
                with open(RESULTS_PATH, "rb") as f:
                    f.seek(results_offset)
                    new_bytes = f.read()
                results_offset += len(new_bytes)
            except FileNotFoundError:
                new_bytes = b""

            chunk = results_leftover + new_bytes.decode("utf-8-sig", errors="ignore")
            if chunk.endswith("\n"):
                new_lines = chunk.splitlines()
                results_leftover = ""
            else:
                split = chunk.splitlines()
                new_lines = split[:-1] if split else []
                results_leftover = split[-1] if split else chunk

            for line in new_lines:
                parts = line.split(",")
                if len(parts) < 3:
                    continue
                crop_id, text, score_str = parts[0], parts[1], parts[2]
                box_id = crop_id.split("__")[0]
                score = float(score_str)
                if box_id not in tracked:
                    continue
                # 번호판 형식(숫자2~3 + 한글 + 숫자4)이 아닌 읽기는 버림 - 잘못 읽은 값이 섞여 박스 글자가 바뀌는 것 방지
                if not is_valid_plate_format(normalize(text)):
                    continue
                if score < 0.5:
                    # 확정(3번 중 2번) 기록에는 안 넣음. 다만 아직 제대로 읽힌 번호가 없는 박스면
                    # "인식중..." 대신 이 결과를 실제 점수 그대로 화면에만 표시(바로 번호가 뜨게)
                    t = tracked[box_id]
                    if not t["history"] and text:
                        t["text"], t["score"] = text, score
                    continue
                hist = tracked[box_id]["history"]
                hist.append((text, score))
                del hist[:-HISTORY_SIZE]
                counts = {}
                for h_text, h_score in hist:
                    counts.setdefault(h_text, []).append(h_score)
                winner = max(counts.items(), key=lambda kv: (len(kv[1]), max(kv[1])))
                hits = len(winner[1])
                tracked[box_id]["text"] = winner[0]
                tracked[box_id]["score"] = max(winner[1])
                tracked[box_id]["hits"] = hits
                plate = normalize(winner[0])
                # 수배 목록에 있고 확정 조건(3번 중 2번 일치)을 만족하면 발견 - 재생당 차량별 1회
                if plate in targets and hits >= CONFIRM_MIN_HITS:
                    if plate not in published:
                        published.add(plate)
                        publish_match(session, plate, max(winner[1]), node_id, hits)

            # 2) 번호판 탐지(ROI 안만, 탐지 스레드) + 크롭을 _live_pending/에 넣어 OCR 워커에 요청
            if frame_id - last_submit >= DETECT_EVERY_N_FRAMES and detect_thread.submit(generation, frame_id, frame):
                last_submit = frame_id
            det = detect_thread.take_result()
            if det is not None and det[0] == generation:
                # 박스 좌표/번호판 모양/크롭은 탐지한 프레임(det_frame) 기준. 지금 프레임 위치로는 아래 follow_boxes가 옮김
                _, det_frame_id, det_frame, plate_boxes = det
                now = time.time()
                rx1, ry1 = roi[0], roi[1]
                taken = set()  # 이번 탐지에서 이미 박스가 이어진 차량
                for score_det, roi_box in plate_boxes:
                    bx1, by1, bx2, by2 = roi_box
                    box = (bx1 + rx1, by1 + ry1, bx2 + rx1, by2 + ry1)
                    # 박스 중심이 사다리꼴 ROI 밖이면 제외
                    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                    if cv2.pointPolygonTest(roi_poly_np, (cx, cy), False) < 0:
                        continue
                    matched_id = find_match(box, tracked, taken)
                    if matched_id is not None:
                        taken.add(matched_id)
                        t = tracked[matched_id]
                        t["box"] = box
                        t["det_box"] = box
                        t["tmpl"] = plate_template(det_frame, box)
                        t["last_seen_frame"] = det_frame_id
                        hist = t["history"]
                        settled = len(hist) >= HISTORY_SIZE and len({h_text for h_text, _ in hist}) == 1
                        cooldown = CONFIRMED_COOLDOWN_SECONDS if settled else REOCR_COOLDOWN_SECONDS
                        if now - t.get("last_ocr_request", 0) < cooldown:
                            continue
                        box_id = matched_id
                    else:
                        box_id = f"{box_prefix}b{next_box_id}"
                        next_box_id += 1
                        taken.add(box_id)
                        tracked[box_id] = {
                            "box": box, "det_box": box, "text": "인식중...", "score": 0.0, "hits": 0,
                            "last_seen_frame": det_frame_id, "last_ocr_request": 0,
                            "history": [], "tmpl": plate_template(det_frame, box),
                        }

                    crop = crop_with_margin(det_frame, box, margin=15)
                    if crop.size == 0:
                        continue
                    # 작은 크롭은 확대해서 OCR 인식률 향상
                    crop_h, crop_w = crop.shape[:2]
                    if crop_w < 120:
                        scale = 100 / crop_w
                        crop = cv2.resize(
                            crop, (int(crop_w * scale), int(crop_h * scale)),
                            interpolation=cv2.INTER_CUBIC,
                        )
                    crop_id = f"{box_id}__{now:.6f}"
                    final_path = PENDING_DIR / f"{crop_id}.jpg"
                    tmp_path = PENDING_DIR / f"{crop_id}.jpg.tmp"
                    imwrite_unicode(tmp_path, crop)
                    tmp_path.rename(final_path)
                    tracked[box_id]["last_ocr_request"] = now

                stale = [bid for bid, t in tracked.items()
                         if frame_id - t["last_seen_frame"] > int(HOLD_SECONDS * fps)]
                for bid in stale:
                    del tracked[bid]
            if tracked:
                # 매 프레임 박스를 번호판 모양으로 따라가게 옮김(탐지 사이 + 탐지가 끝나는 동안 지나간 프레임만큼)
                follow_boxes(frame, tracked, frame_id)

            # 3) 박스/번호 그리기 (ROI 경계를 보려면 아래 줄 주석 해제)
            # cv2.polylines(frame, [roi_poly_np.astype(np.int32)], True, (0, 255, 0), 2)
            # 화면에는 최근 BOX_SHOW_FRAMES 안에 확인된 박스만 표시(한 번 놓쳐도 깜빡이지 않게).
            # 그중 더 최근에 확인된 박스와 겹치는 예전 박스(같은 차의 이전 위치)는 숨김
            visible = sorted(
                (t for t in tracked.values()
                 if frame_id - t["last_seen_frame"] <= BOX_SHOW_FRAMES),
                key=lambda t: -t["last_seen_frame"],
            )
            shown = []
            for t in visible:
                nw = max(t["box"][2] - t["box"][0], t["box"][3] - t["box"][1])
                if any(s["last_seen_frame"] > t["last_seen_frame"]
                       and (iou(t["box"], s["box"]) > 0 or center_dist(t["box"], s["box"]) < nw * MATCH_DIST_RATIO)
                       for s in shown):
                    continue
                shown.append(t)

            text_jobs = []
            for t in shown:
                x1, y1, x2, y2 = t["box"]
                # 빨강: 발견 확정된 수배차량 / 초록: 읽힌 번호판 / 회색: 인식 전
                is_target = normalize(t["text"]) in targets and t["hits"] >= CONFIRM_MIN_HITS
                color = (0, 0, 255) if is_target else ((0, 255, 0) if t["score"] > 0 else (150, 150, 150))
                thickness = 4 if is_target else 2
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                label = t["text"] if t["score"] == 0 else f"{t['text']} ({t['score']:.2f})"
                text_jobs.append((label, (x1, max(0, y1 - 32)), color))

            for text, org, color_bgr in text_jobs:
                draw_label(frame, text, org, color_bgr)

            cv2.putText(frame, f"{frame_id/fps:.1f}s", (20, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if ok:
                last_chunk = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
                yield last_chunk

            target_time = play_start + frame_id * frame_interval
            sleep_s = target_time - time.time()
            if sleep_s > 0:
                time.sleep(sleep_s)
            elif sleep_s < -0.5:
                # 처리가 0.5초 넘게 밀리면(첫 탐지 준비 등) 빨리 감아 따라잡지 않고 지금 위치부터 다시 정상 속도로
                play_start = time.time() - frame_id * frame_interval
            frame_id += 1
    finally:
        detect_thread.stop()
        cap.release()
        SESSION_CONTROL.pop(session, None)


@app.get("/stream")
def stream(targets: str = "", session: str = "", camera: str = DEFAULT_CAMERA, target: str = ""):
    """영상 스트림 - cctv.vue의 <img src>가 호출.
    targets: 대조할 수배차량(쉼표 구분, cctv.vue가 sp_server canShowCase로 받은 공유 목록)
    session: 재생마다 cctv.vue가 만드는 id(/matches 조회 키)
    camera: CAMERA_CONFIG 키 / target: 예전 방식 cctv.vue 호환용(차량 1대)"""
    plates = {normalize(p) for p in (targets or target).split(",") if normalize(p)}
    cam = CAMERA_CONFIG.get(camera, CAMERA_CONFIG[DEFAULT_CAMERA])
    session = session or uuid.uuid4().hex
    prune_sessions()
    SESSION_MATCHES[session] = {"created": time.time(), "matches": []}
    for p in plates:
        LAST_MATCH.pop(p, None)
    return StreamingResponse(
        generate_stream(plates, session, cam),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/matches")
def matches(session: str):
    """이 재생(session)에서 발견된 수배차량 목록 - cctv.vue가 1초마다 조회해 결과 패널 표시 + fa_server 저장."""
    return SESSION_MATCHES.get(session, {}).get("matches", [])


@app.get("/control")
def control(session: str, action: str, sec: float = 0.0):
    """재생 조작 - cctv.vue의 카메라별 버튼이 호출. action: pause / resume / seek(sec초 이동, 음수면 뒤로)."""
    ctl = SESSION_CONTROL.get(session)
    if ctl is None:
        return {"ok": False}
    if action == "pause":
        ctl["paused"] = True
    elif action == "resume":
        ctl["paused"] = False
    elif action == "seek":
        ctl["seek"] += sec
    return {"ok": True, "paused": ctl["paused"]}


@app.get("/last_match")
def last_match(target: str = ""):
    """예전 방식 cctv.vue 호환용 - 이 번호판의 최근 발견 정보, 없으면 빈 값."""
    return LAST_MATCH.get(normalize(target)) or {}


def start_ocr_worker():
    """글자 인식 워커(live_ocr_worker.py)를 ocr_env(paddle 환경)로 같이 실행 - 터미널 하나로 둘 다 켜기 위함.
    paddle은 이 서버(torch)와 다른 환경이라 별도 프로세스로 띄움. 워커 출력은 이 터미널에 같이 나옴."""
    worker_python = ROOT / "ocr_env" / "Scripts" / "python.exe"
    if not worker_python.exists():
        print(f"[경고] {worker_python} 없음 - 글자 인식 워커를 따로 실행하세요(설치 안내 3번 참고)")
        return None
    # -u: 워커 출력("준비 완료" 등)을 쌓아두지 않고 바로 이 터미널에 표시
    return subprocess.Popen([str(worker_python), "-u", str(Path(__file__).parent / "live_ocr_worker.py")], cwd=str(ROOT))


if __name__ == "__main__":
    import uvicorn

    ocr_worker = start_ocr_worker()
    try:
        uvicorn.run(app, host="0.0.0.0", port=8010)
    finally:
        # 서버를 끄면(Ctrl+C) 워커도 같이 종료
        if ocr_worker is not None and ocr_worker.poll() is None:
            ocr_worker.terminate()
