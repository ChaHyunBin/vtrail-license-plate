"""실시간 OCR 담당 프로세스. output/_live_pending/ 폴더를 계속 감시하다가 새 크롭
이미지가 생기면 OCR 돌려서 output/_live_results.csv에 결과를 추가한다.
live_detect_and_play.py(탐지+재생, torch)랑 완전히 별도 프로세스로 실행해야 함
(torch/paddle 같은 프로세스에서 못 씀 - DLL 충돌 확인됨).

크롭을 한 장씩 처리하지 않고 여러 장을 모아서 한 번에 GPU에 넣는 배치 처리로
속도를 높임(한 장씩 처리하는 것보다 훨씬 빠름).

실행: ocr_env/python.exe scripts/live_ocr_worker.py
Ctrl+C로 종료.
"""
import time
from pathlib import Path

import cv2

from recognize_stage_nrtr import CtcInferRecognizer, imread_unicode

ROOT = Path(__file__).resolve().parent.parent
PENDING_DIR = ROOT / "output" / "_live_pending"
RESULTS_PATH = ROOT / "output" / "_live_results.csv"

POLL_INTERVAL = 0.03  # 30ms마다 새 파일 확인
BATCH_SIZE = 8  # 한 번에 GPU에 넣을 최대 크롭 수

# 신뢰도(결과 점수) 계산 방식 - 바꾼 뒤 워커만 재시작하면 적용됨
#   "min" : 글자별 점수 중 가장 낮은 값. 한 글자라도 헷갈리면 점수가 낮아져서 "틀렸는데 0.9 이상"이 줄어듦
#   "mean": 글자별 점수의 평균(PaddleOCR 기본, 예전 방식)
# (영상 2개 평가: 0.9 이상으로 표시된 오답이 mean 74개 -> min 23개)
SCORE_MODE = "min"

# 번호판 이미지 좌우에 붙이는 여백(px). 맨 앞 숫자가 이미지 끝에 붙어 있으면 모델이 빼먹는 문제 완화
# (영상 2개 평가: 앞자리 빠짐 26번 -> 5번, 최종 확정 오답 1대 -> 0대). 0이면 예전 방식
SIDE_PAD = 20

# 밀린 크롭 정리 기준(초) - 요청된 지 이만큼 지난 크롭은 버림.
# CPU가 잠깐 바쁠 때 대기열이 쌓여도 지난 장면 대신 최신 장면을 읽게 함
MAX_CROP_AGE = 1.0


def pick_batch() -> tuple[list[Path], int]:
    """이번에 OCR할 크롭 고르기. 반환: (처리할 크롭, 버린 개수)
    파일 이름 = "박스ID__요청시각.jpg" (live_stream_server가 만듦)
    - 요청된 지 MAX_CROP_AGE초 넘은 크롭은 버림(차가 이미 움직임)
    - 같은 박스(같은 번호판) 크롭이 여러 장이면 가장 최근 것만 남김(결과가 거의 같음)
    - 남은 것은 오래된 순서대로 BATCH_SIZE장"""
    now = time.time()
    newest: dict[str, tuple[float, Path]] = {}  # 박스ID -> (요청시각, 파일)
    dropped = 0
    for p in PENDING_DIR.glob("*.jpg"):
        box_id, _, ts = p.stem.rpartition("__")
        try:
            t = float(ts)
        except ValueError:
            t = now  # 이름 형식이 다르면 방금 요청된 것으로 보고 처리
        if now - t > MAX_CROP_AGE:
            p.unlink(missing_ok=True)
            dropped += 1
            continue
        if box_id in newest:
            older = min(newest[box_id], (t, p))
            older[1].unlink(missing_ok=True)
            dropped += 1
            newest[box_id] = max(newest[box_id], (t, p))
        else:
            newest[box_id] = (t, p)
    batch = [p for _, p in sorted(newest.values())][:BATCH_SIZE]
    return batch, dropped


def process_batch(crop_paths: list[Path], recognizer) -> int:
    crops = []
    valid_paths = []
    for p in crop_paths:
        crop = imread_unicode(p)
        if crop is None or crop.size == 0:
            p.unlink(missing_ok=True)
            continue
        if SIDE_PAD:
            crop = cv2.copyMakeBorder(crop, 0, 0, SIDE_PAD, SIDE_PAD, cv2.BORDER_REPLICATE)
        crops.append(crop)
        valid_paths.append(p)

    if not crops:
        return 0

    results = recognizer.recognize_batch(crops)

    lines = []
    for p, r in zip(valid_paths, results):
        # 두 방식(CTC/NRTR) 중 고르는 기준은 그대로 평균 점수, 결과로 쓰는 점수만 SCORE_MODE에 따름
        head = "nrtr" if r["nrtr_score"] >= r["ctc_score"] else "ctc"
        text = r[f"{head}_text"]
        score = r[f"{head}_min"] if SCORE_MODE == "min" else r[f"{head}_score"]
        lines.append(f"{p.stem},{text},{score:.4f}\n")

    with open(RESULTS_PATH, "a", encoding="utf-8-sig") as f:
        f.writelines(lines)

    for p in valid_paths:
        p.unlink(missing_ok=True)

    return len(valid_paths)


def main():
    # 이전 실행의 찌꺼기 파일들 때문에 헷갈리지 않게 시작할 때 싹 비움
    if PENDING_DIR.exists():
        for f in PENDING_DIR.glob("*.jpg"):
            f.unlink(missing_ok=True)
    PENDING_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text("id,text,score\n", encoding="utf-8-sig")

    print("OCR 워커 시작, 모델 로딩중...")
    recognizer = CtcInferRecognizer()  # CPU용 추론 모델(CTC만). 예전 방식: DualHeadRecognizer()
    print("준비 완료, 대기중...")

    processed = 0
    while True:
        pending_files, dropped = pick_batch()
        if not pending_files:
            time.sleep(POLL_INTERVAL)
            continue

        try:
            n = process_batch(pending_files, recognizer)
        except Exception as e:
            # 배치 하나 실패해도 워커는 계속 돌아야 함
            print(f"[경고] 배치 처리 실패({len(pending_files)}장): {type(e).__name__}: {e}")
            for p in pending_files:
                p.unlink(missing_ok=True)
            continue

        processed += n
        print(f"  {processed}개 처리함 (이번 배치 {n}장" + (f", 밀린 크롭 {dropped}장 버림)" if dropped else ")"))


if __name__ == "__main__":
    main()
