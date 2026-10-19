# services/cctv_service.py
# CCTV 영상 재생 데모용 비즈니스 로직. 지금은 영상 1개짜리 시연이라 고정값(DEMO_*)으로
# 처리함 - 영상이 여러 개로 늘어나면 그때 DB 조회 기반으로 바꾸면 됨.

import os
from datetime import datetime, timedelta
from pathlib import Path

from repositories.cctv_repository import CctvRepository

# 아래 DEMO_* 상수는 전부 "②우리(OCR/영상) 쪽에서 그 순간 새로 만들어내는 값"만 남겨둠
# (신뢰도/노드/시각/이미지/어떤 영상 파일인지). "①수배차량 자체의 정보"(번호판/차종/색상)는
# DB(wanted_case)가 관리하는 값이라 더 이상 여기 하드코딩 안 하고, 매번
# CctvRepository.get_wanted_case_info()로 실시간 조회해서 씀 - 하드코딩해두면 DB 쪽에서
# 값이 바뀌었을 때 여기만 옛날 값으로 남아 어긋나는 문제가 있어서 이렇게 분리함.
DEMO_WANTED_CASE_ID = 961
DEMO_CASE_NAME = "front_test25s"
# 발견(트리거) 시점은 원래대로 첫 매칭 순간(9.84초)을 유지 - 대신 신뢰도는 같은 매칭 구간
# (9.84~11.85초, 계속 같은 수배차량이 찍히던 구간) 안에서 실제로 가장 높았던 값(0.999,
# 구간별 인식 기록 CSV 기준)으로 표시함. 지어낸 값이 아니라 같은 이벤트 안의 실측 최고치.
DEMO_MATCH_TIMESTAMP = 9.84
DEMO_CONFIDENCE = 0.999
# 1차 시연용 캡처 이미지 경로 - 환경변수로 지정 (PC마다 다름)
DEMO_IMAGE_PATH = os.getenv("DEMO_IMAGE_PATH", "")
DEMO_VIDEO_FILENAME_PLATE = os.getenv("DEMO_VIDEO_PLATE", "")  # annotated_{번호판}.mp4 파일명 찾을 때만 씀(표시용 아님)
# 카메라 위치(node)는 우리가 이름까지 지어내지 않고, DB(cctv 테이블)에서 실시간 조회해서 씀
# (monitoring.vue 지도가 쓰는 이름이랑 항상 똑같게 유지하려고 - node_id만 우리가 알고 있으면 됨)
DEMO_NODE_ID = "1230197500"  # 실제 cctv 테이블에 있는 노드로 변경(예전 "1180037801"은 가짜값이라
# cctv 테이블에 매칭이 안 돼서 monitoring.vue 지도에 핀이 안 찍히는 문제가 있었음)
VIDEO_DIR = Path(os.getenv("CCTV_VIDEO_DIR", "../ocr_server/output/video_watch"))

# 같은 사건이 이 시간(초) 안에 또 저장 요청되면 중복으로 보고 건너뜀 - 페이지 새로고침 후
# 같은 구간을 다시 재생했을 때 detection_log에 거의 동일한 기록이 여러 건 쌓이는 걸 방지.
# (video_match_stage.py의 COOLDOWN_SECONDS=5.0과 같은 개념, 다만 여기는 "재생 세션"
# 단위라 리로드+재생까지 시간이 좀 걸릴 수 있어서 5분으로 넉넉하게 잡음)
COOLDOWN_SECONDS = 300.0  # 5분

# CCTV 실시간 탐지 저장: 같은 차량이 이 시간 안에 또 들어오면 관측 기록/알림을 다시 남기지 않음
LIVE_COOLDOWN_MINUTES = 10

# ⚠️ 테스트용 시각 보정(일). 더미 데이터 경로가 10/7 날짜로 들어가 있어서, 지금 시각으로 저장하면
# 기존 경로보다 과거가 되어 지도 맨 끝에 안 찍힘 -> 테스트 동안만 저장 시각을 +6일.
# 시연 당일(10/7)에는 반드시 0으로 되돌릴 것.
LIVE_TIME_OFFSET_DAYS = 0


class CctvService:
    def __init__(self):
        self.repo = CctvRepository()

    def get_detection_video_info(self, wanted_case_id: int) -> dict | None:
        """CCTV 페이지 진입 시 필요한 정보(영상 주소, 저장 트리거 시각) 반환.
        준비된 사건이 아니면 None. 차량번호는 DB(wanted_case)에서 실시간 조회."""
        if wanted_case_id != DEMO_WANTED_CASE_ID:
            return None
        case_info = self.repo.get_wanted_case_info(wanted_case_id)
        if case_info is None:
            return None
        return {
            "video_url": f"/cctv/video-file/{DEMO_CASE_NAME}",
            "match_timestamp": DEMO_MATCH_TIMESTAMP,
            "vehicle_number": case_info["vehicle_number"],
        }

    def get_video_path(self, case_name: str) -> Path:
        return VIDEO_DIR / case_name / f"annotated_{DEMO_VIDEO_FILENAME_PLATE}.mp4"

    def confirm_detection(self, wanted_case_id: int) -> dict | None:
        """영상 재생 중 match_timestamp 지점을 지나갔을 때 호출됨.
        이미 계산해둔 결과를 detection_log에 저장. 준비된 사건이 아니면 None.
        차량번호/차종/색상은 DB(wanted_case)에서 실시간 조회해서 쓴다 - 하드코딩된
        값을 쓰면 DB에서 정보가 수정됐을 때 여기만 옛날 값으로 남는 문제가 있어서."""
        if wanted_case_id != DEMO_WANTED_CASE_ID:
            return None

        case_info = self.repo.get_wanted_case_info(wanted_case_id)
        if case_info is None:
            return None
        vehicle_number = case_info["vehicle_number"]
        vehicle_type = case_info["vehicle_type"]
        vehicle_color = case_info["vehicle_color"]

        # 위치 이름도 우리가 따로 짓지 않고 cctv 테이블에서 실시간 조회 (monitoring.vue
        # 지도가 보여주는 이름이랑 항상 일치하게 하려고 - 안 그러면 우리 결과패널이랑
        # 지도가 같은 node_id를 두고 서로 다른 이름을 보여주는 모순이 생김)
        node_name = self.repo.get_cctv_node_name(DEMO_NODE_ID) or DEMO_NODE_ID

        if self.repo.has_recent_detection(DEMO_WANTED_CASE_ID, COOLDOWN_SECONDS):
            # 이미 방금 저장된 기록이 있음 - 새로 INSERT 안 하고, 있는 값 그대로 보여주기만 함
            return {
                "status": "duplicate_skipped",
                "vehicle_number": vehicle_number,
                "confidence": DEMO_CONFIDENCE,
                "node_name": node_name,
                "detected_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

        self.repo.insert_full_detection(
            vehicle_number=vehicle_number,
            confidence=DEMO_CONFIDENCE,
            node_id=DEMO_NODE_ID,
            wanted_case_id=DEMO_WANTED_CASE_ID,
            image_path=DEMO_IMAGE_PATH,
            vehicle_type=vehicle_type,
            vehicle_color=vehicle_color,
        )
        return {
            "status": "saved",
            "vehicle_number": vehicle_number,
            "confidence": DEMO_CONFIDENCE,
            "node_name": node_name,
            "detected_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    def save_live_detection(self, vehicle_number: str, confidence: float, node_id: str) -> dict:
        """CCTV 실시간 탐지 결과 저장 - cctv.vue가 OCR 서버(/matches)에서 받은 발견 결과를 넘겨줌.
        흐름: 쿨다운 확인 -> 번호판으로 사건 조회 -> 알림 대상 조회 -> 관측 기록 + 알림 저장.
        반환값(node_name 포함)은 cctv.vue 탐지 결과 패널에 표시됨."""
        vehicle_number = vehicle_number.replace(" ", "")
        node_name = self.repo.get_cctv_node_name(node_id) or node_id
        result = {
            "vehicle_number": vehicle_number,
            "confidence": confidence,
            "node_name": node_name,
            "detected_at": (datetime.now() + timedelta(days=LIVE_TIME_OFFSET_DAYS)).strftime("%Y-%m-%d %H:%M:%S"),
        }

        if self.repo.has_recent_vehicle_detection(vehicle_number, LIVE_COOLDOWN_MINUTES, LIVE_TIME_OFFSET_DAYS):
            return {"status": "duplicate_skipped", **result}

        wanted_case_id = self.repo.get_active_wanted_case_id(vehicle_number)
        recipients = self.repo.get_case_recipients(wanted_case_id) if wanted_case_id is not None else []
        content = f"수배 차량({vehicle_number})이 {node_name}에서 통과했습니다."
        self.repo.insert_live_detection(
            vehicle_number=vehicle_number,
            confidence=confidence,
            node_id=node_id,
            wanted_case_id=wanted_case_id,
            recipients=recipients,
            content=content,
            offset_days=LIVE_TIME_OFFSET_DAYS,
        )
        return {"status": "saved", **result}
