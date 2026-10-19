# routers/cctv_router.py
# CCTV 페이지에서 영상 재생 -> 목표 번호판 매칭 순간 -> detection_log 저장까지의
# API 3개. vehicle_router.py와 완전히 분리된 새 파일(팀 작업 충돌 방지 목적).

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from services.cctv_service import CctvService

router = APIRouter(prefix="/cctv", tags=["CCTV"])
service = CctvService()


@router.get("/detection-video/{wanted_case_id}")
def get_detection_video(wanted_case_id: int):
    """CCTV 페이지 진입 시 호출 - 어떤 영상을 재생할지, 몇 초 지점에서 저장을
    트리거해야 하는지 프론트에 알려준다."""
    info = service.get_detection_video_info(wanted_case_id)
    if info is None:
        raise HTTPException(status_code=404, detail="준비된 영상이 없습니다")
    return info


@router.get("/video-file/{case_name}")
def get_video_file(case_name: str):
    """실제 mp4 파일을 그대로 서빙한다 - <video src="..."> 가 직접 가리키는 주소."""
    path = service.get_video_path(case_name)
    if not path.exists():
        raise HTTPException(status_code=404, detail="영상 파일을 찾을 수 없습니다")
    return FileResponse(path, media_type="video/mp4")


class ConfirmDetectionRequest(BaseModel):
    wanted_case_id: int


@router.post("/confirm-detection")
def confirm_detection(body: ConfirmDetectionRequest):
    """프론트가 영상 재생 중 match_timestamp 지점을 지나가는 순간 호출.
    무거운 OCR을 다시 돌리지 않고, 이미 계산해둔 결과를 detection_log에 저장만 한다."""
    result = service.confirm_detection(body.wanted_case_id)
    if result is None:
        raise HTTPException(status_code=404, detail="준비된 사건이 없습니다")
    return result


class LiveDetectionRequest(BaseModel):
    vehicle_number: str
    confidence: float
    node_id: str


@router.post("/detection")
def save_live_detection(body: LiveDetectionRequest):
    """CCTV 실시간 탐지 결과 저장 - 호출: cctv.vue saveDetection().
    OCR 서버(live_stream_server.py)는 DB 저장을 안 하므로, 각 PC의 fa_server가 자기 DB에
    관측 기록(지도 경로) + 알림을 저장함."""
    return service.save_live_detection(body.vehicle_number, body.confidence, body.node_id)
