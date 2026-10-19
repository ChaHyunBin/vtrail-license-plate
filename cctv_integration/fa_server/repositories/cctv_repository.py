# repositories/cctv_repository.py
# CCTV 영상 재생 데모(웹에서 영상 틀고 -> 목표 번호판 매칭되면 detection_log 저장) 전용 DB 접근.
# vehicle_repository.py와 겹치지 않게 완전히 별도 파일로 분리함(팀 작업 중 같은 파일
# 동시 수정으로 인한 git 충돌 방지 목적).

import os

import psycopg2


class CctvRepository:
    def __init__(self):
        self.conn_params = dict(
            host=os.getenv('DB_HOST', 'localhost'),
            dbname=os.getenv('DB_NAME', 'vtrail'),
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD'),
            port=os.getenv('DB_PORT', '5432')
        )

    # wanted_case_id로 등록된 진짜 차량 정보(번호판/차종/색상) 조회.
    # DB(사건 등록) 쪽에서 관리하는 값이라, 우리 쪽에 따로 하드코딩해두면 나중에 어긋날
    # 위험이 있어서 항상 이걸로 실시간 조회해서 씀.
    def get_wanted_case_info(self, wanted_case_id: int) -> dict | None:
        conn = psycopg2.connect(**self.conn_params)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT vehicle_number, vehicle_type, vehicle_color "
                    "FROM wanted_case WHERE wanted_case_id = %s",
                    (wanted_case_id,),
                )
                row = cur.fetchone()
                if row is None:
                    return None
                return {"vehicle_number": row[0], "vehicle_type": row[1], "vehicle_color": row[2]}
        finally:
            conn.close()

    # node_id로 실제 cctv 위치명 조회. monitoring.vue의 지도가 쓰는 것과 똑같은 방식
    # (coalesce(road_name, cctv_name))으로 맞춰서, 우리 결과패널이랑 지도가 같은 이름을 보여주게 함.
    def get_cctv_node_name(self, node_id: str) -> str | None:
        conn = psycopg2.connect(**self.conn_params)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COALESCE(road_name, cctv_name) FROM cctv WHERE node_id = %s",
                    (str(node_id),),
                )
                row = cur.fetchone()
                return row[0] if row else None
        finally:
            conn.close()

    # 최근 cooldown_seconds 안에 같은 사건(wanted_case_id)으로 이미 저장된 기록이 있는지 확인.
    # 페이지 새로고침 후 같은 구간을 다시 재생했을 때 같은 순간이 중복으로 쌓이는 걸 막는 용도
    # (video_match_stage.py의 COOLDOWN_SECONDS 중복제거랑 같은 아이디어를 DB 조회로 구현).
    def has_recent_detection(self, wanted_case_id: int, cooldown_seconds: float) -> bool:
        conn = psycopg2.connect(**self.conn_params)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM detection_log "
                    "WHERE wanted_case_id = %s AND detected_at >= NOW() - (%s || ' seconds')::interval "
                    "LIMIT 1",
                    (wanted_case_id, cooldown_seconds),
                )
                return cur.fetchone() is not None
        finally:
            conn.close()

    # CCTV 데모 페이지용: 이미 계산해둔 OCR 결과 한 건을 detection_log에 그대로 저장.
    # 무거운 탐지/인식을 다시 안 돌리고, 사전에 확인된 값만 넣는 용도.
    def insert_full_detection(
        self, vehicle_number: str, confidence: float, node_id: str, wanted_case_id: int,
        image_path: str, vehicle_type: str | None = None, vehicle_color: str | None = None,
    ) -> None:
        conn = psycopg2.connect(**self.conn_params)
        try:
            with conn.cursor() as cur:
                sql = """
                    INSERT INTO detection_log
                        (vehicle_number, confidence, vehicle_color, vehicle_type, node_id, wanted_case_id, detected_at, image_path)
                    VALUES (%s, %s, %s, %s, %s, %s, NOW(), %s)
                """
                cur.execute(sql, (
                    vehicle_number, confidence, vehicle_color, vehicle_type,
                    str(node_id), wanted_case_id, image_path,
                ))
            conn.commit()
        finally:
            conn.close()

    # 아래 4개: CCTV 실시간 탐지 저장(POST /cctv/detection -> CctvService.save_live_detection)용

    # 번호판으로 진행 중(track_status=0) 수배 사건 id 조회 - 사건 id는 PC(DB)마다 달라서 번호판으로 찾음
    def get_active_wanted_case_id(self, vehicle_number: str) -> int | None:
        conn = psycopg2.connect(**self.conn_params)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT wanted_case_id FROM wanted_case "
                    "WHERE REPLACE(vehicle_number, ' ', '') = %s AND track_status = 0 "
                    "ORDER BY registered_at DESC LIMIT 1",
                    (vehicle_number,),
                )
                row = cur.fetchone()
                return row[0] if row else None
        finally:
            conn.close()

    # 같은 차량이 최근 cooldown_minutes 안에 기록됐는지 - 영상 반복 재생 시 중복 저장 방지
    # offset_days: 테스트용 시각 보정(CctvService.LIVE_TIME_OFFSET_DAYS) - 저장 시각과 같은 기준으로 비교
    def has_recent_vehicle_detection(self, vehicle_number: str, cooldown_minutes: int, offset_days: int = 0) -> bool:
        conn = psycopg2.connect(**self.conn_params)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM detection_log "
                    "WHERE vehicle_number = %s "
                    "AND detected_at >= NOW() + (%s || ' days')::interval - (%s || ' minutes')::interval "
                    "LIMIT 1",
                    (vehicle_number, offset_days, cooldown_minutes),
                )
                return cur.fetchone() is not None
        finally:
            conn.close()

    # 알림 받을 사람: 사건 담당자 + 공유 승인된 팀원(case_share_member)
    def get_case_recipients(self, wanted_case_id: int) -> list[str]:
        conn = psycopg2.connect(**self.conn_params)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT manager_id, case_id FROM wanted_case WHERE wanted_case_id = %s",
                    (wanted_case_id,),
                )
                row = cur.fetchone()
                if not row:
                    return []
                manager_id, case_id = row
                cur.execute(
                    "SELECT employee_id FROM case_share_member WHERE case_id = %s AND approval_status = 1",
                    (case_id,),
                )
                shared_ids = [r[0] for r in cur.fetchall()]
                return list(dict.fromkeys([manager_id, *shared_ids]))  # 중복 제거(순서 유지)
        finally:
            conn.close()

    # 관측 기록(detection_log -> 지도 경로) + 알림(notification -> 상단 알림)을 한 트랜잭션으로 저장
    def insert_live_detection(
        self, vehicle_number: str, confidence: float, node_id: str,
        wanted_case_id: int | None, recipients: list[str], content: str, offset_days: int = 0,
    ) -> None:
        conn = psycopg2.connect(**self.conn_params)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO detection_log "
                    "(vehicle_number, confidence, node_id, wanted_case_id, detected_at) "
                    "VALUES (%s, %s, %s, %s, NOW() + (%s || ' days')::interval)",
                    (vehicle_number, confidence, str(node_id), wanted_case_id, offset_days),
                )
                for employee_id in recipients:
                    cur.execute(
                        "INSERT INTO notification "
                        "(receiver_id, wanted_case_id, type, content, created_at) "
                        "VALUES (%s, %s, %s, %s, NOW() + (%s || ' days')::interval)",
                        (employee_id, wanted_case_id, "수배차량통과알림", content, offset_days),
                    )
            conn.commit()
        finally:
            conn.close()
