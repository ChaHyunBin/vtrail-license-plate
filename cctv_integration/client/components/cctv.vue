<template>
  <main class="content">
    <section class="card video-card">
      <div class="card-title">
        <h2>CCTV 영상 확인</h2>
      </div>

      <!-- 대상 차량: 내 공유 수배차량(진행 중) 목록 = OCR 서버 대조 목록. 어느 카메라든 발견되면 빨간 뱃지 -->
      <div class="video-controls">
        <div class="case-list">
          대상 차량:
          <span
            v-for="c in activeCases"
            :key="c.wantedCaseId"
            :class="['badge', 'case-badge', { 'badge-detected': detectedVehicles.includes(normalizePlate(c.vehicleNumber)) }]"
          >
            {{ c.vehicleNumber }}<template v-if="detectedVehicles.includes(normalizePlate(c.vehicleNumber))"> 발견</template>
          </span>
        </div>
      </div>

      <p v-if="loadError" class="info-placeholder">{{ loadError }}</p>

      <!-- 카메라별 2칸: 각 칸 = 영상 + 그 카메라의 탐지 결과. 화면이 좁으면 위아래로 쌓임 -->
      <div v-else class="camera-grid">
        <div v-for="cam in cameras" :key="cam.key" class="camera-col">
          <div class="camera-title">
            <h3>{{ cam.label }}</h3>
            <span class="camera-desc">{{ cam.desc }}</span>
          </div>

          <!-- 카메라별 조작: 재생(처음부터) / 5초 뒤로 / 일시정지·재개 / 5초 앞으로 / 크게 보기 -->
          <div class="camera-controls">
            <button class="btn btn-primary btn-sm" :disabled="activeCases.length === 0" @click="playStream(cam)">재생</button>
            <button class="btn-ctl" :disabled="!cam.streaming" @click="seek(cam, -5)">5초 뒤로</button>
            <button class="btn-ctl" :disabled="!cam.streaming" @click="togglePause(cam)">
              {{ cam.paused ? '재개' : '일시정지' }}
            </button>
            <button class="btn-ctl" :disabled="!cam.streaming" @click="seek(cam, 5)">5초 앞으로</button>
            <button class="btn-ctl btn-fullscreen" :disabled="!cam.streaming" @click="openFullscreen(cam.key)">크게 보기</button>
          </div>

          <div :ref="el => (videoEls[cam.key] = el)" class="video-wrapper">
            <!-- CCTV 영상: OCR 서버(live_stream_server.py) /stream?camera= -->
            <img v-if="cam.streaming" :src="streamUrl(cam)" class="live-stream" />
            <p v-else class="info-placeholder">재생 버튼을 누르면 실시간 탐지가 시작됩니다.</p>
          </div>

          <!-- 탐지 결과: 이 카메라의 가장 최근 발견 차량 (fa_server 저장 응답 기준) -->
          <div class="result-panel">
            <div class="result-title">
              <h3>탐지 결과</h3>
            </div>

            <p v-if="!cam.result" class="info-placeholder">
              아직 매칭된 결과가 없습니다.
            </p>

            <dl v-else class="result-list">
              <div class="result-row">
                <dt>인식된 번호판</dt>
                <dd><span class="badge badge-active">{{ cam.result.vehicle_number }}</span></dd>
              </div>
              <div class="result-row">
                <dt>인식 신뢰도</dt>
                <dd>{{ (cam.result.confidence * 100).toFixed(1) }}%</dd>
              </div>
              <div class="result-row">
                <dt>인식 지점</dt>
                <dd>{{ cam.result.node_name }}</dd>
              </div>
              <div class="result-row">
                <dt>인식 시각</dt>
                <dd>{{ cam.result.detected_at }}</dd>
              </div>
            </dl>
          </div>
        </div>
      </div>
    </section>
  </main>
</template>

<script>
import axios from 'axios';

const FA_SERVER = 'http://localhost:8000';
const SP_SERVER = 'http://localhost:3000';
// OCR 서버(live_stream_server.py) - 각자 PC에서 CPU로 실행
const OCR_STREAM_SERVER = 'http://localhost:8010';

// 화면에 띄울 카메라 - key는 OCR 서버 CAMERA_CONFIG의 키(/stream?camera=)와 같아야 함
// desc: 카메라 설치 위치(OCR 서버 CAMERA_CONFIG의 node_id 위치와 맞출 것)
const CAMERAS = [
  { key: 'front', label: '앞 카메라', desc: '여의상류IC' },
  { key: 'back', label: '뒤 카메라', desc: '한국통신목동지점' },
];

// 번호판 비교용 - 숫자/한글만 남김(OCR 서버 normalize와 같은 기준)
const normalizePlate = text => (text || '').replace(/[^0-9가-힣]/g, '');

export default {
  name: 'CctvPage',
  data() {
    return {
      caseList: [],        // 내게 공유된 수배 사건 목록(sp_server canShowCase)
      loadError: '',
      pollTimer: null,     // /matches 1초 조회 타이머(재생 중인 카메라 공용)
      // 카메라별 상태 - streaming/paused: 재생·일시정지 여부
      //   sessionId: 재생마다 새로 만드는 id(OCR 서버 /stream, /matches, /control에 사용)
      //   result: 탐지 결과 칸의 최근 발견, sentIds: fa_server에 이미 보낸 발견, detected: 이 카메라에서 발견된 번호판
      cameras: CAMERAS.map(c => ({
        ...c, streaming: false, paused: false, sessionId: '', result: null, sentIds: [], detected: [],
      })),
      videoEls: {},        // 카메라별 영상 칸 요소 - "크게 보기"(전체 화면)에 사용
    };
  },
  computed: {
    // 어느 카메라에서든 발견된 번호판 - 대상 차량 뱃지 빨간색 표시
    detectedVehicles() {
      return [...new Set(this.cameras.flatMap(c => c.detected))];
    },
    // 진행 중(trackStatus 0) 사건만 - canShowCase는 종료 후 14일 이내 사건도 포함함
    activeCases() {
      return this.caseList.filter(c => c.trackStatus === 0);
    },
    // OCR 서버에 넘길 대조 목록(targets)
    targetPlates() {
      return this.activeCases.map(c => normalizePlate(c.vehicleNumber)).filter(Boolean);
    },
  },
  mounted() {
    this.loadCaseList();
  },
  beforeUnmount() {
    if (this.pollTimer) clearInterval(this.pollTimer);
  },
  methods: {
    // 내 공유 수배 사건 목록 - sp_server canShowCase(대시보드 "수배 차량 관리"와 같은 API)
    loadCaseList() {
      const employeeId = sessionStorage.getItem('employeeId');
      axios.post(`${SP_SERVER}/canShowCase`, null, { params: { employeeId } })
        .then(resp => {
          this.caseList = resp.data || [];
          if (this.activeCases.length === 0) {
            this.loadError = '진행 중인 수배차량이 없습니다.';
          }
        })
        .catch(err => {
          console.error(err);
          this.loadError = '담당 차량 목록을 불러오지 못했습니다.';
        });
    },

    normalizePlate,

    // 카메라별 영상 주소 - 대조 목록 + 세션 id + 카메라 전달. 세션이 바뀌면 처음부터 새로 재생
    streamUrl(cam) {
      const targets = encodeURIComponent(this.targetPlates.join(','));
      return `${OCR_STREAM_SERVER}/stream?targets=${targets}&session=${cam.sessionId}&camera=${cam.key}`;
    },

    // 영상 칸 하나를 전체 화면으로 - Esc로 원래 화면 복귀
    openFullscreen(key) {
      const el = this.videoEls[key];
      if (el && el.requestFullscreen) el.requestFullscreen();
    },

    // 카메라 하나를 처음부터 재생 - 그 카메라 결과만 초기화
    playStream(cam) {
      cam.streaming = false;
      cam.paused = false;
      cam.result = null;
      cam.sentIds = [];
      cam.detected = [];

      // <img>를 한 번 없앴다가 새 세션 주소로 다시 붙여서 영상을 처음부터 재생
      this.$nextTick(() => {
        // 무작위 값을 앞에 둠 - OCR 서버가 앞 8글자로 박스를 구분하므로 동시 재생끼리 겹치지 않게
        cam.sessionId = `${Math.random().toString(36).slice(2, 10)}${cam.key}${Date.now().toString(36)}`;
        cam.streaming = true;
        this.startPolling();
      });
    },

    // 1초마다 재생 중인 카메라의 발견 목록(/matches) 확인 -> 새 발견은 fa_server에 저장
    startPolling() {
      if (this.pollTimer) return;
      this.pollTimer = setInterval(() => {
        for (const cam of this.cameras.filter(c => c.streaming)) {
          axios.get(`${OCR_STREAM_SERVER}/matches`, { params: { session: cam.sessionId } })
            .then(resp => {
              for (const match of resp.data || []) {
                if (match.detection_id && !cam.sentIds.includes(match.detection_id)) {
                  cam.sentIds.push(match.detection_id);
                  this.saveDetection(cam, match);
                }
              }
            })
            .catch(() => {}); // OCR 서버 응답이 없어도 무시(다음 조회에서 재시도)
        }
      }, 1000);
    },

    // 재생 조작 - OCR 서버 /control (그 카메라 세션만)
    sendControl(cam, action, sec = 0) {
      return axios.get(`${OCR_STREAM_SERVER}/control`, { params: { session: cam.sessionId, action, sec } })
        .catch(err => console.error('재생 조작 실패:', err));
    },

    togglePause(cam) {
      cam.paused = !cam.paused;
      this.sendControl(cam, cam.paused ? 'pause' : 'resume');
    },

    // sec초 이동(음수면 뒤로). 일시정지 중이면 이동한 장면에서 계속 멈춰 있음
    seek(cam, sec) {
      this.sendControl(cam, 'seek', sec);
    },

    // 발견 결과를 내 fa_server(POST /cctv/detection)로 보내 내 DB에 관측 기록 + 알림 저장.
    // 응답(위치 이름 포함)으로 그 카메라의 결과 칸 갱신. 같은 차가 두 카메라에 모두 잡혀도
    // fa_server 10분 중복 방지로 DB에는 1건만 저장됨. 저장 실패해도 영상/표시는 유지
    saveDetection(cam, match) {
      const plate = normalizePlate(match.vehicle_number);
      if (!cam.detected.includes(plate)) cam.detected.push(plate);
      cam.result = { ...match, node_name: match.node_id };
      axios.post(`${FA_SERVER}/cctv/detection`, {
        vehicle_number: match.vehicle_number,
        confidence: match.confidence,
        node_id: match.node_id,
      })
        .then(resp => {
          cam.result = resp.data;
        })
        .catch(err => {
          console.error('탐지 결과 저장 실패:', err);
        });
    },
  },
};
</script>

<style scoped>
.content {
  max-width: 1920px;  /* 영상 2칸이 충분히 크게 보이도록 넓게 */
  margin: 24px auto 0;
  padding: 0 16px;
}

.card {
  background: #fff;
  border-radius: 14px;
  box-shadow: 0 2px 10px rgba(20, 30, 60, 0.06);
  padding: 20px 22px;
}

.card-title {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 16px;
}

.card-title h2 {
  font-size: 17px;
  font-weight: 700;
  color: var(--navy);
  margin: 0;
}

.video-controls {
  margin-bottom: 16px;
  display: flex;
  align-items: center;
  gap: 16px;
  font-size: 14px;
  color: var(--text);
}

.case-list {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
}

.case-badge {
  font-size: 13px;
}

.badge-detected {
  background: #feecef;
  color: var(--danger);
}

/* 카메라 2칸 - 같은 너비, 좁은 화면(1100px 이하)에서는 위아래로 쌓음 */
.camera-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 16px;
}

@media (max-width: 1100px) {
  .camera-grid {
    grid-template-columns: minmax(0, 1fr);
  }
}

.camera-col {
  display: flex;
  flex-direction: column;
  min-width: 0;
}

.camera-title {
  display: flex;
  align-items: baseline;
  gap: 8px;
  margin-bottom: 10px;
}

.camera-title h3 {
  font-size: 15px;
  font-weight: 700;
  margin: 0;
  color: var(--text);
}

.camera-desc {
  font-size: 12px;
  color: var(--muted);
}

/* 카메라별 조작 버튼 줄 */
.camera-controls {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 6px;
  margin-bottom: 10px;
}

.btn-sm {
  padding: 6px 14px;
  font-size: 13px;
}

.btn-ctl {
  border: 1px solid var(--line);
  background: #fff;
  color: var(--text);
  border-radius: 6px;
  padding: 5px 10px;
  font-size: 12px;
  cursor: pointer;
  white-space: nowrap;
}

.btn-ctl:hover:not(:disabled) {
  background: var(--blue-bg);
  border-color: var(--blue-line);
  color: var(--navy);
}

.btn-ctl:disabled {
  color: var(--gray-light);
  cursor: not-allowed;
}

.btn-fullscreen {
  margin-left: auto;  /* 크게 보기는 오른쪽 끝으로 */
}

/* 전체 화면일 때는 화면 크기에 맞춰 영상 표시(모서리 둥글기 제거) */
.video-wrapper:fullscreen {
  border-radius: 0;
  aspect-ratio: auto;
}

/* 영상 칸은 16:9 고정 - 칸 너비가 바뀌어도 영상이 찌그러지지 않음 */
.video-wrapper {
  background: #000;
  border-radius: 10px;
  overflow: hidden;
  aspect-ratio: 16 / 9;
  display: flex;
  align-items: center;
  justify-content: center;
}

.video-wrapper video,
.video-wrapper .live-stream {
  width: 100%;
  height: 100%;
  object-fit: contain;
  display: block;
}

.btn {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  border: none;
  border-radius: 8px;
  padding: 9px 18px;
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  white-space: nowrap;
}

.btn-primary {
  background: var(--navy);
  color: #fff;
}

.btn-primary:hover:not(:disabled) {
  opacity: 0.88;
}

.btn-primary:disabled {
  background: var(--blue-line);
  cursor: not-allowed;
}

/* 탐지 결과 패널 */
.result-panel {
  margin-top: 24px;
  padding-top: 20px;
  border-top: 1px solid #f0f2f7;
}

.result-title {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 14px;
}

.result-title h3 {
  font-size: 15px;
  font-weight: 700;
  margin: 0;
  color: var(--text);
}

.result-list {
  margin: 0;
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px 24px;
  background: #f7f9fc;
  border-radius: 10px;
  padding: 16px 20px;
}

.result-row {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.result-row dt {
  font-size: 12px;
  color: var(--muted);
}

.result-row dd {
  margin: 0;
  font-size: 14px;
  font-weight: 600;
  color: var(--text);
}

.badge {
  display: inline-block;
  background: var(--blue-bg);
  color: var(--navy);
  font-size: 12px;
  font-weight: 600;
  padding: 5px 10px;
  border-radius: 999px;
}

.info-placeholder {
  color: var(--muted);
  font-size: 14px;
  text-align: center;
  padding: 30px 0;
}
</style>