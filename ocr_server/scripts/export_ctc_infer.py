"""학습 체크포인트(best_accuracy.pdparams) -> CPU용 추론 모델(CTC만) 변환. 한 번만 실행.

학습 방식(동적 그래프)으로 돌리면 CPU에서 번호판 1장에 약 0.9초 걸림.
추론 모델 + oneDNN으로 돌리면 약 20ms라서 live_ocr_worker가 이 모델을 쓰게 함.
NRTR 헤드는 빼고 CTC 헤드만 내보냄(NRTR은 글자를 한 자씩 생성해서 CPU에서 느림).

실행: ocr_env/Scripts/python.exe scripts/export_ctc_infer.py
결과: output/plate_rec_v5_infer/model.json, model.pdiparams
"""
import types

import numpy as np
import paddle

from recognize_stage_nrtr import ROOT, DualHeadRecognizer

EXPORT_DIR = ROOT / "output" / "plate_rec_v5_infer"


def _ctc_only_forward(self, x, targets=None):
    """MultiHead.forward 대체 - CTC 브랜치만 계산(NRTR 브랜치 제외)."""
    if self.use_pool:
        x = self.pool(x.reshape([0, 3, -1, self.in_channels]).transpose([0, 3, 1, 2]))
    return self.ctc_head(self.ctc_encoder(x), targets)


def main():
    recognizer = DualHeadRecognizer(device="cpu")
    model = recognizer.model
    model.head.forward = types.MethodType(_ctc_only_forward, model.head)

    # 변환 전 결과 - 변환 후 결과와 비교용
    sample = np.random.rand(2, 3, 48, 320).astype("float32")
    before = model(paddle.to_tensor(sample)).numpy()

    # PPLCNetV3의 학습용 분기 구조를 추론용 단일 구조로 합침(결과는 같고 속도만 빨라짐)
    for layer in model.sublayers():
        if hasattr(layer, "rep") and not getattr(layer, "is_repped", False):
            layer.rep()

    static_model = paddle.jit.to_static(
        model,
        input_spec=[paddle.static.InputSpec([None, 3, 48, -1], "float32")],
        full_graph=True,
    )
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    paddle.jit.save(static_model, str(EXPORT_DIR / "model"))

    after = static_model(paddle.to_tensor(sample)).numpy()
    diff = float(np.abs(before - after).max())
    print(f"저장 완료: {EXPORT_DIR}")
    print(f"변환 전후 최대 차이: {diff:.6f} (0.001보다 작으면 정상)")


if __name__ == "__main__":
    main()
