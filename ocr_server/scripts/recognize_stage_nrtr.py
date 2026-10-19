"""NRTR(어텐션) 헤드를 추론에서도 사용해보는 실험용 모듈. 기존 recognize_stage.py는
전혀 안 건드림(완전히 별도 파일). CTC 헤드 결과랑 NRTR 헤드 결과를 둘 다 뽑아서
비교할 수 있게 만듦.

MultiHead.forward()가 eval 모드에서 "if not self.training: return ctc_out"로
NRTR(gtc) 브랜치를 아예 건너뛰길래, 그 인스턴스의 forward만 우회하도록 몽키패치함
(model.eval() 상태 그대로 유지 - BatchNorm/Dropout 안 건드림, gtc_head 자체는
Transformer.forward_test()가 self.training=False일 때 알아서 자동회귀 생성하도록
이미 구현되어 있어서 그거 그대로 씀).
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
PADDLEOCR_DIR = ROOT / "PaddleOCR"
sys.path.insert(0, str(PADDLEOCR_DIR))

if "torch" not in sys.modules:
    _fake_torch = types.ModuleType("torch")
    _fake_torch.Tensor = type("Tensor", (), {})
    _fake_torch.from_numpy = lambda arr: arr
    sys.modules["torch"] = _fake_torch

import paddle  # noqa: E402
from ppocr.data import create_operators, transform  # noqa: E402
from ppocr.modeling.architectures import build_model  # noqa: E402
from ppocr.postprocess import build_post_process  # noqa: E402
from ppocr.postprocess.rec_postprocess import NRTRLabelDecode  # noqa: E402
from ppocr.utils.save_load import load_model  # noqa: E402

OCR_CONFIG_PATH = ROOT / "configs" / "plate_ocr_v5.yml"
OCR_CHECKPOINT = ROOT / "output" / "plate_rec_v5" / "best_accuracy"


def imread_unicode(path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _ctc_char_min(ctc_probs: np.ndarray) -> list[float]:
    """CTC 결과에서 글자별 점수 중 가장 낮은 값(배치 각 장마다).
    CTCLabelDecode와 같은 기준(연속 중복·blank(0) 제거 후 남은 글자)으로 계산."""
    idx, prob = ctc_probs.argmax(axis=2), ctc_probs.max(axis=2)
    out = []
    for b in range(len(idx)):
        sel = np.ones(len(idx[b]), dtype=bool)
        sel[1:] = idx[b][1:] != idx[b][:-1]
        sel &= idx[b] != 0
        conf = prob[b][sel]
        out.append(float(conf.min()) if conf.size else 0.0)
    return out


def _nrtr_char_min(dec_seq: np.ndarray, dec_prob: np.ndarray, character: list) -> list[float]:
    """NRTR 결과에서 글자별 점수 중 가장 낮은 값(배치 각 장마다). NRTRLabelDecode와 같은 기준
    (시작 토큰 제외, 끝 토큰 </s> 전까지)으로 계산."""
    if dec_seq[0][0] == 2:  # 맨 앞 시작 토큰(<s>) 제외
        dec_seq, dec_prob = dec_seq[:, 1:], dec_prob[:, 1:]
    out = []
    for b in range(len(dec_seq)):
        conf = []
        for k, char_id in enumerate(dec_seq[b]):
            if int(char_id) >= len(character):
                continue
            if character[int(char_id)] == "</s>":
                break
            conf.append(dec_prob[b][k])
        out.append(float(min(conf)) if conf else 0.0)
    return out


def _patched_multihead_forward(self, x, targets=None):
    """MultiHead.forward를 대체 - eval 모드에서도 NRTR(gtc) 브랜치 계산."""
    if self.use_pool:
        x = self.pool(x.reshape([0, 3, -1, self.in_channels]).transpose([0, 3, 1, 2]))
    ctc_encoder = self.ctc_encoder(x)
    ctc_out = self.ctc_head(ctc_encoder, targets)
    gtc_out = self.gtc_head(self.before_gtc(x))  # targets=None -> forward_test(자동회귀 생성)
    return {"ctc": ctc_out, "gtc": gtc_out}


class DualHeadRecognizer:
    """CTC랑 NRTR 결과를 둘 다 뽑아서 비교할 수 있게 하는 클래스."""

    def __init__(self, config_path=OCR_CONFIG_PATH, checkpoint=OCR_CHECKPOINT, device: str | None = None):
        with open(config_path, encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.config["Global"]["checkpoints"] = str(checkpoint)
        self.config["Global"]["pretrained_model"] = None
        # yml 파일 안의 character_dict_path가 학습한 PC 기준 절대경로로 박혀있어서
        # 다른 컴퓨터/계정으로 옮기면 깨짐 -> 실행 시점에 항상 이 컴퓨터 기준 경로로 덮어씀
        self.config["Global"]["character_dict_path"] = str(ROOT / "datasets" / "plate_ocr" / "plate_dict.txt")
        if device is not None:
            use_gpu = device != "cpu"
        else:
            use_gpu = paddle.device.is_compiled_with_cuda()
        self.config["Global"]["use_gpu"] = use_gpu
        paddle.set_device("gpu" if use_gpu else "cpu")

        global_config = self.config["Global"]
        self.ctc_post_process = build_post_process(self.config["PostProcess"], global_config)
        char_num = len(getattr(self.ctc_post_process, "character"))
        self.config["Architecture"]["Head"]["out_channels_list"] = {
            "CTCLabelDecode": char_num,
            "SARLabelDecode": char_num + 2,
            "NRTRLabelDecode": char_num + 3,
        }

        self.model = build_model(self.config["Architecture"])
        load_model(self.config, self.model)
        self.model.eval()

        # NRTR 디코더용 후처리 (CTC랑 별도)
        self.nrtr_post_process = NRTRLabelDecode(
            character_dict_path=global_config["character_dict_path"],
            use_space_char=global_config.get("use_space_char", False),
        )

        # 몽키패치: 이 모델 인스턴스의 head.forward만 교체 (다른 모델/전역 영향 없음)
        self.model.head.forward = types.MethodType(_patched_multihead_forward, self.model.head)

        transforms = []
        for op in self.config["Eval"]["dataset"]["transforms"]:
            op_name = list(op)[0]
            if "Label" in op_name:
                continue
            if op_name == "RecResizeImg":
                op[op_name]["infer_mode"] = True
            elif op_name == "KeepKeys":
                op[op_name]["keep_keys"] = ["image"]
            transforms.append(op)
        global_config["infer_mode"] = True
        self.ops = create_operators(transforms, global_config)

    def recognize(self, crop_bgr: np.ndarray) -> dict:
        results = self.recognize_batch([crop_bgr])
        return results[0]

    def recognize_batch(self, crops_bgr: list[np.ndarray]) -> list[dict]:
        """여러 장을 한 번에 GPU에 넣어서 처리(한 장씩 처리하는 것보다 훨씬 빠름).
        전처리(RecResizeImg)는 보통 48x320으로 맞추지만, 가로:세로가 320/48(약 6.67)을 넘는
        이미지(화면 가장자리에 걸려 얇게 잘린 번호판 등)는 폭이 더 넓게 나옴. 폭이 다른 걸 한 묶음으로
        쌓으면 에러가 나서 묶음 전체가 버려지므로, 같은 크기끼리 따로 묶어서 처리함
        (정상 크기 이미지는 기존과 똑같이 처리됨)."""
        # *_score: 글자별 점수의 평균(PaddleOCR 기본), *_min: 글자별 점수 중 가장 낮은 값
        empty = {"ctc_text": "", "ctc_score": 0.0, "ctc_min": 0.0, "nrtr_text": "", "nrtr_score": 0.0, "nrtr_min": 0.0}
        if not crops_bgr:
            return []

        groups: dict[tuple, list[tuple[int, np.ndarray]]] = {}  # 이미지 크기 -> [(원래 순서, 배열), ...]
        for i, crop_bgr in enumerate(crops_bgr):
            ok, buf = cv2.imencode(".png", crop_bgr)
            if not ok:
                continue
            batch = transform({"image": buf.tobytes()}, self.ops)
            groups.setdefault(batch[0].shape, []).append((i, batch[0]))

        results = [dict(empty) for _ in crops_bgr]
        for items in groups.values():
            images = paddle.to_tensor(np.stack([arr for _, arr in items], axis=0))
            preds = self.model(images)  # 패치된 forward라 dict{"ctc":..., "gtc":[dec_seq,dec_prob]}

            ctc_results = self.ctc_post_process(preds["ctc"])
            dec_seq, dec_prob = preds["gtc"]
            nrtr_results = self.nrtr_post_process([dec_seq, dec_prob])
            ctc_mins = _ctc_char_min(preds["ctc"].numpy())
            nrtr_mins = _nrtr_char_min(dec_seq.numpy(), dec_prob.numpy(), self.nrtr_post_process.character)

            for j, (orig_i, _) in enumerate(items):
                ctc_text, ctc_score = ctc_results[j]
                nrtr_text, nrtr_score = nrtr_results[j]
                results[orig_i] = {
                    "ctc_text": ctc_text, "ctc_score": float(ctc_score), "ctc_min": ctc_mins[j],
                    "nrtr_text": nrtr_text, "nrtr_score": float(nrtr_score), "nrtr_min": nrtr_mins[j],
                }
        return results


INFER_MODEL_DIR = ROOT / "output" / "plate_rec_v5_infer"
CPU_THREADS = 4  # OCR에 쓸 CPU 스레드 수(탐지 서버와 CPU를 나눠 쓰므로 전부 쓰지 않음)


class CtcInferRecognizer:
    """CPU용 - export_ctc_infer.py로 변환한 추론 모델(CTC만)을 paddle 추론 엔진 + oneDNN으로 실행.
    DualHeadRecognizer(학습 체크포인트를 그대로 실행)는 CPU에서 1장에 약 0.9초라 실시간을 못 따라감.
    이건 약 20ms. recognize_batch 반환 형식은 DualHeadRecognizer와 같음(NRTR 결과는 빈 값)."""

    def __init__(self, model_dir=INFER_MODEL_DIR, config_path=OCR_CONFIG_PATH, cpu_threads=CPU_THREADS):
        from paddle import inference

        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        global_config = config["Global"]
        # yml 안의 경로가 다른 PC 기준 절대경로라 이 PC 기준으로 덮어씀(DualHeadRecognizer와 같은 이유)
        global_config["character_dict_path"] = str(ROOT / "datasets" / "plate_ocr" / "plate_dict.txt")
        global_config["infer_mode"] = True
        self.post_process = build_post_process(config["PostProcess"], global_config)

        # 전처리 - DualHeadRecognizer와 같은 방식(라벨 관련 단계 제외)
        transforms = []
        for op in config["Eval"]["dataset"]["transforms"]:
            op_name = list(op)[0]
            if "Label" in op_name:
                continue
            if op_name == "RecResizeImg":
                op[op_name]["infer_mode"] = True
            elif op_name == "KeepKeys":
                op[op_name]["keep_keys"] = ["image"]
            transforms.append(op)
        self.ops = create_operators(transforms, global_config)

        infer_config = inference.Config(str(model_dir / "model.json"), str(model_dir / "model.pdiparams"))
        infer_config.disable_gpu()
        infer_config.enable_mkldnn()  # oneDNN - Intel CPU 가속
        infer_config.set_cpu_math_library_num_threads(cpu_threads)
        infer_config.disable_glog_info()
        self.predictor = inference.create_predictor(infer_config)
        self.input = self.predictor.get_input_handle(self.predictor.get_input_names()[0])
        self.output = self.predictor.get_output_handle(self.predictor.get_output_names()[0])

    def recognize_batch(self, crops_bgr: list[np.ndarray]) -> list[dict]:
        """여러 장을 한 번에 처리. 폭이 다른 이미지는 같은 크기끼리 따로 묶음(DualHeadRecognizer와 같은 이유)."""
        # *_score: 글자별 점수의 평균, *_min: 글자별 점수 중 가장 낮은 값(DualHeadRecognizer와 같은 형식)
        empty = {"ctc_text": "", "ctc_score": 0.0, "ctc_min": 0.0, "nrtr_text": "", "nrtr_score": 0.0, "nrtr_min": 0.0}
        if not crops_bgr:
            return []

        groups: dict[tuple, list[tuple[int, np.ndarray]]] = {}  # 이미지 크기 -> [(원래 순서, 배열), ...]
        for i, crop_bgr in enumerate(crops_bgr):
            ok, buf = cv2.imencode(".png", crop_bgr)
            if not ok:
                continue
            batch = transform({"image": buf.tobytes()}, self.ops)
            groups.setdefault(batch[0].shape, []).append((i, batch[0]))

        results = [dict(empty) for _ in crops_bgr]
        for items in groups.values():
            images = np.stack([arr for _, arr in items], axis=0).astype("float32")
            self.input.reshape(list(images.shape))
            self.input.copy_from_cpu(images)
            self.predictor.run()
            preds = self.output.copy_to_cpu()

            ctc_mins = _ctc_char_min(preds)
            for (orig_i, _), (text, score), ctc_min in zip(items, self.post_process(preds), ctc_mins):
                results[orig_i] = {**empty, "ctc_text": text, "ctc_score": float(score), "ctc_min": ctc_min}
        return results
