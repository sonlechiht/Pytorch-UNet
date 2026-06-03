"""
InferenceImage_trt.py
──────────────────────
Inference UNet bằng TensorRT engine — nhanh hơn PyTorch ~3-5x.

Yêu cầu:
    pip install tensorrt pycuda

Cách dùng:
    # FP32
    python InferenceImage_trt.py --engine Result/engine/unet_fp32.trt

    # FP16
    python InferenceImage_trt.py --engine Result/engine/unet_fp16.trt

    # Toàn bộ tham số
    python InferenceImage_trt.py \\
        --engine   Result/engine/unet_fp16.trt \\
        --nk       3 \\
        --min_val  54175 \\
        --max_val  56989 \\
        --input_dir  "G:/AutoImageProcessing/WaferData/Image9/Invert_20260422023152" \\
        --output_dir Result/output_trt

Lưu ý: padding reflect đã được baked vào ONNX/TRT graph bởi export_tensorrt.py.
        File này KHÔNG cần tự thêm padding nữa.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import cv2

# ─────────────────────────────────────────────────────────────────────────────
# Config mặc định
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_ENGINE    = "savecheckpoints/checkpoints20260526_20-40/engine/unet_fp32.trt"
DEFAULT_INPUT_DIR = r"G:\AutoImageProcessing\WaferData\Image9\Invert_20260422023152"
DEFAULT_OUTPUT_DIR= "Result/output_trt"
SIZE_THRESHOLD    = 2000   # chia tile nếu H hoặc W vượt ngưỡng


# ─────────────────────────────────────────────────────────────────────────────
# Normalize helpers (giữ nguyên từ v2)
# ─────────────────────────────────────────────────────────────────────────────
def normalize_wafer(img):
    p1  = np.percentile(img, 0.1)
    p99 = np.percentile(img, 99.9)
    if p1 == p99:
        p1, p99 = img.min(), img.max()
    return np.clip((img - p1) / (p99 - p1), 0, 1)

def normalize_wafer_fixed(img, p1, p99):
    return np.clip((img - p1) / (p99 - p1), 0, 1)

def load_raw_image(img_path):
    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Cannot load image: {img_path}")
    return img.astype(np.float32) / 65535.0

def save_image(img, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), (img * 65535).astype(np.uint16))


# ─────────────────────────────────────────────────────────────────────────────
# TensorRT Engine wrapper
# ─────────────────────────────────────────────────────────────────────────────
class TRTEngine:
    """
    Load và chạy TensorRT engine.
    Tự động detect TRT 8.x / 10+ và dùng đúng API.
    """

    def __init__(self, engine_path: str):
        import tensorrt as trt
        import pycuda.autoinit   # noqa: F401
        import pycuda.driver as cuda

        self._trt  = trt
        self._cuda = cuda
        self._ver  = int(trt.__version__.split('.')[0])

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime    = trt.Runtime(TRT_LOGGER)

        with open(engine_path, 'rb') as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context      = self.engine.create_execution_context()
        self.stream       = cuda.Stream()
        self.input_name   = 'input'
        self.output_name  = 'output'

        print(f"  ✓ TRT engine loaded : {engine_path}")
        print(f"    TRT version       : {trt.__version__}")

    def infer(self, input_np: np.ndarray) -> np.ndarray:
        """
        input_np : float32 numpy (1, 1, H, W)
        Trả về   : float32 numpy (1, 1, H, W)
        """
        assert input_np.ndim == 4 and input_np.shape[:2] == (1, 1), \
            f"Expect (1,1,H,W), got {input_np.shape}"
        assert input_np.dtype == np.float32

        _, _, H, W = input_np.shape
        self.context.set_input_shape(self.input_name, (1, 1, H, W))

        out_np = np.empty_like(input_np)
        d_in   = self._cuda.mem_alloc(input_np.nbytes)
        d_out  = self._cuda.mem_alloc(out_np.nbytes)

        self._cuda.memcpy_htod_async(d_in, input_np, self.stream)

        # ── TRT 10+ dùng execute_async_v3 + set_tensor_address ───────────────
        # ── TRT 8/9  dùng execute_async_v2 + bindings list ───────────────────
        if self._ver >= 10:
            self.context.set_tensor_address(self.input_name,  int(d_in))
            self.context.set_tensor_address(self.output_name, int(d_out))
            self.context.execute_async_v3(self.stream.handle)
        else:
            self.context.execute_async_v2(
                bindings      = [int(d_in), int(d_out)],
                stream_handle = self.stream.handle,
            )

        self._cuda.memcpy_dtoh_async(out_np, d_out, self.stream)
        self.stream.synchronize()

        d_in.free()
        d_out.free()

        return out_np


# ─────────────────────────────────────────────────────────────────────────────
# Inference 1 tile (gọi engine, không cần thêm padding thủ công)
# ─────────────────────────────────────────────────────────────────────────────
def infer_tile_trt(engine: TRTEngine, tile_np: np.ndarray) -> np.ndarray:
    """
    tile_np : float32 (H, W), đã normalize [0,1]
    Trả về  : float32 (H, W)
    Padding reflect đã được xử lý bên trong TRT graph.
    """
    inp = tile_np[np.newaxis, np.newaxis].astype(np.float32)  # (1,1,H,W)
    out = engine.infer(inp)                                    # (1,1,H,W)
    result = np.clip(out.squeeze(), 0, 1)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Tự động chia tile nếu ảnh lớn, rồi merge
# ─────────────────────────────────────────────────────────────────────────────
def infer_image_trt(engine: TRTEngine, img_np: np.ndarray,
                    threshold: int = SIZE_THRESHOLD) -> np.ndarray:
    """
    img_np    : float32 (H, W), đã normalize [0,1]
    threshold : chia tile nếu H > threshold hoặc W > threshold
    """
    H, W = img_np.shape

    if H <= threshold and W <= threshold:
        return infer_tile_trt(engine, img_np)

    # ── Chia 4 tile ─────────────────────────────────────────────────────────
    mh, mw = H // 2, W // 2
    tiles = {
        "TL": img_np[:mh,  :mw],
        "TR": img_np[:mh,  mw:],
        "BL": img_np[mh:,  :mw],
        "BR": img_np[mh:,  mw:],
    }

    print(f"  Ảnh lớn ({H}×{W}) → chia 4 tile")

    results = {}
    for name, tile in tiles.items():
        t0 = time.perf_counter()
        results[name] = infer_tile_trt(engine, tile)
        print(f"    tile {name} ({tile.shape[0]}×{tile.shape[1]}) "
              f"→ {(time.perf_counter()-t0)*1000:.1f} ms")

    # ── Merge ────────────────────────────────────────────────────────────────
    top    = np.concatenate([results["TL"], results["TR"]], axis=1)
    bottom = np.concatenate([results["BL"], results["BR"]], axis=1)
    merged = np.concatenate([top, bottom], axis=0)
    return merged[:H, :W]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    ap = argparse.ArgumentParser(description="Inference UNet với TensorRT engine")
    ap.add_argument('--engine',     default=DEFAULT_ENGINE,
                    help="Path file .trt engine đã export")
    ap.add_argument('--input_dir',  default=DEFAULT_INPUT_DIR)
    ap.add_argument('--output_dir', default=DEFAULT_OUTPUT_DIR)
    ap.add_argument('--nk',    type=int, default=3)
    ap.add_argument('--step',  type=int, default=1)
    ap.add_argument('--min_val',    type=int, default=54175,
                    help="Giá trị normalize _min (giống InferenceImage_v2.py)")
    ap.add_argument('--max_val',    type=int, default=56989,
                    help="Giá trị normalize _max (giống InferenceImage_v2.py)")
    ap.add_argument('--threshold',  type=int, default=SIZE_THRESHOLD,
                    help="Ngưỡng chia tile (px)")
    return ap.parse_args()


if __name__ == '__main__':
    args = parse_args()

    print("\n╔══════════════════════════════════════════════════╗")
    print("║         UNet Inference — TensorRT                ║")
    print("╚══════════════════════════════════════════════════╝")
    print(f"  engine      : {args.engine}")
    print(f"  output_dir  : {args.output_dir}")
    print(f"  normalize   : [{args.min_val}, {args.max_val}]")
    print(f"  tile split  : > {args.threshold} px")

    # ── Load engine (1 lần duy nhất cho cả vòng lặp) ────────────────────────
    engine = TRTEngine(args.engine)

    nk   = args.nk
    step = args.step
    _p1  = args.min_val / 65535.0
    _p99 = args.max_val / 65535.0

    total_images = 0
    total_time   = 0.0

    for i in range(0, 4**nk, step):
        for j in range(i, i + step):
            input_path = (
                Path(args.input_dir) / f"Branch_{nk}_{j}.tif"
            )
            output_path = Path(args.output_dir) / f"Branch_{nk}_{j}.tif"

            if not input_path.exists():
                print(f"  ⚠ Không tìm thấy: {input_path} — bỏ qua")
                continue

            print(f"\nProcessing Branch_{nk}_{j} ...")
            t0 = time.perf_counter()

            # Load + normalize
            raw      = load_raw_image(input_path)
            raw_norm = normalize_wafer_fixed(raw, _p1, _p99)

            # Inference TRT
            enhanced = infer_image_trt(engine, raw_norm, args.threshold)

            # Lưu
            save_image(normalize_wafer(enhanced), output_path)

            elapsed = (time.perf_counter() - t0) * 1000
            total_time   += elapsed
            total_images += 1
            print(f"  ✓ Saved: {output_path}  ({elapsed:.1f} ms)")

    if total_images > 0:
        print(f"\n{'='*50}")
        print(f"Tổng: {total_images} ảnh  |  "
              f"Trung bình: {total_time/total_images:.1f} ms/ảnh  |  "
              f"Tổng thời gian: {total_time/1000:.2f} s")