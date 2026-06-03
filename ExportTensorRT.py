"""
export_tensorrt.py
──────────────────
Xuất UNet PyTorch checkpoint → TensorRT engine (.trt)

Tương thích:
    TensorRT 8.x  — dùng EXPLICIT_BATCH flag (tự động detect)
    TensorRT 10+  — EXPLICIT_BATCH bị bỏ, create_network() không cần flag

Yêu cầu:
    pip install tensorrt pycuda onnx

Quy trình:
    PyTorch .pth  →  ONNX (1 file duy nhất)  →  TensorRT engine

Cách dùng:
    python export_tensorrt.py
    python export_tensorrt.py --fp16
    python export_tensorrt.py --tile_h 1200 --tile_w 1200 --fp16
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from unet import UNet
# from unet.student_unet_model import StudentUNet

# ─────────────────────────────────────────────────────────────────────────────
# Config mặc định
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_CHECKPOINT  = "savecheckpoints/checkpoints20260526_20-40/checkpoint_epoch20.pth"
DEFAULT_ONNX        = "savecheckpoints/checkpoints20260526_20-40/engine/unet.onnx"
DEFAULT_ENGINE      = "savecheckpoints/checkpoints20260526_20-40/engine/unet_fp32.trt"
DEFAULT_ENGINE_FP16 = "savecheckpoints/checkpoints20260526_20-40/engine/unet_fp16.trt"

PAD    = 100   # reflect padding (px mỗi cạnh) — baked vào ONNX graph
TILE_H = 1200  # chiều cao tile mẫu để TRT tối ưu hoá
TILE_W = 1200  # chiều rộng tile mẫu


# ─────────────────────────────────────────────────────────────────────────────
# Helper: detect TensorRT major version
# ─────────────────────────────────────────────────────────────────────────────
def _trt_major(trt) -> int:
    try:
        return int(trt.__version__.split('.')[0])
    except Exception:
        return 8


# ─────────────────────────────────────────────────────────────────────────────
# Wrapper: nhúng reflect padding vào ONNX graph
# ─────────────────────────────────────────────────────────────────────────────
class UNetWithPad(torch.nn.Module):
    """
    Input  : (1, 1, H, W)  — tile gốc, chưa pad
    Output : (1, 1, H, W)  — enhanced, đã crop về kích thước gốc
    Padding reflect được xử lý bên trong → không cần thêm ngoài engine.
    """
    def __init__(self, model: torch.nn.Module, pad: int):
        super().__init__()
        self.model = model
        self.pad   = pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p     = self.pad
        x_pad = F.pad(x, (p, p, p, p), mode='reflect')
        res   = self.model(x_pad)
        out   = x_pad * (1 + res)
        return out[:, :, p:-p, p:-p]


# ─────────────────────────────────────────────────────────────────────────────
# Bước 1 — PyTorch → ONNX (1 file duy nhất, không external data)
# ─────────────────────────────────────────────────────────────────────────────
def export_onnx(checkpoint_path: str, onnx_path: str,
                tile_h: int, tile_w: int, pad: int) -> None:
    print(f"\n{'='*60}")
    print("[1/3] Xuất ONNX")
    print(f"  checkpoint : {checkpoint_path}")
    print(f"  tile size  : {tile_h} x {tile_w}  (pad={pad}px reflect)")
    print(f"  onnx output: {onnx_path}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  device     : {device}")
    print(f"  torch      : {torch.__version__}")

    base = UNet(n_channels=1, n_classes=1)
    base.load_state_dict(torch.load(checkpoint_path, map_location=device))
    base.to(device).eval()

    wrapped = UNetWithPad(base, pad).to(device).eval()

    # Dummy input: kích thước cố định tile + pad (legacy tracer không cần dynamic dummy)
    dummy = torch.zeros(1, 1, tile_h, tile_w, device=device)

    Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)

    # ── Export bằng legacy TorchScript tracer ────────────────────────────────
    # Lý do dùng dynamo=False:
    #   - Torch >= 2.2 mặc định bật dynamo exporter mới
    #   - Dynamo không hỗ trợ dynamic_axes cú pháp cũ, và thường fail với
    #     opset < 18 do lỗi version converter của onnxscript
    #   - Legacy tracer (TorchScript) ổn định hơn, hỗ trợ đầy đủ dynamic_axes
    #     và tương thích tốt với TensorRT OnnxParser
    with torch.no_grad():
        torch.onnx.export(
            wrapped,
            dummy,
            onnx_path,
            export_params       = True,
            opset_version       = 18,   # opset 18: mức thấp nhất torch mới hỗ trợ ổn định
            do_constant_folding = True,
            input_names         = ['input'],
            output_names        = ['output'],
            dynamic_axes        = {     # cho phép H, W thay đổi lúc TRT inference
                'input' : {2: 'height', 3: 'width'},
                'output': {2: 'height', 3: 'width'},
            },
            dynamo = False,             # BẮT BUỘC: dùng legacy tracer, không dynamo
        )

    print(f"  ✓ ONNX export xong: {onnx_path}")

    # ── Merge external data vào 1 file nếu bị split ──────────────────────────
    # torch.onnx.export đôi khi tách weights lớn ra file .onnx.data riêng
    # → TensorRT OnnxParser chỉ đọc được 1 file duy nhất, cần gom lại.
    try:
        import onnx
        from onnx.external_data_helper import load_external_data_for_model

        # load_external_data=False để đọc proto nhanh, không load weights vào RAM ngay
        model_proto = onnx.load(onnx_path, load_external_data=False)

        has_external = any(
            t.HasField('data_location') and t.data_location == onnx.TensorProto.EXTERNAL
            for t in model_proto.graph.initializer
        )

        if has_external:
            print("  ⚠ Phát hiện external data — đang merge vào 1 file...")
            # FIX: load_external_data_for_model yêu cầu str, KHÔNG phải Path object
            onnx_dir = str(Path(onnx_path).parent)
            load_external_data_for_model(model_proto, onnx_dir)
            # Ghi lại thành 1 file duy nhất (weights nhúng trực tiếp vào .onnx)
            onnx.save(model_proto, onnx_path)
            # Dọn dẹp tất cả file .data thừa trong cùng thư mục
            for stale in Path(onnx_path).parent.glob("*.data"):
                stale.unlink()
                print(f"  ✓ Đã xóa file thừa: {stale.name}")
            print("  ✓ Merge xong — ONNX là 1 file duy nhất")
            # Reload proto sau khi ghi lại để validate
            model_proto = onnx.load(onnx_path)
        else:
            print("  ✓ ONNX xuất 1 file (không có external data)")

        onnx.checker.check_model(model_proto)
        print(f"  ✓ ONNX validation OK  →  {onnx_path}")

    except ImportError:
        print("  ⚠ Chạy: pip install onnx   để auto-merge external data và validate")
        print(f"  ✓ ONNX saved: {onnx_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Bước 2 — ONNX → TensorRT engine
# ─────────────────────────────────────────────────────────────────────────────
def build_engine(onnx_path: str, engine_path: str,
                 tile_h: int, tile_w: int,
                 fp16: bool = False) -> None:
    print(f"\n{'='*60}")
    print(f"[2/3] Build TensorRT engine  (fp16={'ON' if fp16 else 'OFF'})")
    print(f"  onnx   : {onnx_path}")
    print(f"  engine : {engine_path}")

    try:
        import tensorrt as trt
    except ImportError:
        raise RuntimeError(
            "TensorRT chưa cài.\n"
            "Xem: https://docs.nvidia.com/deeplearning/tensorrt/install-guide/"
        )

    ver = _trt_major(trt)
    print(f"  TensorRT: {trt.__version__}  (major={ver})")

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder    = trt.Builder(TRT_LOGGER)

    # TRT 8.x : EXPLICIT_BATCH phải truyền tường minh vào create_network()
    # TRT 10+  : EXPLICIT_BATCH là mặc định, create_network() không nhận tham số
    if ver >= 10:
        network = builder.create_network()
    else:
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )

    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, 'rb') as f:
        raw = f.read()

    if not parser.parse(raw):
        for i in range(parser.num_errors):
            print(f"  ✗ {parser.get_error(i)}")
        raise RuntimeError("ONNX parse thất bại — xem lỗi bên trên")

    print(f"  ✓ ONNX parsed OK  ({len(raw)/1024/1024:.1f} MB)")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4 GB

    if fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("  ✓ FP16 enabled")
        else:
            print("  ⚠ GPU không hỗ trợ fast FP16 — fallback FP32")

    # Dynamic shape profile
    # min/max bao phủ range tile thực tế; opt = kích thước thường gặp nhất
    profile  = builder.create_optimization_profile()
    min_h, opt_h, max_h = 64, tile_h, 2048
    min_w, opt_w, max_w = 64, tile_w, 2048

    profile.set_shape(
        'input',
        min=(1, 1, min_h, min_w),
        opt=(1, 1, opt_h, opt_w),
        max=(1, 1, max_h, max_w),
    )
    config.add_optimization_profile(profile)

    print(f"  Dynamic shape: "
          f"min(1,1,{min_h},{min_w})  opt(1,1,{opt_h},{opt_w})  max(1,1,{max_h},{max_w})")
    print("  Building... (lần đầu mất vài phút)")

    t0         = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Build engine thất bại — kiểm tra VRAM và workspace size")

    Path(engine_path).parent.mkdir(parents=True, exist_ok=True)
    with open(engine_path, 'wb') as f:
        f.write(serialized)

    size_mb = Path(engine_path).stat().st_size / (1024 ** 2)
    print(f"  ✓ Engine saved: {engine_path}  ({size_mb:.1f} MB, {time.time()-t0:.1f}s)")


# ─────────────────────────────────────────────────────────────────────────────
# Bước 3 — Kiểm tra engine nhanh
# ─────────────────────────────────────────────────────────────────────────────
def verify_engine(engine_path: str, tile_h: int, tile_w: int) -> None:
    print(f"\n{'='*60}")
    print("[3/3] Kiểm tra engine")

    try:
        import tensorrt as trt
        import pycuda.autoinit   # noqa: F401
        import pycuda.driver as cuda
    except ImportError as e:
        print(f"  ⚠ Bỏ qua verify: {e}")
        return

    ver        = _trt_major(trt)
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    runtime    = trt.Runtime(TRT_LOGGER)

    with open(engine_path, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()
    context.set_input_shape('input', (1, 1, tile_h, tile_w))

    in_np  = np.random.rand(1, 1, tile_h, tile_w).astype(np.float32)
    out_np = np.empty_like(in_np)
    d_in   = cuda.mem_alloc(in_np.nbytes)
    d_out  = cuda.mem_alloc(out_np.nbytes)
    stream = cuda.Stream()

    cuda.memcpy_htod_async(d_in, in_np, stream)

    # TRT 10+ dùng execute_async_v3 + set_tensor_address
    # TRT 8/9  dùng execute_async_v2 + bindings list
    if ver >= 10:
        context.set_tensor_address('input',  int(d_in))
        context.set_tensor_address('output', int(d_out))
        context.execute_async_v3(stream.handle)
    else:
        context.execute_async_v2(
            bindings      = [int(d_in), int(d_out)],
            stream_handle = stream.handle,
        )

    cuda.memcpy_dtoh_async(out_np, d_out, stream)
    stream.synchronize()

    print(f"  Input  : {in_np.shape}  range [{in_np.min():.3f}, {in_np.max():.3f}]")
    print(f"  Output : {out_np.shape}  range [{out_np.min():.4f}, {out_np.max():.4f}]")
    assert out_np.shape == in_np.shape, "Shape mismatch!"
    print("  ✓ Engine chạy OK")

    d_in.free()
    d_out.free()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    ap = argparse.ArgumentParser(
        description="Export UNet → ONNX (single-file) → TensorRT engine"
    )
    ap.add_argument('--checkpoint',  default=DEFAULT_CHECKPOINT)
    ap.add_argument('--onnx',        default=DEFAULT_ONNX)
    ap.add_argument('--engine',      default=None,
                    help="Path .trt output (mặc định tự chọn theo --fp16)")
    ap.add_argument('--tile_h',      type=int, default=TILE_H)
    ap.add_argument('--tile_w',      type=int, default=TILE_W)
    ap.add_argument('--pad',         type=int, default=PAD)
    ap.add_argument('--fp16',        action='store_true',
                    help="Bật FP16 (~2x nhanh hơn, cần GPU Turing+)")
    ap.add_argument('--skip_verify', action='store_true')
    return ap.parse_args()


if __name__ == '__main__':
    args = parse_args()

    engine_path = args.engine or (DEFAULT_ENGINE_FP16 if args.fp16 else DEFAULT_ENGINE)

    print("\n╔══════════════════════════════════════════════════╗")
    print("║         UNet → ONNX → TensorRT Export           ║")
    print("╚══════════════════════════════════════════════════╝")
    print(f"  precision : {'FP16' if args.fp16 else 'FP32'}")
    print(f"  tile size : {args.tile_h} x {args.tile_w}")
    print(f"  pad       : {args.pad} px reflect (baked vào graph)")

    export_onnx(
        checkpoint_path = args.checkpoint,
        onnx_path       = args.onnx,
        tile_h          = args.tile_h,
        tile_w          = args.tile_w,
        pad             = args.pad,
    )

    build_engine(
        onnx_path   = args.onnx,
        engine_path = engine_path,
        tile_h      = args.tile_h,
        tile_w      = args.tile_w,
        fp16        = args.fp16,
    )

    if not args.skip_verify:
        verify_engine(engine_path, args.tile_h, args.tile_w)

    print(f"\n{'='*60}")
    print("Xong! Chạy inference:")
    print(f"  python InferenceImage_trt.py --engine {engine_path}")