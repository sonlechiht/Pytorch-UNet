import torch
import numpy as np
import cv2
from pathlib import Path
import time
import torch.nn.functional as F
from unet import UNet
from PIL import Image
import tifffile as tiff
# from unet.student_unet_model import StudentUNet, StudentUNet2, StudentUNet3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
PAD = 100           # reflect padding (pixels) mỗi cạnh
SIZE_THRESHOLD = 1600  # chia tile nếu H hoặc W vượt ngưỡng này

# ─────────────────────────────────────────────
# Normalize helpers
# ─────────────────────────────────────────────
def normalize_wafer(img):
    p1  = np.percentile(img, 0.1)
    p99 = np.percentile(img, 99.9)
    if p1 == p99:
        p1, p99 = img.min(), img.max()
    img = np.clip((img - p1) / (p99 - p1), 0, 1)
    return img

def normalize_wafer_min_max(img):
    p1, p99 = img.min(), img.max()
    img = np.clip((img - p1) / (p99 - p1), 0, 1)
    return img

def normalize_wafer_fixed(img, p1, p99):
    img = np.clip((img - p1) / (p99 - p1), 0, 1)
    return img

# ─────────────────────────────────────────────
# Load / save
# ─────────────────────────────────────────────
# def load_raw_image(img_path):
#     """Đọc ảnh 16-bit, trả về numpy float32 shape (H, W), chưa normalize."""
#     load_image = Image.open(img_path)
#     original_info = load_image.info.copy()
#     img = np.array(load_image)
#     # img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
#     if img is None:
#         raise RuntimeError(f"Cannot load image: {img_path}")
#     return img.astype(np.float32) / 65535.0 , original_info

# def save_image(img, path, info):
#     # cv2.imwrite(path, (img * 65535).astype(np.uint16))
#     out_img = Image.fromarray((img * 65535).astype(np.uint16))
#     out_img.save(path,**info)

def load_raw_image(img_path):
    """Đọc ảnh 16-bit, tự động trích xuất động độ phân giải và đơn vị gốc."""
    with tiff.TiffFile(img_path) as tif:
        img = tif.asarray()
        tags = tif.pages[0].tags

        # 1. Tự động lấy cấu trúc phân số XResolution và YResolution gốc
        x_res_tag = tags.get("XResolution", None)
        y_res_tag = tags.get("YResolution", None)
        x_resolution = x_res_tag.value if x_res_tag else (1344, 1)
        y_resolution = y_res_tag.value if y_res_tag else (1344, 1)

        # 2. Tự động lấy đơn vị ResolutionUnit gốc (2: Inch, 3: Centimeter...)
        # Nếu thẻ không tồn tại, mặc định lấy 3 (Centimeter) theo ảnh mẫu của bạn
        res_unit_tag = tags.get("ResolutionUnit", None)
        res_unit = res_unit_tag.value.value if res_unit_tag else 3

        # Đôi khi giá trị trả về dạng Enum, ta ép kiểu về int để an toàn
        if hasattr(res_unit, "value"):
            res_unit = res_unit.value
        res_unit = int(res_unit)

    if img is None:
        raise RuntimeError(f"Cannot load image: {img_path}")

    # Gom tất cả thông tin metadata cần thiết lại thành một tuple/dict
    meta_info = {
        "resolution": (x_resolution[0] / x_resolution[1], y_resolution[0] / y_resolution[1]),
        "resolutionunit": res_unit
    }

    return img.astype(np.float32) / 65535.0, meta_info


def save_image(img, path, meta_info):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    # Chuyển đổi ngược về ảnh 16-bit (uint16)
    img_uint16 = (img * 65535).astype(np.uint16)

    # 3. Sử dụng các tham số chuẩn tích hợp sẵn của tifffile để ghi đè thay vì extratags
    # compression=1 ứng với Uncompressed
    tiff.imwrite(
        path,
        img_uint16,
        photometric="minisblack",
        compression=1,                     # Giữ nguyên không nén (Uncompressed)
        resolution=meta_info["resolution"], # Tự động điền phân giải gốc (X, Y)
        resolutionunit=meta_info["resolutionunit"] # Tự động điền đơn vị gốc (3 hoặc số khác)
    )
# ─────────────────────────────────────────────
# Reflect padding + inference cho 1 tile
# ─────────────────────────────────────────────
def infer_tile(model, tile_np, pad=PAD):
    """
    tile_np : numpy float32 (H, W), đã normalize về [0,1]
    Trả về  : numpy float32 (H, W) — kết quả sau khi crop padding
    """
    # 1. Chuẩn bị tensor: (1, 1, H, W)
    tensor = torch.from_numpy(tile_np[np.newaxis, np.newaxis].copy()).float()

    # 2. Reflect padding (đảo ngược pixel ở cạnh — tự nhiên hơn zero/constant)
    #    F.pad thứ tự: (left, right, top, bottom)
    tensor_padded = F.pad(tensor, (pad, pad, pad, pad), mode='reflect')

    tensor_padded = tensor_padded.to(device)

    with torch.no_grad():
        residual = model(tensor_padded)
        enhanced = tensor_padded * (1 + residual)

    enhanced_np = enhanced.squeeze().cpu().numpy()
    enhanced_np = np.clip(enhanced_np, 0, 1)

    # 3. Crop bỏ vùng padding
    h_pad, w_pad = enhanced_np.shape
    enhanced_cropped = enhanced_np[pad: h_pad - pad, pad: w_pad - pad]

    return enhanced_cropped

# ─────────────────────────────────────────────
# Chia ảnh → infer từng tile → merge
# ─────────────────────────────────────────────
def infer_image(model, img_np, threshold=SIZE_THRESHOLD, pad=PAD):
    """
    img_np    : numpy float32 (H, W), đã normalize về [0,1]
    threshold : nếu H > threshold hoặc W > threshold thì chia 4 tile
    Trả về    : numpy float32 (H, W)
    """
    H, W = img_np.shape

    if H <= threshold and W <= threshold:
        # Ảnh nhỏ: inference trực tiếp
        return infer_tile(model, img_np, pad)

    # ── Chia 4 tile (TL, TR, BL, BR) ────────────────────────────────────────
    # Dùng điểm cắt giữa (có thể lệch 1px nếu kích thước lẻ)
    mh, mw = H // 2, W // 2

    tiles = {
        "TL": img_np[:mh,  :mw ],
        "TR": img_np[:mh,  mw: ],
        "BL": img_np[mh:,  :mw ],
        "BR": img_np[mh:,  mw: ],
    }

    print(f"  Ảnh lớn ({H}x{W}) → chia 4 tile: TL{tiles['TL'].shape}, TR{tiles['TR'].shape}, "
          f"BL{tiles['BL'].shape}, BR{tiles['BR'].shape}")

    results = {}
    for name, tile in tiles.items():
        print(f"    Inferring tile {name} ({tile.shape[0]}x{tile.shape[1]})...")
        results[name] = infer_tile(model, tile, pad)

    # ── Ghép lại ────────────────────────────────────────────────────────────
    top    = np.concatenate([results["TL"], results["TR"]], axis=1)
    bottom = np.concatenate([results["BL"], results["BR"]], axis=1)
    merged = np.concatenate([top, bottom], axis=0)

    # Đảm bảo kích thước đầu ra khớp với đầu vào (phòng trường hợp lệch 1px)
    merged = merged[:H, :W]

    return merged

def infer_image_custom(model, img_np, threshold=SIZE_THRESHOLD, pad=PAD):
    """
    img_np    : numpy float32 (H, W), đã normalize về [0,1]
    threshold : nếu H > threshold hoặc W > threshold thì chia 4 tile
    Trả về    : numpy float32 (H, W)
    """
    H, W = img_np.shape

    if H <= threshold and W <= threshold:
        # Ảnh nhỏ: inference trực tiếp
        return infer_tile(model, img_np, pad)

    if H <= 3000 and W <= 3000:
    # ── Chia 4 tile (TL, TR, BL, BR) ────────────────────────────────────────
    # Dùng điểm cắt giữa (có thể lệch 1px nếu kích thước lẻ)
        mh, mw = H // 2, W // 2
        
        tiles = {
            "TL": img_np[:mh,  :mw ],
            "TR": img_np[:mh,  mw: ],
            "BL": img_np[mh:,  :mw ],
            "BR": np.flip(img_np[mh:,  mw: ], axis=None),
        }

        print(f"  Ảnh lớn ({H}x{W}) → chia 4 tile: TL{tiles['TL'].shape}, TR{tiles['TR'].shape}, "
            f"BL{tiles['BL'].shape}, BR{tiles['BR'].shape}")

        results = {}
        for name, tile in tiles.items():
            print(f"    Inferring tile {name} ({tile.shape[0]}x{tile.shape[1]})...")
            results[name] = infer_tile(model, tile, pad)

        # ── Ghép lại ────────────────────────────────────────────────────────────
        top    = np.concatenate([results["TL"], results["TR"]], axis=1)
        bottom = np.concatenate([results["BL"], np.flip(results["BR"], axis=None)], axis=1)
        merged = np.concatenate([top, bottom], axis=0)

        # Đảm bảo kích thước đầu ra khớp với đầu vào (phòng trường hợp lệch 1px)
        merged = merged[:H, :W]

        return merged

    else:
        mh1, mw1, mh2, mw2 = H // 3, W // 3, 2*H // 3, 2*W // 3

        tiles = {
            "P1": img_np[:mh1,  :mw1 ],
            "P2": img_np[:mh1,  mw1:mw2 ],
            "P3": img_np[:mh1,  mw2: ],
            "P4": img_np[mh1:mh2,  :mw1 ],
            "P5": img_np[mh1:mh2,  mw1:mw2 ],
            "P6": img_np[mh1:mh2,  mw2: ],
            "P7": img_np[mh2:,  :mw1 ],
            "P8": img_np[mh2:,  mw1:mw2 ],
            "P9": img_np[mh2:,  mw2: ],
        }

        print(f"  Ảnh lớn ({H}x{W}) → chia 9 tile: P1{tiles['P1'].shape}, P2{tiles['P2'].shape}, P3{tiles['P3'].shape},"
              f"P4{tiles['P4'].shape}, P5{tiles['P5'].shape}, P6{tiles['P6'].shape},"
            f"P7{tiles['P7'].shape}, P8{tiles['P8'].shape}, P9{tiles['P9'].shape}")

        results = {}
        for name, tile in tiles.items():
            print(f"    Inferring tile {name} ({tile.shape[0]}x{tile.shape[1]})...")
            results[name] = infer_tile(model, tile, pad)

        # ── Ghép lại ────────────────────────────────────────────────────────────
        r1    = np.concatenate([results["P1"], results["P2"], results["P3"]], axis=1)
        r2    = np.concatenate([results["P4"], results["P5"], results["P6"]], axis=1)
        r3    = np.concatenate([results["P7"], results["P8"], results["P9"]], axis=1)
        merged = np.concatenate([r1, r2, r3], axis=0)

        # Đảm bảo kích thước đầu ra khớp với đầu vào (phòng trường hợp lệch 1px)
        merged = merged[:H, :W]

        return merged

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
model_path = "savecheckpoints/checkpoints20260515_60-80/checkpoint_epoch20.pth"
model = UNet(n_channels=1, n_classes=1)
model.load_state_dict(torch.load(model_path, map_location=device))
print(f"Model dtype: {next(model.parameters()).dtype}")
model.to(device)
model.eval()

_min = 46484
_max = 50937
nk   = 0
step = 1
tic = time.time()
for i in range(0, 4**nk, step):
    for j in range(i, i + step):
        input_image = (
            f"G:\\AutoImageProcessing\\WaferData\\FimilarOutput\\MitsElec_3inch_01_220_0.000_0.000_26Aug21_115129_-91.351_Survey_003\\Input\\Branch_{nk}_{j}.tif"
        )

        print(f"Processing Branch_{nk}_{j} ...")
        t0 = time.time()

        # Load + normalize bằng giá trị cố định _min/_max
        raw, info = load_raw_image(input_image)
        raw = normalize_wafer_fixed(raw, _min / 65535, _max / 65535)

        # Inference (tự động chia tile nếu ảnh lớn)
        enhanced = infer_image_custom(model, raw)

        # Normalize output rồi lưu
        output_image = f"Result\\output\\Branch_{nk}_{j}.tif"
        save_image(normalize_wafer(enhanced), output_image, info)

        print(f"  Saved: {output_image}  ({time.time()-t0:.2f}s)")

print("Total time:", time.time()-tic)