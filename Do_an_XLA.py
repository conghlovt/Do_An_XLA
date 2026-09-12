"""Đồ án XLA: giải mã QR/Data Matrix từ ảnh, xử lý theo từng mức độ khó."""

import argparse
import json
import logging
import math
import sys
import time
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import requests
import zxingcpp

# Thư viện tuỳ chọn (nếu có thì dùng thêm):
try:
    from pyzbar.pyzbar import ZBarSymbol, decode as decode_pyzbar
except Exception:
    decode_pyzbar = None

try:
    from pylibdmtx.pylibdmtx import decode as decode_dmtx
except Exception:
    decode_dmtx = None


# =========================
# Cấu hình
# =========================
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = BASE_DIR / "wechat_models"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
LOGGER = logging.getLogger(__name__)

ENGINE_PRIORITY = {
    "WeChat": 5,
    "ZXing": 4,
    "OpenCVQR": 3,
    "ZBar": 2,
    "DataMatrix": 2,
}

# QR hợp lệ có thể chỉ chứa một ký tự.
MIN_TEXT_LEN = 1

# Lọc text rác: nếu tỷ lệ ký tự điều khiển cao thì loại.
MAX_CONTROL_CHAR_RATIO = 0.12  # Tối đa 12% ký tự điều khiển

# Lọc hình học cho tứ giác (kích thước/tỷ lệ/độ lồi...).
MIN_AREA_FRAC = 0.0  # QR nhỏ trong ảnh lớn vẫn hợp lệ sau khi decode thành công
MAX_AREA_FRAC = 1.0  # Hỗ trợ QR chiếm gần toàn bộ ảnh/ROI
MIN_SIDE_PX = 2
MAX_SIDE_RATIO = 3.2  # max(cạnh)/min(cạnh) quá lớn ⇒ loại (hình quá dài)
BBOX_AR_MIN = 0.28  # Tỷ lệ bbox tối thiểu (w/h)
BBOX_AR_MAX = 3.6  # Tỷ lệ bbox tối đa (w/h)

# Loại trùng theo IoU (nhiều engine trả về cùng một mã).
DEDUP_IOU_THR = 0.55

# Nếu muốn “cứng” hơn (ít false positives hơn), có thể bật tuỳ chọn dưới đây:
# ONLY_STRONG_ENGINES = True  # Chỉ dùng WeChat/ZXing/OpenCVQR (giảm FP nhưng có thể bỏ sót).
ONLY_STRONG_ENGINES = False
DECODE_SCALES = (1.0, 1.6, 2.2)
DECODE_ROTATION_GROUPS = ((0, 90, 180, 270), (45, 135, 225, 315))


# =========================
# Tải model WeChatQR (tự động nếu thiếu)
# =========================
def download_wechat_models(model_dir=DEFAULT_MODEL_DIR):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    urls = {
        "detect.prototxt": "https://raw.githubusercontent.com/WeChatCV/opencv_3rdparty/wechat_qrcode/detect.prototxt",
        "detect.caffemodel": "https://raw.githubusercontent.com/WeChatCV/opencv_3rdparty/wechat_qrcode/detect.caffemodel",
        "sr.prototxt": "https://raw.githubusercontent.com/WeChatCV/opencv_3rdparty/wechat_qrcode/sr.prototxt",
        "sr.caffemodel": "https://raw.githubusercontent.com/WeChatCV/opencv_3rdparty/wechat_qrcode/sr.caffemodel",
    }

    print("⏳ Checking & downloading WeChat QR models...")
    for name, url in urls.items():
        path = model_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            temporary = path.with_suffix(path.suffix + ".part")
            try:
                with requests.get(url, timeout=(10, 30), stream=True) as response:
                    response.raise_for_status()
                    with temporary.open("wb") as output:
                        for chunk in response.iter_content(chunk_size=65536):
                            output.write(chunk)
                if temporary.stat().st_size == 0:
                    raise OSError("Empty model download")
                temporary.replace(path)
                print(f"✅ Downloaded: {name}")
            except (requests.RequestException, OSError) as e:
                print(f"⚠️ Failed to download {name}: {e}")
            finally:
                temporary.unlink(missing_ok=True)
    ready = wechat_models_ok(model_dir)
    print("✅ WeChat models ready." if ready else "⚠️ WeChat models incomplete; other engines remain available.")
    return ready


def wechat_models_ok(model_dir=DEFAULT_MODEL_DIR) -> bool:
    model_dir = Path(model_dir)
    need = ["detect.prototxt", "detect.caffemodel", "sr.prototxt", "sr.caffemodel"]
    return all((model_dir / n).is_file() and (model_dir / n).stat().st_size > 0 for n in need)


# =========================
# Hàm hỗ trợ hiển thị/debug
# =========================
def _to_bgr(img):
    if img is None:
        return None
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def _resize_max_side(img, max_side=900):
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img
    scale = max_side / m
    return cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)


def add_caption_below(img_bgr, caption, cap_h=34):
    h, w = img_bgr.shape[:2]
    strip = np.zeros((cap_h, w, 3), dtype=img_bgr.dtype)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    thickness = 2

    (tw, th), _ = cv2.getTextSize(caption, font, font_scale, thickness)
    while tw > w - 20 and font_scale > 0.35:
        font_scale -= 0.05
        (tw, th), _ = cv2.getTextSize(caption, font, font_scale, thickness)

    x = max(10, (w - tw) // 2)
    y = (cap_h + th) // 2
    cv2.putText(strip, caption, (x, y), font, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)
    return np.vstack([img_bgr, strip])


def show_window(win_name, img, wait_ms=1, max_side=1000):
    if img is None:
        return
    vis = _to_bgr(img)
    vis = _resize_max_side(vis, max_side=max_side)
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.imshow(win_name, vis)
    cv2.waitKey(wait_ms)


def show_montage(win_name, images, labels, ncols=3, cap_h=34, tile_max_side=380, wait_ms=1):
    tiles = []
    for img, lab in zip(images, labels):
        if img is None:
            continue
        t = _to_bgr(img)
        t = _resize_max_side(t, max_side=tile_max_side)
        t = add_caption_below(t, lab, cap_h=cap_h)
        tiles.append(t)
    if not tiles:
        return

    rows = []
    for i in range(0, len(tiles), ncols):
        row = tiles[i:i + ncols]
        max_h = max(im.shape[0] for im in row)
        padded = []
        for im in row:
            if im.shape[0] < max_h:
                pad = np.zeros((max_h - im.shape[0], im.shape[1], 3), dtype=im.dtype)
                im = np.vstack([im, pad])
            padded.append(im)
        rows.append(np.hstack(padded))

    max_w = max(r.shape[1] for r in rows)
    padded_rows = []
    for r in rows:
        if r.shape[1] < max_w:
            pad = np.zeros((r.shape[0], max_w - r.shape[1], 3), dtype=r.dtype)
            r = np.hstack([r, pad])
        padded_rows.append(r)

    final = np.vstack(padded_rows)
    show_window(win_name, final, wait_ms=wait_ms, max_side=1400)


# =========================
# Hàm hỗ trợ xử lý ảnh & hình học (tăng tỷ lệ decode)
# =========================
def order_points(pts: np.ndarray) -> np.ndarray:
    try:
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    except (ValueError, TypeError):
        return None
    if pts.shape[0] < 4 or not np.isfinite(pts).all():
        return None
    hull = cv2.convexHull(pts).reshape(-1, 2)
    if len(hull) < 4:
        return None
    if pts.shape[0] > 4:
        hull = cv2.boxPoints(cv2.minAreaRect(pts))
    # Sắp góc quanh tâm: không lặp đỉnh khi QR có góc 45 độ.
    delta = hull - hull.mean(axis=0)
    ordered = hull[np.argsort(np.arctan2(delta[:, 1], delta[:, 0]))]
    start = np.lexsort((ordered[:, 1], ordered.sum(axis=1)))[0]
    return np.roll(ordered, -int(start), axis=0).astype(np.float32)


def bbox_from_pts(pts):
    xs = pts[:, 0]
    ys = pts[:, 1]
    x1, y1 = float(xs.min()), float(ys.min())
    x2, y2 = float(xs.max()), float(ys.max())
    return (x1, y1, x2, y2)


def bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = area_a + area_b - inter + 1e-9
    return inter / union


def normalize_quad(pts, w, h):
    if pts is None or w <= 0 or h <= 0:
        return None
    pts = order_points(pts)
    if pts is None:
        return None

    # Clip điểm về trong biên ảnh để vẽ polygon không bị “văng” ra ngoài.
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    return pts


def quad_ok(pts, w, h):
    pts = pts.astype(np.float32)
    area = float(cv2.contourArea(pts))
    img_area = float(w * h)

    if area < max(4.0, MIN_AREA_FRAC * img_area):
        return False
    if area > MAX_AREA_FRAC * img_area:
        return False

    (tl, tr, br, bl) = pts
    sides = [
        np.linalg.norm(tr - tl),
        np.linalg.norm(br - tr),
        np.linalg.norm(bl - br),
        np.linalg.norm(tl - bl),
    ]
    if min(sides) < MIN_SIDE_PX:
        return False
    if (max(sides) / (min(sides) + 1e-9)) > MAX_SIDE_RATIO:
        return False

    x1, y1, x2, y2 = bbox_from_pts(pts)
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    ar = bw / bh
    if ar < BBOX_AR_MIN or ar > BBOX_AR_MAX:
        return False

    # Kiểm tra tứ giác có lồi (convex) hay không.
    if not cv2.isContourConvex(pts):
        return False

    return True


def text_ok(s: str) -> bool:
    if s is None:
        return False
    s = s.strip()
    if len(s) < MIN_TEXT_LEN:
        return False

    # Tính tỷ lệ ký tự điều khiển trong text (để loại rác).
    ctrl = 0
    for ch in s:
        o = ord(ch)
        if o < 32 and ch not in ("\n", "\r", "\t"):
            ctrl += 1
    if len(s) > 0 and (ctrl / len(s)) > MAX_CONTROL_CHAR_RATIO:
        return False

    return True


def filter_and_dedup_results(results, w, h, iou_thr=DEDUP_IOU_THR):
    cleaned = []
    for r in results:
        txt = r.get("text") or ""
        if not text_ok(txt):
            continue

        pts = normalize_quad(r.get("points"), w, h)
        if pts is None:
            continue
        if not quad_ok(pts, w, h):
            continue

        bb = bbox_from_pts(pts)
        rr = dict(r)
        rr["text"] = txt
        rr["points"] = pts
        rr["bbox"] = bb
        cleaned.append(rr)

    # Sắp xếp theo: độ ưu tiên engine → độ dài text → diện tích vùng phát hiện.
    def score(rr):
        pri = ENGINE_PRIORITY.get(rr.get("engine", ""), 0)
        ln = len(rr.get("text", ""))
        area = cv2.contourArea(rr["points"].astype(np.float32))
        return pri, ln, area

    cleaned.sort(key=score, reverse=True)

    kept = []
    for r in cleaned:
        dup = False
        for k in kept:
            if bbox_iou(r["bbox"], k["bbox"]) >= iou_thr:
                dup = True
                break
        if not dup:
            kept.append(r)

    return kept


def warp_by_points(img, points):
    result = warp_by_points_with_matrix(img, points)
    return result[0] if result is not None else None


def warp_by_points_with_matrix(img, points):
    """Warp a detected quadrilateral and return (image, source-to-ROI matrix)."""
    if points is None:
        return None

    h, w = img.shape[:2]
    pts = normalize_quad(points, w, h)
    if pts is None or len(pts) != 4 or not quad_ok(pts, w, h):
        return None

    tl, tr, br, bl = pts
    max_width = max(2, int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))))
    max_height = max(2, int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))))
    dst = np.array(
        [[0, 0], [max_width - 1, 0],
         [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype=np.float32,
    )
    try:
        M = cv2.getPerspectiveTransform(pts, dst)
        warped = cv2.warpPerspective(img, M, (max_width, max_height))
        return warped, M
    except Exception:
        return None


@lru_cache(maxsize=8)
def gamma_table(gamma):
    inv = 1.0 / max(gamma, 1e-6)
    table = (np.arange(256) / 255.0) ** inv * 255.0
    return np.clip(table, 0, 255).astype(np.uint8)


def gamma_correct(gray, gamma=1.2):
    return cv2.LUT(gray, gamma_table(gamma))


def unsharp_mask(gray, amount=1.2, radius=2):
    blur = cv2.GaussianBlur(gray, (0, 0), radius)
    sharp = cv2.addWeighted(gray, 1.0 + amount, blur, -amount, 0)
    return sharp


def glare_inpaint(gray, thr=245):
    mask = (gray >= thr).astype(np.uint8) * 255
    # Không inpaint cả nền trắng và các module trắng của QR.
    if mask.mean() < 1 or np.count_nonzero(mask) / mask.size > 0.15:
        return gray
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)
    return cv2.inpaint(gray, mask, 3, cv2.INPAINT_TELEA)


# =========================
# Thu thập danh sách ảnh đầu vào (quét đệ quy)
# =========================
def collect_images(path: str):
    source = Path(path).expanduser()
    if source.is_file():
        return [str(source.resolve())] if source.suffix.lower() in IMAGE_EXTENSIONS else []
    if not source.is_dir():
        return []

    return sorted(
        str(file.resolve())
        for file in source.rglob("*")
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
    )


def read_image(path):
    """Đọc được đường dẫn tiếng Việt trên Windows."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    except (OSError, cv2.error):
        return None


def rotate_bound(gray, angle):
    """Xoay trên nền trắng và mở rộng khung để không cắt mất QR ở mép."""
    if angle == 0:
        return gray, np.eye(3)
    h, w = gray.shape[:2]
    matrix = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), angle, 1.0)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    width = int(math.ceil(w * cos + h * sin - 1e-8))
    height = int(math.ceil(h * cos + w * sin - 1e-8))
    matrix[0, 2] += (width - w) / 2
    matrix[1, 2] += (height - h) / 2
    return (
        cv2.warpAffine(gray, matrix, (width, height), borderValue=255),
        np.vstack([matrix, [0, 0, 1]]),
    )


def map_results(results, source_to_work, width, height):
    """Đưa các góc ở ảnh biến đổi/ROI về ảnh đầu vào."""
    try:
        inverse = np.linalg.inv(source_to_work)
    except np.linalg.LinAlgError:
        return []
    mapped = []
    for result in results:
        points = np.asarray(result["points"], np.float32).reshape(-1, 1, 2)
        mapped.append({**result, "points": cv2.perspectiveTransform(points, inverse).reshape(-1, 2)})
    return filter_and_dedup_results(mapped, width, height)


# =========================
# PipelineMaster: điều phối toàn bộ pipeline
# =========================
class PipelineMaster:
    def __init__(
        self,
        show_steps=False,
        pause_each_image=False,
        model_dir=DEFAULT_MODEL_DIR,
        display=False,
        strong_only=ONLY_STRONG_ENGINES,
        time_budget=8.0,
        max_side=1600,
    ):
        if not math.isfinite(time_budget) or time_budget < 0:
            raise ValueError("time_budget must be finite and >= 0")
        if max_side < 64:
            raise ValueError("max_side must be >= 64")
        self.model_dir = Path(model_dir)
        self.show_steps = show_steps
        self.pause_each_image = pause_each_image
        self.display = display
        self.strong_only = strong_only
        self.time_budget = time_budget
        self.max_side = max_side
        self._engine_errors = set()
        self.clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))

        self.cv_qr = cv2.QRCodeDetector()

        self.detector = None
        if not wechat_models_ok(self.model_dir):
            LOGGER.warning("WeChat models missing; using OpenCV/ZXing. Use --download_models to enable WeChat.")
            return
        try:
            from cv2 import wechat_qrcode
            self.detector = wechat_qrcode.WeChatQRCode(
                str(self.model_dir / "detect.prototxt"),
                str(self.model_dir / "detect.caffemodel"),
                str(self.model_dir / "sr.prototxt"),
                str(self.model_dir / "sr.caffemodel"),
            )
        except Exception as e:
            LOGGER.warning("WeChatQRCode init failed: %s", e)
            self.detector = None

    def pipeline_preprocess(self, img_bgr):
        stages = {}

        if img_bgr.ndim == 3:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_bgr.copy()
        stages["gray"] = gray

        # Khử vùng chói (glare) bằng inpaint.
        deglare = glare_inpaint(gray, thr=245)
        stages["deglare"] = deglare

        # Tăng tương phản cục bộ bằng CLAHE.
        enhanced = self.clahe.apply(deglare)
        stages["enhanced"] = enhanced

        # Làm nét nhẹ (unsharp mask).
        sharp = unsharp_mask(enhanced, amount=1.0, radius=2)
        stages["sharp"] = sharp

        # Thử gamma correction ở 2 mức để bắt QR tối/tối hơn.
        stages["gamma_0.8"] = gamma_correct(sharp, gamma=0.8)
        stages["gamma_1.3"] = gamma_correct(sharp, gamma=1.3)

        # Giảm nhiễu nhẹ trước khi nhị phân hoá.
        gaussian = cv2.GaussianBlur(sharp, (5, 5), 0)
        denoised = cv2.medianBlur(gaussian, 3)
        stages["denoised"] = denoised

        # Nhị phân hoá (Adaptive/Otsu) + thêm bản đảo màu (invert).
        binary_adapt = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 21, 5
        )
        stages["binary_adapt"] = binary_adapt
        stages["binary_adapt_inv"] = cv2.bitwise_not(binary_adapt)

        _, binary_otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        stages["binary_otsu"] = binary_otsu
        stages["binary_otsu_inv"] = cv2.bitwise_not(binary_otsu)

        # Giảm ảnh hưởng vùng bóng (ước lượng nền rồi chuẩn hoá).
        dilated = cv2.dilate(gray, np.ones((7, 7), np.uint8))
        bg_blur = cv2.medianBlur(dilated, 21)
        diff = 255 - cv2.absdiff(gray, bg_blur)
        norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
        stages["de_shadow"] = norm.astype(np.uint8)

        return stages

    def _engine_error(self, engine, error):
        if engine not in self._engine_errors:
            LOGGER.warning("%s failed: %s", engine, error)
            self._engine_errors.add(engine)

    def _decode_all_engines(self, img_gray, deadline=math.inf):
        decoded, points_only = [], []

        def add(text, points, engine):
            if text:
                decoded.append({"text": text, "points": np.asarray(points, np.float32), "engine": engine})
            elif points is not None:
                points_only.append(np.asarray(points, np.float32).reshape(-1, 2))

        # ZXing tự xử lý chiều xoay và đảo màu; thử trước các engine nặng hơn.
        if time.perf_counter() < deadline:
            try:
                for result in zxingcpp.read_barcodes(
                    img_gray, formats=[zxingcpp.BarcodeFormat.QRCode, zxingcpp.BarcodeFormat.DataMatrix]
                ):
                    pos = result.position
                    points = [[p.x, p.y] for p in
                              (pos.top_left, pos.top_right, pos.bottom_right, pos.bottom_left)]
                    add(result.text, points, "ZXing")
            except Exception as error:
                self._engine_error("ZXing", error)

        if self.detector is not None and time.perf_counter() < deadline:
            try:
                texts, points = self.detector.detectAndDecode(img_gray)
                if points is not None:
                    for text, quad in zip(texts, points):
                        add(text, quad, "WeChat")
            except Exception as error:
                self._engine_error("WeChat", error)

        if time.perf_counter() < deadline:
            try:
                _, texts, points, _ = self.cv_qr.detectAndDecodeMulti(img_gray)
                if points is not None:
                    for text, quad in zip(texts, points):
                        add(text, quad, "OpenCVQR")
                # Single-code detector is useful when multi misses a lone code.
                if not decoded and time.perf_counter() < deadline:
                    text, points, _ = self.cv_qr.detectAndDecode(img_gray)
                    add(text, points, "OpenCVQR")
            except Exception as error:
                self._engine_error("OpenCVQR", error)

        if not self.strong_only and decode_pyzbar is not None and time.perf_counter() < deadline:
            try:
                for result in decode_pyzbar(img_gray, symbols=[ZBarSymbol.QRCODE]):
                    points = [[p.x, p.y] for p in result.polygon]
                    if len(points) < 4:
                        rect = result.rect
                        points = [[rect.left, rect.top], [rect.left + rect.width, rect.top],
                                  [rect.left + rect.width, rect.top + rect.height],
                                  [rect.left, rect.top + rect.height]]
                    add(result.data.decode("utf-8", errors="replace"), points, "ZBar")
            except Exception as error:
                self._engine_error("ZBar", error)

        if not self.strong_only and decode_dmtx is not None and time.perf_counter() < deadline:
            try:
                remaining_ms = 40 if not math.isfinite(deadline) else max(
                    1, min(40, int((deadline - time.perf_counter()) * 1000))
                )
                for result in decode_dmtx(img_gray, timeout=remaining_ms):
                    rect = result.rect
                    # libdmtx dùng gốc tọa độ dưới-trái (dmtximage.c).
                    top = img_gray.shape[0] - 1 - (rect.top + rect.height)
                    bottom = img_gray.shape[0] - 1 - rect.top
                    points = [[rect.left, top], [rect.left + rect.width, top],
                              [rect.left + rect.width, bottom], [rect.left, bottom]]
                    add(result.data.decode("utf-8", errors="replace"), points, "DataMatrix")
            except Exception as error:
                self._engine_error("DataMatrix", error)

        return decoded, points_only

    def _decode_roi(self, work, points, deadline):
        """Thêm quiet zone quanh ROI, rồi đưa tọa độ decode trở lại ảnh work."""
        warped = warp_by_points_with_matrix(work, points)
        if warped is None or time.perf_counter() >= deadline:
            return []
        roi, source_to_roi = warped
        if min(roi.shape[:2]) < 8:
            return []
        factor = min(2.2, 1200 / max(roi.shape[:2]))
        enlarged = cv2.resize(roi, None, fx=factor, fy=factor, interpolation=cv2.INTER_NEAREST)
        sx, sy = enlarged.shape[1] / roi.shape[1], enlarged.shape[0] / roi.shape[0]
        padding = max(8, round(min(enlarged.shape[:2]) * 0.1))
        padded = cv2.copyMakeBorder(enlarged, padding, padding, padding, padding,
                                    cv2.BORDER_CONSTANT, value=255)
        resize_and_border = np.array(
            [[sx, 0, padding + (sx - 1) / 2],
             [0, sy, padding + (sy - 1) / 2], [0, 0, 1]], dtype=np.float64
        )
        raw, _ = self._decode_all_engines(padded, deadline)
        valid = filter_and_dedup_results(raw, padded.shape[1], padded.shape[0])
        return map_results(valid, resize_and_border @ source_to_roi,
                           work.shape[1], work.shape[0])

    def decode_with_adjustment_loop(self, img_bgr, fname=""):
        """Trả về (kết quả trong tọa độ ảnh gốc, ảnh gốc, bước xử lý).

        time_budget là giới hạn mềm: kiểm tra giữa các lần gọi thư viện C++.
        Một lần gọi engine đang chạy có thể vượt quá thời gian này.
        """
        if img_bgr is None or img_bgr.size == 0 or img_bgr.dtype != np.uint8:
            raise ValueError("Expected a non-empty uint8 image")
        if img_bgr.ndim not in (2, 3):
            raise ValueError("Expected a grayscale or color image")
        started = time.perf_counter()
        deadline = started + self.time_budget if self.time_budget else math.inf
        height, width = img_bgr.shape[:2]
        if min(height, width) < 8:
            return [], img_bgr, "image_too_small"
        base = _resize_max_side(img_bgr, self.max_side)
        gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY) if base.ndim == 3 else base
        sx, sy = base.shape[1] / width, base.shape[0] / height
        source_to_base = np.array(
            [[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]], np.float64
        )
        roi_seen = set()

        def attempt(work, transform):
            raw, points_only = self._decode_all_engines(work, deadline)
            results = filter_and_dedup_results(raw, work.shape[1], work.shape[0])
            # Thử ROI ngay khi có vùng nghi ngờ, không đợi hết 192 lần thử.
            for points in points_only:
                if time.perf_counter() >= deadline or len(roi_seen) >= 8:
                    break
                pts = normalize_quad(points, work.shape[1], work.shape[0])
                if pts is None or not quad_ok(pts, work.shape[1], work.shape[0]):
                    continue
                original = cv2.perspectiveTransform(pts.reshape(-1, 1, 2),
                                                     np.linalg.inv(transform))
                key = tuple(np.round(original.reshape(-1) / 8).astype(int))
                if key in roi_seen:
                    continue
                roi_seen.add(key)
                results.extend(self._decode_roi(work, pts, deadline))
            return map_results(results, transform, width, height)

        # Giữ nguyên các module QR trước khi thử CLAHE/inpaint/sharpen.
        result = attempt(gray, source_to_base)
        if result:
            return result, img_bgr, f"Raw | kept={len(result)}"
        if time.perf_counter() >= deadline:
            return [], img_bgr, "time_budget_exceeded"

        stages = self.pipeline_preprocess(base)
        if self.display and self.show_steps:
            names = ("gray", "deglare", "enhanced", "sharp", "binary_adapt", "binary_otsu")
            show_montage("PREPROCESS", [stages[name] for name in names],
                         list(names), ncols=3)

        candidates = [
            ("CLAHE", stages["enhanced"]),
            ("Sharp", stages["sharp"]),
            ("DeShadow", stages["de_shadow"]),
            ("BinaryAdapt", stages["binary_adapt"]),
            ("BinaryOtsu", stages["binary_otsu"]),
            ("Gamma0.8", stages["gamma_0.8"]),
            ("Gamma1.3", stages["gamma_1.3"]),
            ("BinaryAdaptInv", stages["binary_adapt_inv"]),
            ("BinaryOtsuInv", stages["binary_otsu_inv"]),
        ]
        for rotation_group in DECODE_ROTATION_GROUPS:
            for scale in DECODE_SCALES:
                # Cache chỉ một mức phóng, tạo khi cần để giảm RAM và startup cost.
                scaled = {}
                for angle in rotation_group:
                    for name, candidate in candidates:
                        if time.perf_counter() >= deadline:
                            return [], img_bgr, "time_budget_exceeded"
                        if name not in scaled:
                            scaled[name] = candidate if scale == 1.0 else cv2.resize(
                                candidate, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
                            )
                        image = scaled[name]
                        scale_x, scale_y = image.shape[1] / gray.shape[1], image.shape[0] / gray.shape[0]
                        scaling = np.array(
                            [[scale_x, 0, (scale_x - 1) / 2],
                             [0, scale_y, (scale_y - 1) / 2], [0, 0, 1]], np.float64
                        )
                        work, rotation = rotate_bound(image, angle)
                        result = attempt(work, rotation @ scaling @ source_to_base)
                        if result:
                            return result, img_bgr, f"{name} | rot={angle} | s={scale} | kept={len(result)}"
        return [], img_bgr, "not_found"

    def run(self, input_path, output_path=None, limit=None):
        files = collect_images(input_path)
        if not files:
            raise ValueError(f"No supported images found: {input_path}")
        if limit is not None:
            if limit < 1:
                raise ValueError("limit must be >= 1")
            files = files[:limit]
        if output_path is not None:
            target = Path(output_path).resolve()
            if target.exists():
                raise ValueError(f"Output already exists; choose a new file: {target}")
            if target.suffix.lower() != ".json":
                raise ValueError("Output must have the .json extension")
        print(f"START: {len(files)} images | input={Path(input_path).resolve()}")
        records, success, errors = [], 0, 0
        started = time.perf_counter()
        try:
            for filename in files:
                begin = time.perf_counter()
                img = read_image(filename)
                decoded = []
                if img is None:
                    info = "read_error"
                    errors += 1
                else:
                    try:
                        decoded, _, info = self.decode_with_adjustment_loop(img, Path(filename).name)
                    except (cv2.error, ValueError) as error:
                        LOGGER.warning("Cannot process %s: %s", filename, error)
                        info = "processing_error"
                        errors += 1
                if decoded:
                    success += 1
                record = {
                    "image": filename,
                    "status": "ok" if decoded else info,
                    "method": info,
                    "elapsed_seconds": round(time.perf_counter() - begin, 6),
                    "codes": [
                        {"text": d["text"], "engine": d["engine"], "points": d["points"].tolist()}
                        for d in decoded
                    ],
                }
                records.append(record)
                print(f"[{len(records)}/{len(files)}] {Path(filename).name}: {info}", flush=True)
                for index, code in enumerate(decoded, 1):
                    print(f"  [{index}] {code['engine']}: {code['text'][:150]!r}")
                if self.display and img is not None:
                    vis = img.copy()
                    for index, code in enumerate(decoded, 1):
                        pts = code["points"].astype(np.int32).reshape(-1, 1, 2)
                        cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
                        x, y = pts[0, 0]
                        cv2.putText(vis, str(index), (int(x), max(15, int(y) - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    show_window("INPUT", img)
                    show_window("RESULT", vis)
                    if self.pause_each_image and cv2.waitKey(0) & 0xFF == 27:
                        break
            if self.display and not self.pause_each_image:
                print("Press a key in an OpenCV window to exit.")
                cv2.waitKey(0)
        finally:
            if self.display:
                cv2.destroyAllWindows()

        summary = {
            "total": len(records), "success": success, "errors": errors,
            "elapsed_seconds": time.perf_counter() - started, "images": records,
        }
        print(f"RESULT: {success}/{len(records)} images decoded | errors={errors}"
              f" | time={summary['elapsed_seconds']:.2f}s")
        if output_path is not None:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("x", encoding="utf-8") as output:
                json.dump(summary, output, ensure_ascii=False, indent=2)
        return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Decode QR/Data Matrix codes from images.")
    parser.add_argument("--input", required=True, help="Folder path OR single image path")
    parser.add_argument("--show_steps", "--show-steps", action="store_true", help="Show preprocess montage")
    parser.add_argument("--pause", action="store_true", help="Pause each displayed image; Escape stops the batch")
    display = parser.add_mutually_exclusive_group()
    display.add_argument("--display", action="store_true", help="Show OpenCV windows")
    display.add_argument("--no-display", action="store_true", help="Run without windows (default)")
    parser.add_argument("--strong-only", action="store_true", help="Disable optional ZBar/libdmtx engines")
    parser.add_argument("--download_models", "--download-models", action="store_true",
                        help="Download missing/empty WeChat models; normal runs work offline")
    parser.add_argument("--model_dir", "--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--time-budget", type=float, default=8.0,
                        help="Soft seconds per image, checked between engine calls; 0 = all attempts")
    parser.add_argument("--max-side", type=int, default=1600, help="Maximum input working dimension")
    parser.add_argument("--limit", type=int, help="Process only the first N images")
    parser.add_argument("--output", help="Save a JSON report to a new file")
    parser.add_argument("--verbose", action="store_true", help="Enable diagnostic logging")
    args = parser.parse_args(argv)
    if not math.isfinite(args.time_budget) or args.time_budget < 0:
        parser.error("--time-budget must be finite and >= 0")
    if args.max_side < 64:
        parser.error("--max-side must be >= 64")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")
    if not collect_images(args.input):
        parser.error("Input is missing, empty, or contains no supported image files")
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING)
    if args.download_models:
        download_wechat_models(args.model_dir)
    pipeline = PipelineMaster(
        show_steps=args.show_steps,
        pause_each_image=args.pause,
        model_dir=args.model_dir,
        display=not args.no_display and (args.display or args.show_steps or args.pause),
        strong_only=args.strong_only,
        time_budget=args.time_budget,
        max_side=args.max_side,
    )
    try:
        summary = pipeline.run(args.input, args.output, args.limit)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
