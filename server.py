import base64
import io
import os
import time
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

# Guard cv2 import — give clear error if not installed
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("\n[WARNING] opencv-python-headless not installed.")
    print("[WARNING] Run: pip install opencv-python-headless")
    print("[WARNING] Validation layer DISABLED — AI-only mode\n")

app = Flask(__name__)
CORS(app)

MODEL_CANDIDATES = [
    "Model1_EfficientNetB0_best.keras",
    "EfficientNetB0_3Class_Robust.keras",
]

def _resolve_model_path():
    base_dir = os.path.dirname(__file__)
    for c in MODEL_CANDIDATES:
        p = os.path.join(base_dir, c)
        if os.path.exists(p):
            return p
    return os.path.join(base_dir, MODEL_CANDIDATES[-1])

MODEL_PATH   = _resolve_model_path()
CLASS_NAMES  = ["healthy", "rotten", "unknown"]

# Rainbow palette (low -> high): indigo, blue, cyan, green, yellow, orange, red, magenta
RAINBOW_PALETTE = [
    (63, 0, 125),
    (0, 58, 168),
    (0, 196, 255),
    (0, 230, 118),
    (255, 214, 0),
    (255, 111, 0),
    (244, 67, 54),
    (170, 0, 110),
]
MIN_CONFIDENCE = 0.55
MIN_MARGIN     = 0.12

model                = None
tf_module            = None
input_size           = (224, 224)
last_conv_layer_name = None

# Runtime config exposed to the frontend
RUNTIME_CONFIG = {
    "preprocess_mode": "auto",
    "force_classification": False,
    "min_confidence": MIN_CONFIDENCE,
    "min_margin": MIN_MARGIN,
    "last_conv_layer_override": None,
    "mock_inference": False,
}

# ══════════════════════════════════════════════════════════════════════════════
# VALIDATION THRESHOLDS
#
# Key fixes vs hibiscus / pineapple:
#   HSV pink:  sat >= 115 (was 90)  — hibiscus is less saturated
#   HSV pink:  val >= 60  (was 50)  — darker flowers excluded
#   crimson:   sat >= 165 (was 155) — pineapple bracts excluded
#   max_edge:  0.32 (was 0.38)      — hibiscus petals have more edges
#   min_solid: 0.42 (was 0.38)      — flowers scatter more
#   min_fill:  0.30 (was 0.25)      — petal gaps show up more
# ══════════════════════════════════════════════════════════════════════════════
VAL = dict(
    px_min     = 120,   # minimum qualifying pixels
    min_solid  = 0.42,  # blob solidity (hibiscus petals: ~0.15-0.30)
    min_fill   = 0.30,  # interior fill  (flowers have gaps)
    max_edge   = 0.32,  # edge density on blob — KEY discriminator
                        #   dragon fruit skin:   0.08-0.25 (smooth, waxy)
                        #   hibiscus petals:     0.30-0.55 (petal veins + edges)
                        #   rose petals:         0.38-0.65
                        #   rambutan spikes:     0.50-0.80
    max_iqr    = 18.0,  # hue IQR — DF color uniform; flowers vary
    min_cf     = 0.18,  # center fill — stamen/center of flower fails
    rot_conf   = 0.68,  # rotten bypass threshold
    rot_px     = 400,   # rotten color pixels needed for bypass
)

# ══════════════════════════════════════════════════════════════════════════════
# HSV MASKS — stricter than before
# ══════════════════════════════════════════════════════════════════════════════
def _df_mask(hsv):
    """
    Dragon fruit HSV fingerprint.
    
    Pink variety:   hue 145-168, sat >= 115, val >= 60
      - Raised sat from 90→115: hibiscus is less saturated (sat ~70-100)
      - Raised val from 50→60:  excludes darker/shadowed flowers
    
    Crimson variety: hue 0-4 or 172-180, sat >= 165, val >= 55
      - Raised sat from 155→165: pineapple bracts have lower sat (~120-140)
      - Keeps hue range 0-4 ONLY (was 0-4, stays same)
    
    What gets excluded:
      Hibiscus:   hue 145-165, sat 70-105 → sat gate blocks it
      Pineapple bracts: hue 0-10, sat 120-140 → sat gate blocks it
      Roses:      hue 5-15 → hue range blocks it
      Rambutan:   hue 6-20 → hue range blocks it
    """
    h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    pink    = (h>=145)&(h<=168)&(s>=115)&(v>=60)
    crimson = ((h<=4)|(h>=172))&(s>=165)&(v>=55)
    return (pink|crimson).astype(np.uint8)

def _rotten_mask(hsv):
    """Rotten dragon fruit color signature."""
    h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    dark       = (v<80)&(v>8)
    faded_pink = (h>=130)&(h<=178)&(s>=30)&(s<90)&(v>=40)
    brown      = (h>=8)&(h<=35)&(s>=50)&(v>=35)&(v<185)
    return (dark|faded_pink|brown).astype(np.uint8)

def _rotten_confirmed(pil_224):
    if not CV2_AVAILABLE: return False
    bgr = cv2.cvtColor(np.array(pil_224), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    rm  = _rotten_mask(hsv)
    rm  = cv2.morphologyEx((rm*255).astype(np.uint8), cv2.MORPH_CLOSE,
          cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))
    return int(np.sum(rm>0)) >= VAL["rot_px"]

# ══════════════════════════════════════════════════════════════════════════════
# 7-LAYER VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
def validate_dragon_fruit(pil_224):
    """
    Returns dict: ok, px_count, edge_density, fail_reason
    
    Layer 1: HSV pixel count
    Layer 2: Shape solidity (scattered petals fail)
    Layer 3: Interior fill  (flower gaps fail)
    Layer 4: Edge density ON BLOB ONLY (hibiscus/rambutan/rose fail)
    Layer 5: Hue IQR (inconsistent color = not DF)
    Layer 6: Center fill (flower stamen = yellow center fails)
    Layer 7: Compactness (star-shaped flowers fail)
    """
    if not CV2_AVAILABLE:
        return dict(ok=True, px_count=999, edge_density=0.0,
                    fail_reason="", note="cv2 not installed — validation skipped")

    bgr  = cv2.cvtColor(np.array(pil_224), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    raw  = _df_mask(hsv)
    total= int(np.sum(raw))
    r    = dict(ok=False, px_count=total, edge_density=0.0, fail_reason="")

    # Layer 1: Pixel count
    if total < VAL["px_min"]:
        r["fail_reason"] = (
            f"No dragon fruit color — {total} px (need {VAL['px_min']}+). "
            "Expected sat>=115 magenta-pink hue 145-168, or crimson sat>=165 hue 0-4/172-180.")
        return r

    # Morphological clean
    k11 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
    k5  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    m   = (raw*255).astype(np.uint8)
    m   = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k11)
    m   = cv2.morphologyEx(m, cv2.MORPH_OPEN,  k5)

    n, lbl, stats, _ = cv2.connectedComponentsWithStats(m)
    if n<=1:
        r["fail_reason"] = "No solid blob after morphological filtering."
        return r

    blobs = sorted([(i, stats[i,cv2.CC_STAT_AREA]) for i in range(1,n)],
                   key=lambda x:x[1], reverse=True)

    for bidx, barea in blobs:
        if barea < 60: break
        bm   = np.where(lbl==bidx, 255, 0).astype(np.uint8)
        px_b = int(np.sum(raw & (bm>0)))
        if px_b < 60: continue

        # Layer 2: Solidity
        cnts,_ = cv2.findContours(bm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: continue
        cnt  = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        if area < 60: continue
        hull = cv2.convexHull(cnt)
        ha   = cv2.contourArea(hull)
        sol  = float(area/ha) if ha>0 else 0.
        if sol < VAL["min_solid"]:
            r["fail_reason"] = (
                f"Shape too scattered (solidity {sol:.2f} < {VAL['min_solid']}) "
                "— flower petals spread outward.")
            continue

        # Layer 3: Interior fill
        hc = np.zeros((224,224), np.uint8)
        cv2.drawContours(hc, [hull], 0, 255, -1)
        inf = float(np.sum((hc>0)&raw) / max(np.sum(hc>0),1))
        if inf < VAL["min_fill"]:
            r["fail_reason"] = (
                f"Interior too sparse (fill {inf:.2f} < {VAL['min_fill']}) "
                "— gaps between petals or background visible.")
            continue

        # Layer 4: Edge density ON BLOB ONLY — KEY check
        edges = cv2.Canny(gray, 40, 110)
        ed    = int(np.sum(edges[bm>0]>0)) / max(px_b, 1)
        r["edge_density"] = round(ed, 3)
        if ed > VAL["max_edge"]:
            r["fail_reason"] = (
                f"Surface too rough (edge density {ed:.2f} > {VAL['max_edge']}) "
                f"— hibiscus/rambutan/flower texture detected. "
                f"Dragon fruit skin is smooth (0.08-0.25).")
            continue

        # Layer 5: Hue IQR
        h_c  = hsv[:,:,0]
        hues = h_c[bm>0]
        if len(hues)>0:
            ph  = hues[(hues>=130)&(hues<=180)]
            rh  = hues[hues<=10]
            ip  = float(np.percentile(ph,75)-np.percentile(ph,25)) if len(ph)>10 else 999.
            ir  = float(np.percentile(rh,75)-np.percentile(rh,25)) if len(rh)>10 else 999.
            iqr = min(ip,ir)
        else: iqr=999.
        if iqr > VAL["max_iqr"]:
            r["fail_reason"] = (
                f"Color inconsistent (IQR {iqr:.1f} > {VAL['max_iqr']}) "
                "— mixed colors indicate non-dragon-fruit.")
            continue

        # Layer 6: Center fill (stamen check)
        M = cv2.moments(bm)
        if M["m00"]>0:
            cx_b = int(M["m10"]/M["m00"])
            cy_b = int(M["m01"]/M["m00"])
            br   = max(int(np.sqrt(area/np.pi)*0.28), 5)
            cc   = np.zeros((224,224), np.uint8)
            cv2.circle(cc,(cx_b,cy_b),br,255,-1)
            cf = float(np.sum((cc>0)&raw)/max(np.sum(cc>0),1))
            if cf < VAL["min_cf"]:
                r["fail_reason"] = (
                    f"Non-pink center (cf {cf:.2f} < {VAL['min_cf']}) "
                    "— flower stamen or yellow/white center detected.")
                continue

        # Layer 7: Compactness (star-shaped = low compactness)
        perim = cv2.arcLength(cnt, True)
        comp  = float(4*np.pi*area/perim**2) if perim>0 else 0.
        if comp < 0.06:
            r["fail_reason"] = (
                f"Shape too irregular (compactness {comp:.3f} < 0.06) "
                "— star or spiky shape, not dragon fruit.")
            continue

        # Passed all 7 layers
        r["ok"]        = True
        r["px_count"]  = px_b
        r["fail_reason"] = ""
        return r

    if not r["fail_reason"]:
        r["fail_reason"] = (
            "No blob passed all 7 validation checks. "
            "Color pixels found but shape/texture/center do not match dragon fruit.")
    return r


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════
def load_inference_model():
    global model, tf_module, input_size, last_conv_layer_name
    if model is not None: return True
    try:
        import tensorflow as tf
        tf_module = tf
        print(f"\n[AI ENGINE] Loading model: {MODEL_PATH}")
        model = tf.keras.models.load_model(MODEL_PATH)
        input_size = _resolve_input_size(model)
        last_conv_layer_name = _find_last_conv_layer_name(model)
        # Adjust CLASS_NAMES depending on model output shape
        try:
            out_shape = None
            shape = getattr(model, "output_shape", None)
            if isinstance(shape, list):
                shape = shape[0]
            out_shape = shape
            if out_shape is not None and len(out_shape) >= 1:
                n_out = int(out_shape[-1])
                if n_out == 1:
                    # Binary sigmoid model: produce [healthy, rotten]
                    CLASS_NAMES.clear()
                    CLASS_NAMES.extend(["healthy", "rotten"])
                elif n_out == 2:
                    # Two-output softmax: assume order [healthy, rotten]
                    CLASS_NAMES.clear()
                    CLASS_NAMES.extend(["healthy", "rotten"])
                elif n_out == 3:
                    CLASS_NAMES.clear()
                    CLASS_NAMES.extend(["healthy", "rotten", "unknown"])
                else:
                    # Generic: create placeholder names
                    CLASS_NAMES.clear()
                    for i in range(n_out):
                        CLASS_NAMES.append(f"class_{i}")
        except Exception:
            pass
        print(f"[AI ENGINE] Ready | Input: {model.input_shape} | GradCAM: {last_conv_layer_name}")
        if CV2_AVAILABLE:
            print("[AI ENGINE] Validation: ENABLED (7-layer HSV+shape+edge)")
        else:
            print("[AI ENGINE] Validation: DISABLED (install opencv-python-headless)")
        return True
    except ImportError:
        print("\n[AI ERROR] TensorFlow not found. Run: pip install tensorflow\n")
        return False
    except Exception as exc:
        print(f"\n[AI ERROR] {exc}\n")
        return False

def _resolve_input_size(m):
    shape = getattr(m, "input_shape", None)
    if isinstance(shape, list): shape = shape[0]
    if shape and len(shape)>=3 and shape[1] and shape[2]:
        return int(shape[1]), int(shape[2])
    return 224, 224

def _find_last_conv_layer_name(m):
    conv_types = {"Conv2D","DepthwiseConv2D","SeparableConv2D"}
    for layer in reversed(m.layers):
        if layer.__class__.__name__ in conv_types:
            return layer.name
    for layer in reversed(m.layers):
        try:
            os_ = getattr(layer,"output_shape",None)
            if os_ and len(os_)==4: return layer.name
        except TypeError: continue
    return m.layers[-1].name

def _prepare_image(file_storage):
    image = Image.open(file_storage.stream)
    image = ImageOps.exif_transpose(image)
    if image.mode!="RGB": image=image.convert("RGB")
    display_image = ImageOps.pad(image, input_size,
        method=Image.Resampling.BILINEAR, color=(0,0,0), centering=(0.5,0.5))
    image_array = np.asarray(display_image, dtype=np.float32)
    return display_image, image_array

def _to_data_url(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

def _normalize_predictions(predictions):
    scores = np.asarray(predictions, dtype=np.float32).reshape(-1)
    if scores.size==0: return scores
    total = float(np.sum(scores))
    # Handle single-output sigmoid (binary) models by converting to 2-vector
    if scores.size == 1:
        # If the model already emits a probability, keep it as-is.
        # Only apply sigmoid when the value looks like a logit.
        p = float(scores[0])
        if not np.isfinite(p):
            p = 0.5
        elif p < 0.0 or p > 1.0:
            p = 1.0 / (1.0 + np.exp(-p))
        else:
            p = np.clip(p, 0.0, 1.0)
        scores = np.asarray([1.0 - p, p], dtype=np.float32)
        return scores
    # If scores don't look like a normalized softmax, apply softmax
    if np.any(scores < 0) or not np.isfinite(total) or abs(total-1.0) > 1e-3:
        e = np.exp(scores - np.max(scores))
        scores = e / np.sum(e)
    return scores


# ══════════════════════════════════════════════════════════════════════════════
# VALIDATED PREDICTION — main decision function
# ══════════════════════════════════════════════════════════════════════════════
def _choose_prediction_with_validation(predictions, display_image):
    """
    Decision flow:
    1. Normalize softmax
    2. Rotten bypass: AI rotten + high conf + rotten color confirmed → ROTTEN
    3. Run 7-layer validation
    4. Validation fails → UNKNOWN (overrides AI)
    5. AI unknown but validation passed → use 2nd highest class
    6. Apply confidence/margin gates
    """
    scores      = _normalize_predictions(predictions)
    top_idx     = int(np.argmax(scores))
    top_score   = float(scores[top_idx])
    sorted_s    = np.sort(scores)
    second      = float(sorted_s[-2]) if len(sorted_s)>1 else 0.
    margin      = top_score - second
    ai_class    = CLASS_NAMES[top_idx]

    # ── Rotten bypass ─────────────────────────────────────────────────────────
    if ai_class=="rotten" and top_score>=VAL["rot_conf"]:
        if _rotten_confirmed(display_image):
            return "rotten", top_idx, top_score, margin, (
                f"High-conf rotten ({top_score*100:.1f}%) + rotten color confirmed.")
        return "unknown", -1, top_score, margin, (
            f"AI predicted rotten ({top_score*100:.1f}%) but rotten color NOT confirmed.")

    # ── 7-layer validation ────────────────────────────────────────────────────
    val = validate_dragon_fruit(display_image)
    val_note = f" [{val['px_count']}px, edge={val['edge_density']:.3f}]"

    if not val["ok"]:
        # Validation failed → UNKNOWN regardless of AI
        return "unknown", -1, top_score, margin, (
            f"Validation failed: {val['fail_reason']}"
            f" [AI wanted: {ai_class} @ {top_score*100:.1f}%]")

    # Validation passed
    # Override: AI said unknown but DF confirmed
    if ai_class=="unknown" and val["px_count"]>=150:
        sorted_idx = np.argsort(scores)[::-1]
        alt_idx    = int(sorted_idx[1])
        alt_score  = float(scores[alt_idx])
        return CLASS_NAMES[alt_idx], alt_idx, alt_score, margin, (
            f"Validation confirmed DF{val_note}. "
            f"AI voted unknown — using 2nd class: {CLASS_NAMES[alt_idx]} @ {alt_score*100:.1f}%.")

    # Apply confidence/margin gates
    effective = ai_class
    if effective!="unknown" and (top_score<MIN_CONFIDENCE or margin<MIN_MARGIN):
        return "unknown", -1, top_score, margin, (
            f"Conf ({top_score:.2f}) or margin ({margin:.2f}) below gate"
            f" (conf≥{MIN_CONFIDENCE}, margin≥{MIN_MARGIN}). Validation passed{val_note}.")

    return effective, top_idx, top_score, margin, (
        f"Validation passed{val_note}. AI: {ai_class} @ {top_score*100:.1f}%.")


def _top_k_predictions(predictions, k=3):
    idx = [{"index":i,"class_name":CLASS_NAMES[i],"probability":round(float(s),4)}
           for i,s in enumerate(predictions)]
    idx.sort(key=lambda x:x["probability"], reverse=True)
    return idx[:k]


# ══════════════════════════════════════════════════════════════════════════════
# GRAD-CAM (unchanged)
# ══════════════════════════════════════════════════════════════════════════════
def _build_color_map(heatmap, palette):
    heatmap   = np.clip(np.asarray(heatmap,dtype=np.float32),0.,1.)
    positions = np.linspace(0.,1.,len(palette),dtype=np.float32)
    rgb       = np.zeros((*heatmap.shape,3),dtype=np.float32)
    for c in range(3):
        cv_ = np.array([p[c] for p in palette],dtype=np.float32)
        rgb[:,:,c] = np.interp(heatmap,positions,cv_)
    return Image.fromarray(np.clip(rgb,0,255).astype(np.uint8),mode="RGB")

def _overlay_heatmap(base_image, heatmap, palette, blur_radius, opacity):
    hm_img=Image.fromarray((np.clip(heatmap,0.,1.)*255).astype(np.uint8),mode="L")
    if blur_radius>0: hm_img=hm_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    hm_arr=np.asarray(hm_img,dtype=np.float32)/255.
    colorized=_build_color_map(hm_arr,palette).convert("RGBA")
    alpha=np.clip(np.power(hm_arr,0.42)*opacity*255.,0,255).astype(np.uint8)
    colorized.putalpha(Image.fromarray(alpha,mode="L"))
    overlay=Image.new("RGBA",base_image.size,(0,0,0,0))
    overlay.alpha_composite(colorized.resize(base_image.size,Image.Resampling.BILINEAR))
    return ImageEnhance.Contrast(overlay).enhance(1.16)

def _render_blocky_heatmap(heatmap, size=(224, 224)):
    small_size = (14, 14)
    heatmap_img = Image.fromarray((np.clip(heatmap, 0.0, 1.0) * 255).astype(np.uint8), mode="L")
    heatmap_img = heatmap_img.resize(small_size, Image.Resampling.BILINEAR)
    heatmap_img = heatmap_img.resize(size, Image.Resampling.NEAREST)
    hm_arr = np.asarray(heatmap_img, dtype=np.float32) / 255.0

    colorized = _build_color_map(hm_arr, RAINBOW_PALETTE).convert("RGBA")
    alpha = np.clip(np.power(hm_arr, 0.72) * 255.0, 0, 255).astype(np.uint8)
    colorized.putalpha(Image.fromarray(alpha, mode="L"))

    background = Image.new("RGBA", size, (14, 14, 18, 255))
    heatmap_only = Image.alpha_composite(background, colorized)
    heatmap_only = ImageEnhance.Color(heatmap_only).enhance(1.38)
    heatmap_only = ImageEnhance.Contrast(heatmap_only).enhance(1.42)
    return heatmap_only

def _composite_overlay_on_base(base_image, overlay_image):
    base_rgba = base_image.convert("RGBA")
    overlay_rgba = overlay_image.convert("RGBA")
    composite = Image.alpha_composite(base_rgba, overlay_rgba)
    composite = ImageEnhance.Color(composite).enhance(1.08)
    composite = ImageEnhance.Contrast(composite).enhance(1.04)
    return composite

def _extract_focus_points(heatmap, limit=5):
    w=np.array(heatmap,copy=True); H,W=w.shape
    radius=max(3,min(H,W)//14); points=[]
    for _ in range(limit*4):
        flat=int(np.argmax(w)); val=float(w.flat[flat])
        if val<=0.: break
        y,x=np.unravel_index(flat,w.shape)
        points.append({"x":round(((x+.5)/W)*100.,2),
                       "y":round(((y+.5)/H)*100.,2),
                       "score":round(val,4)})
        y0=max(0,y-radius);y1=min(H,y+radius+1)
        x0=max(0,x-radius);x1=min(W,x+radius+1)
        w[y0:y1,x0:x1]=0.
        if len(points)>=limit: break
    return points or [{"x":50.,"y":50.,"score":0.}]

def _draw_pointer_circles(image, focus_points):
    """Draw glowing pointer circles on the image at focus point locations."""
    if not focus_points or len(focus_points) == 0:
        return image
    
    img = image.copy()
    W, H = img.size
    draw = None
    
    # Try to use cv2 for drawing if available, otherwise use PIL
    if CV2_AVAILABLE:
        try:
            import cv2
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            for idx, point in enumerate(focus_points[:5]):
                x = int((point['x'] / 100.0) * W)
                y = int((point['y'] / 100.0) * H)
                radius = max(8, min(W, H) // 25)
                
                # Draw outer glow
                cv2.circle(img_cv, (x, y), radius + 6, (255, 255, 255), 2)
                cv2.circle(img_cv, (x, y), radius + 3, (255, 255, 255), 1)
                # Draw inner circle
                cv2.circle(img_cv, (x, y), radius, (255, 255, 255), -1)
                # Draw center dot
                cv2.circle(img_cv, (x, y), 3, (0, 0, 0), -1)
            
            img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
            return img
        except Exception as e:
            print(f"[WARN] cv2 drawing failed: {e}, falling back to PIL")
    
    # Fallback to PIL drawing
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    for idx, point in enumerate(focus_points[:5]):
        x = (point['x'] / 100.0) * W
        y = (point['y'] / 100.0) * H
        radius = max(8, min(W, H) // 25)
        
        # Draw outer glow
        draw.ellipse([x - radius - 6, y - radius - 6, x + radius + 6, y + radius + 6], 
                    outline=(255, 255, 255), width=2)
        draw.ellipse([x - radius - 3, y - radius - 3, x + radius + 3, y + radius + 3], 
                    outline=(255, 255, 255), width=1)
        # Draw inner circle
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], 
                    fill=(255, 255, 255))
        # Draw center dot
        draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(0, 0, 0))
    
    return img


def _render_pointer_overlay(size, focus_points):
    """Create a transparent RGBA overlay containing only pointer circles."""
    from PIL import ImageDraw, ImageFilter
    overlay = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    W, H = size
    for idx, point in enumerate(focus_points[:5]):
        x = (point['x'] / 100.0) * W
        y = (point['y'] / 100.0) * H
        radius = max(8, min(W, H) // 25)

        # Outer soft glow (semi-transparent larger ellipse)
        glow_bbox = [x - radius - 8, y - radius - 8, x + radius + 8, y + radius + 8]
        draw.ellipse(glow_bbox, fill=(255, 255, 255, 80))

        # Secondary outline
        outline_bbox = [x - radius - 3, y - radius - 3, x + radius + 3, y + radius + 3]
        draw.ellipse(outline_bbox, outline=(255, 255, 255, 180), width=2)

        # Inner circle (solid)
        inner_bbox = [x - radius, y - radius, x + radius, y + radius]
        draw.ellipse(inner_bbox, fill=(255, 255, 255, 220))

        # Center dot
        dot_bbox = [x - 3, y - 3, x + 3, y + 3]
        draw.ellipse(dot_bbox, fill=(0, 0, 0, 255))

    # Optionally apply a small Gaussian blur to soften glow (PIL ImageFilter requires conversion)
    try:
        overlay = overlay.filter(ImageFilter.GaussianBlur(radius=1.2))
    except Exception:
        pass
    return overlay

def _generate_gradcam_bundle(display_image, image_array, class_index):
    """Try standard GradCAM; if it fails, fall back to input-gradient saliency.

    Returns a dict with overlays and focus points, or None on fatal error.
    """
    if tf_module is None or model is None:
        return None
    try:
        inp = tf_module.convert_to_tensor(image_array[None, ...])
        hm = None

        # Attempt GradCAM if we have a conv layer name
        try:
            if last_conv_layer_name and any(getattr(l, 'name', None) == last_conv_layer_name for l in model.layers):
                grad_model = tf_module.keras.models.Model(
                    [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
                )
                with tf_module.GradientTape() as tape:
                    conv_out, preds = grad_model(inp, training=False)
                    # Support both single-output (sigmoid) and multi-output models
                    if getattr(preds, 'shape', None) is not None and preds.shape[-1] == 1:
                        cls_out = preds[:, 0]
                    else:
                        cls_out = preds[:, class_index]
                grads = tape.gradient(cls_out, conv_out)
                if grads is not None:
                    pooled = tf_module.reduce_mean(grads, axis=(0, 1, 2))
                    hm_t = tf_module.reduce_sum(conv_out[0] * pooled, axis=-1)
                    hm_t = tf_module.nn.relu(hm_t)
                    denom = tf_module.reduce_max(hm_t) + tf_module.keras.backend.epsilon()
                    hm = np.clip((hm_t / denom).numpy(), 0.0, 1.0)
        except Exception as e:
            hm = None

        # Fallback: input-gradient saliency
        if hm is None:
            try:
                with tf_module.GradientTape() as tape:
                    tape.watch(inp)
                    preds = model(inp, training=False)
                    if getattr(preds, 'shape', None) is not None and preds.shape[-1] == 1:
                        cls_out = preds[:, 0]
                    else:
                        cls_out = preds[:, class_index]
                grads = tape.gradient(cls_out, inp)
                if grads is None:
                    return None
                # Aggregate absolute gradients across channels
                sal = tf_module.reduce_mean(tf_module.abs(grads), axis=-1)[0]
                sal_np = sal.numpy()
                sal_np = sal_np - sal_np.min()
                if sal_np.max() > 0:
                    sal_np = sal_np / sal_np.max()
                hm = np.clip(sal_np, 0.0, 1.0)
            except Exception as e:
                print(f"[WARN] Saliency fallback failed: {e}")
                return None

        # Ensure heatmap is 2D and resized to model input spatial dims if needed
        if hm is None:
            return None

        # Extract focus points
        focus_points = _extract_focus_points(hm, limit=5)

        heatmap_layer = _render_blocky_heatmap(hm, size=display_image.size)
        # Use vivid rainbow palette for gradcam/heatmap overlays
        palette = RAINBOW_PALETTE
        opacity = 2.05
        try:
            gradcam_layer = _overlay_heatmap(display_image, hm, palette=palette, blur_radius=3, opacity=opacity)
            gradcam_layer = _composite_overlay_on_base(display_image, gradcam_layer)
        except Exception:
            gradcam_layer = display_image

        # Create a transparent pointer-only overlay (separate from gradcam)
        try:
            pointer_img = _render_pointer_overlay(display_image.size, focus_points)
        except Exception as e:
            print(f"[WARN] Pointer rendering failed: {e}")
            pointer_img = display_image.convert("RGBA")

        return {
            "heatmap_overlay": _to_data_url(heatmap_layer),
            "gradcam_overlay": _to_data_url(gradcam_layer),
            "pointer_overlay": _to_data_url(pointer_img),
            "focus_points": focus_points,
            "heatmap_peak": round(float(np.max(hm)), 4),
        }
    except Exception as exc:
        print(f"[WARN] GradCAM overall failure: {exc}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def _run_model_analysis(file_storage):
    if model is None and not load_inference_model():
        return None,(jsonify({"success":False,
            "error":"Model not loaded. Install dependencies."}),500)
    if file_storage is None:
        return None,(jsonify({"success":False,
            "error":"No image file. POST to form param 'file'."}),400)
    if file_storage.filename=="":
        return None,(jsonify({"success":False,"error":"Empty filename."}),400)
    try:
        t0 = time.time()
        display_image, image_array = _prepare_image(file_storage)
        batch = np.expand_dims(image_array, axis=0)

        t1       = time.time()
        raw      = model.predict(batch, verbose=0)[0]
        raw      = _normalize_predictions(raw)
        inf_ms   = (time.time()-t1)*1000
        total_ms = (time.time()-t0)*1000

        predicted_class, raw_index, confidence, margin, decision_reason = \
            _choose_prediction_with_validation(raw, display_image)

        n_out = len(raw)
        probabilities = {CLASS_NAMES[i]: float(raw[i]) for i in range(min(len(CLASS_NAMES), n_out))}
        if predicted_class in CLASS_NAMES:
            class_index = CLASS_NAMES.index(predicted_class)
        else:
            # unknown or out-of-distribution label — fall back to argmax
            class_index = int(np.argmax(raw))
        top_predictions = _top_k_predictions(raw, k=min(3, n_out))

        explanation = _generate_gradcam_bundle(display_image, image_array, class_index)
        if explanation is None:
            explanation = {
                "heatmap_overlay":_to_data_url(display_image.convert("RGBA")),
                "gradcam_overlay":_to_data_url(display_image.convert("RGBA")),
                "focus_points":[{"x":50.,"y":50.,"score":0.}],
                "heatmap_peak":0.0,
            }

        print(f"[INFERENCE] {predicted_class.upper()} | "
              f"{confidence*100:.2f}% | {inf_ms:.0f}ms | {decision_reason[:80]}")

        return {
            "success":True,
            "prediction":predicted_class,
            "raw_prediction":CLASS_NAMES[int(np.argmax(raw))],
            "raw_prediction_index":int(np.argmax(raw)),
            "confidence":round(confidence*100.,2),
            "confidence_text":f"{confidence*100.:.2f}%",
            "probability":round(confidence,4),
            "margin":round(margin,4),
            "raw_outputs":[round(float(s),4) for s in raw.tolist()],
            "top_predictions":top_predictions,
            "latency_ms":round(inf_ms,1),
            "total_latency_ms":round(total_ms,1),
            "probabilities":probabilities,
            "class_names":CLASS_NAMES,
            "decision_reason":decision_reason,
            "display_image":_to_data_url(display_image.convert("RGBA")),
            "heatmap_overlay":explanation["heatmap_overlay"],
            "gradcam_overlay":explanation["gradcam_overlay"],
            "focus_points":explanation["focus_points"],
            "heatmap_peak":explanation["heatmap_peak"],
            "model_info":{
                "input_size":list(input_size),
                "last_conv_layer":last_conv_layer_name,
                "min_confidence":MIN_CONFIDENCE,
                "min_margin":MIN_MARGIN,
                "validation_enabled":CV2_AVAILABLE,
                "validation_thresholds":VAL,
            },
        }, None
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return None,(jsonify({"success":False,"error":str(exc)}),500)


# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/health", methods=["GET"])
def health():
    installed=False
    try:
        import tensorflow as tf; installed=True
    except ImportError: pass
    return jsonify({
        "status":"online" if model else "offline",
        "tensorflow_installed":installed,
        "opencv_installed":CV2_AVAILABLE,
        "model_loaded":model is not None,
        "model_path":MODEL_PATH,
        "model_candidates":MODEL_CANDIDATES,
        "input_size":list(input_size),
        "last_conv_layer":last_conv_layer_name,
        "class_names":CLASS_NAMES,
        "confidence_threshold":MIN_CONFIDENCE,
        "margin_threshold":MIN_MARGIN,
        "validation_enabled":CV2_AVAILABLE,
        "validation_thresholds":VAL,
        "prediction_mode":"3-class-softmax + 7-layer-hsv-shape-edge-validation",
        "api_name":"Dragonfruit Diagnostics API v3",
    })


@app.route('/config', methods=['GET', 'POST'])
def config():
    """Get or set runtime configuration for the inference engine.

    GET returns the current runtime config. POST accepts a JSON body with any
    of these fields: preprocess_mode, force_classification, min_confidence,
    min_margin, last_conv_layer_override. The server applies numeric fields
    immediately so the frontend can tune thresholds at runtime.
    """
    global MIN_CONFIDENCE, MIN_MARGIN, last_conv_layer_name
    if request.method == 'GET':
        out = RUNTIME_CONFIG.copy()
        out.update({
            'model_loaded': model is not None,
            'last_conv_layer': last_conv_layer_name,
            'input_size': list(input_size),
        })
        return jsonify(out)

    # POST: apply new config
    try:
        payload = request.get_json(force=True)
        if not isinstance(payload, dict):
            raise ValueError('Invalid payload')
        # Allowed keys
        if 'preprocess_mode' in payload:
            RUNTIME_CONFIG['preprocess_mode'] = str(payload['preprocess_mode'])
        if 'force_classification' in payload:
            RUNTIME_CONFIG['force_classification'] = bool(payload['force_classification'])
        if 'min_confidence' in payload:
            v = float(payload['min_confidence'])
            MIN_CONFIDENCE = v
            RUNTIME_CONFIG['min_confidence'] = v
        if 'min_margin' in payload:
            v = float(payload['min_margin'])
            MIN_MARGIN = v
            RUNTIME_CONFIG['min_margin'] = v
        if 'last_conv_layer_override' in payload:
            val = payload['last_conv_layer_override']
            RUNTIME_CONFIG['last_conv_layer_override'] = val if val else None
            # apply override if provided
            if val:
                last_conv_layer_name = val
        return jsonify({**RUNTIME_CONFIG, 'success': True})
    except Exception as exc:
        return jsonify({'success': False, 'error': str(exc)}), 400


@app.route('/reload_model', methods=['POST'])
def reload_model():
    """Reload the inference model at runtime.

    This forces the server to discard the loaded model and call
    `load_inference_model()` again. Returns JSON with success flag.
    """
    global model, tf_module, last_conv_layer_name
    try:
        # unset current model to force reload
        model = None
        tf_module = None
        ok = load_inference_model()
        return jsonify({'success': bool(ok), 'mock_inference': RUNTIME_CONFIG.get('mock_inference', False)})
    except Exception as exc:
        return jsonify({'success': False, 'error': str(exc)}), 500

@app.route("/analyze", methods=["POST"])
def analyze():
    result,err=_run_model_analysis(request.files.get("file"))
    if err: return err
    return jsonify(result)

@app.route("/predict", methods=["POST"])
def predict():
    result,err=_run_model_analysis(request.files.get("file"))
    if err: return err
    return jsonify(result)

if __name__=="__main__":
    load_inference_model()
    print("[SERVER] http://127.0.0.1:5000 (dev) — debug=False, use_reloader=False")
    # For local debugging keep a single process (disable reloader) and bind to
    # localhost to avoid system firewall or network differences causing
    # intermittent connection refused errors in the browser.
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)