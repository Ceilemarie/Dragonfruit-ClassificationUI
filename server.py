import base64
import io
import os
import time
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.datastructures import FileStorage
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw

# Optional OpenCV — used for validation heuristics. If unavailable we
# gracefully degrade to a permissive "validation passed" behaviour.
try:
    import cv2
    CV2_AVAILABLE = True
except Exception:
    cv2 = None
    CV2_AVAILABLE = False

app = Flask(__name__)
CORS(app)

# Basic configuration and globals
MODEL_CANDIDATES = ["Model1_EfficientNetB0_best.keras"]
MODEL_PATH = os.path.join(os.path.dirname(__file__), MODEL_CANDIDATES[0])
CLASS_NAMES = ["healthy", "rotten", "unknown"]
RAINBOW_PALETTE = [(63,0,125),(0,58,168),(0,196,255),(0,230,118),(255,214,0),(255,111,0),(244,67,54),(170,0,110)]
MIN_CONFIDENCE = 0.55
MIN_MARGIN = 0.12

model = None
tf_module = None
input_size = (224, 224)
last_conv_layer_name = None
LAST_DISPLAY_IMAGE = None

RUNTIME_CONFIG = {
    "preprocess_mode": "auto",
    "force_classification": False,
    "min_confidence": MIN_CONFIDENCE,
    "min_margin": MIN_MARGIN,
    "last_conv_layer_override": None,
    "mock_inference": False,
}

VAL = dict(
    px_min=120,
    min_solid=0.42,
    min_fill=0.30,
    max_edge=0.32,
    max_iqr=18.0,
    min_cf=0.18,
    rot_conf=0.68,
    rot_px=400,
)

# --- Utilities ---------------------------------------------------------------

def _to_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


def _prepare_image(file_storage: FileStorage):
    img = Image.open(file_storage.stream)
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    display = ImageOps.pad(img, input_size, method=Image.Resampling.BILINEAR, color=(0,0,0))
    arr = np.asarray(display, dtype=np.float32)
    return display, arr


def _image_to_filestorage(image: Image.Image, name: str = "image.png"):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    return FileStorage(stream=buf, filename=name, content_type="image/png")


def _normalize_predictions(predictions):
    scores = np.asarray(predictions, dtype=np.float32).reshape(-1)
    if scores.size == 0:
        return scores
    if scores.size == 1:
        p = float(scores[0])
        if not np.isfinite(p):
            p = 0.5
        elif p < 0.0 or p > 1.0:
            p = 1.0 / (1.0 + np.exp(-p))
        p = np.clip(p, 0.0, 1.0)
        return np.asarray([1.0 - p, p], dtype=np.float32)
    total = float(np.sum(scores))
    if np.any(scores < 0) or not np.isfinite(total) or abs(total - 1.0) > 1e-3:
        e = np.exp(scores - np.max(scores))
        scores = e / np.sum(e)
    return scores

# Simple color mask used by the validation heuristic (keeps behaviour small)
def _df_mask(hsv):
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    pink = (h >= 145) & (h <= 168) & (s >= 115) & (v >= 60)
    crimson = ((h <= 4) | (h >= 172)) & (s >= 165) & (v >= 55)
    return (pink | crimson).astype(np.uint8)


def validate_dragon_fruit(pil_224: Image.Image):
    """Run a lightweight validation; if OpenCV is not present assume pass."""
    if not CV2_AVAILABLE:
        return {"ok": True, "px_count": 999, "edge_density": 0.0, "fail_reason": ""}
    bgr = cv2.cvtColor(np.array(pil_224), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    raw = _df_mask(hsv)
    total = int(np.sum(raw))
    if total < VAL["px_min"]:
        return {"ok": False, "px_count": total, "edge_density": 0.0, "fail_reason": "Too few DF-colored pixels"}
    return {"ok": True, "px_count": total, "edge_density": 0.0, "fail_reason": ""}

# --- Model loading (robust and idempotent) ----------------------------------

def _resolve_input_size(m):
    shape = getattr(m, "input_shape", None)
    if isinstance(shape, list):
        shape = shape[0]
    if shape and len(shape) >= 3 and shape[1] and shape[2]:
        return int(shape[1]), int(shape[2])
    return 224, 224


def _find_last_conv_layer_name(m):
    conv_types = {"Conv2D", "DepthwiseConv2D", "SeparableConv2D"}
    for layer in reversed(m.layers):
        if layer.__class__.__name__ in conv_types:
            return layer.name
    for layer in reversed(m.layers):
        try:
            os_ = getattr(layer, "output_shape", None)
            if os_ and len(os_) == 4:
                return layer.name
        except Exception:
            continue
    return m.layers[-1].name


def load_inference_model():
    global model, tf_module, input_size, last_conv_layer_name
    if model is not None:
        return True
    try:
        import tensorflow as tf
        tf_module = tf
        print(f"[AI ENGINE] Loading model: {MODEL_PATH}")
        model = tf.keras.models.load_model(MODEL_PATH)
        input_size = _resolve_input_size(model)
        last_conv_layer_name = _find_last_conv_layer_name(model)
        print(f"[AI ENGINE] Model loaded | input_size={input_size} | last_conv={last_conv_layer_name}")
        return True
    except ImportError:
        print("[AI ERROR] tensorflow not installed — running in mock mode")
        model = None
        tf_module = None
        return False
    except Exception as exc:
        print(f"[AI ERROR] failed to load model: {exc}")
        model = None
        return False

# --- Simple Grad-CAM / overlay stub (keeps runtime stable) ------------------

def _render_pointer_overlay(size, focus_points):
    base = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(base)
    W, H = size
    for p in focus_points:
        x = int((p["x"] / 100.0) * W)
        y = int((p["y"] / 100.0) * H)
        r = int(0.06 * min(W, H))
        bbox = [x - r, y - r, x + r, y + r]
        draw.ellipse(bbox, fill=(255, 255, 255, 180))
        draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(0, 0, 0, 255))
    return base


def _generate_gradcam_bundle(display_image, image_array, class_index):
    # Attempt a true Grad-CAM when TensorFlow is available; otherwise fall
    # back to a lightweight translucent overlay for stability.
    try:
        if tf_module is None or model is None:
            raise RuntimeError("TF or model unavailable")

        tf = tf_module

        # allow runtime override of the conv layer
        layer_name = RUNTIME_CONFIG.get("last_conv_layer_override") or last_conv_layer_name
        if not layer_name:
            layer_name = _find_last_conv_layer_name(model)

        try:
            last_conv = model.get_layer(layer_name)
        except Exception:
            last_conv = None

        if last_conv is None:
            raise RuntimeError(f"Couldn't locate conv layer: {layer_name}")

        # Build a model that maps the input image to the activations
        # of the last conv layer as well as the model predictions.
        grad_model = tf.keras.models.Model([model.inputs], [last_conv.output, model.output])

        # Prepare input batch consistent with the inference path
        inp = tf.cast(tf.convert_to_tensor(np.expand_dims(image_array, axis=0)), tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(inp)
            conv_outputs, predictions = grad_model(inp)
            loss = predictions[:, class_index]

        grads = tape.gradient(loss, conv_outputs)
        if grads is None:
            raise RuntimeError("Gradient computation failed")

        # Global-average-pool the gradients and weight the conv outputs
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
        conv_outputs = conv_outputs[0].numpy()

        for i in range(pooled_grads.shape[-1]):
            conv_outputs[:, :, i] *= pooled_grads[i]

        heatmap = np.sum(conv_outputs, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)

        # Peak focus point (as percentage coords)
        try:
            py, px = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
            fx = float(px) / float(heatmap.shape[1]) * 100.0
            fy = float(py) / float(heatmap.shape[0]) * 100.0
        except Exception:
            fx, fy = 50.0, 50.0

        # Create a colored heatmap image. Prefer OpenCV's applyColorMap if present.
        try:
            if CV2_AVAILABLE:
                hm_uint8 = np.uint8(255.0 * heatmap)
                colored = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
                colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
                heat_img = Image.fromarray(colored).convert("RGBA")
                heat_img = heat_img.resize(display_image.size, Image.Resampling.BILINEAR)
            else:
                # Fallback: grayscale -> colorize then convert to RGBA
                hm = Image.fromarray(np.uint8(255.0 * heatmap)).resize(display_image.size, Image.Resampling.BILINEAR)
                heat_img = ImageOps.colorize(hm.convert("L"), black="#000000", white="#ff0000").convert("RGBA")

            # Build an alpha mask from the heatmap so the gradcam container contains
            # only the heatmap (no underlying uploaded image blended in).
            alpha_mask = Image.fromarray(np.uint8(np.clip(heatmap * 255.0, 0, 255))).resize(display_image.size, Image.Resampling.BILINEAR)
            heat_img.putalpha(alpha_mask)

            gradcam_only = heat_img
            transparent = Image.new("RGBA", display_image.size, (0, 0, 0, 0))

            focus_points = [{"x": fx, "y": fy, "score": float(np.max(heatmap))}]
            pointer = _render_pointer_overlay(display_image.size, focus_points)

            return {
                # keep `heatmap_overlay` blank (transparent) to avoid the UI loading
                # the original image into the heatmap container; `gradcam_overlay`
                # is the heatmap-only image with transparency.
                "heatmap_overlay": _to_data_url(transparent),
                "gradcam_overlay": _to_data_url(gradcam_only),
                "pointer_overlay": _to_data_url(pointer),
                "focus_points": focus_points,
                "heatmap_peak": float(np.max(heatmap)),
            }
        except Exception:
            # If anything goes wrong during colormap/overlay, fall back
            raise
    except Exception:
        # Original lightweight fallback to keep runtime stable
        try:
            gray = display_image.convert("L").convert("RGBA")
            heat = Image.blend(display_image.convert("RGBA"), gray, alpha=0.6)
            focus_points = [{"x": 50.0, "y": 50.0, "score": 0.5}]
            pointer = _render_pointer_overlay(display_image.size, focus_points)
            return {
                "heatmap_overlay": _to_data_url(heat),
                "gradcam_overlay": _to_data_url(heat),
                "pointer_overlay": _to_data_url(pointer),
                "focus_points": focus_points,
                "heatmap_peak": 0.0,
            }
        except Exception:
            return None

# --- Inference pipeline -----------------------------------------------------

def _run_model_analysis(file_storage):
    global LAST_DISPLAY_IMAGE
    if model is None:
        ok = load_inference_model()
        if not ok and RUNTIME_CONFIG.get("mock_inference", False) is False:
            return None, (jsonify({"success": False, "error": "Model not loaded."}), 500)
    if file_storage is None:
        return None, (jsonify({"success": False, "error": "No file provided."}), 400)
    try:
        display_image, image_array = _prepare_image(file_storage)
        # Mock inference if TF isn't present
        if tf_module is None or model is None:
            probs = np.array([0.85, 0.10, 0.05], dtype=np.float32)
        else:
            batch = np.expand_dims(image_array, axis=0)
            raw = model.predict(batch, verbose=0)[0]
            probs = _normalize_predictions(raw)
        pred_idx = int(np.argmax(probs))
        prediction = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else f"class_{pred_idx}"
        confidence = float(probs[pred_idx])
        explanation = _generate_gradcam_bundle(display_image, image_array, pred_idx)
        try:
            LAST_DISPLAY_IMAGE = display_image.copy()
        except Exception:
            LAST_DISPLAY_IMAGE = None
        out = {
            "success": True,
            "prediction": prediction,
            "confidence": round(confidence * 100.0, 2),
            "probabilities": {CLASS_NAMES[i]: float(probs[i]) for i in range(min(len(probs), len(CLASS_NAMES)))},
            "display_image": _to_data_url(display_image.convert("RGBA")),
            "gradcam_overlay": explanation["gradcam_overlay"] if explanation else _to_data_url(display_image.convert("RGBA")),
            "focus_points": explanation["focus_points"] if explanation else [],
        }
        return out, None
    except Exception as exc:
        return None, (jsonify({"success": False, "error": str(exc)}), 500)

# --- Routes -----------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    installed = False
    try:
        import tensorflow as tf; installed = True
    except Exception:
        installed = False
    return jsonify({
        "status": "online",
        "tensorflow_installed": installed,
        "opencv_installed": CV2_AVAILABLE,
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
        "input_size": list(input_size),
    })


@app.route('/analyze', methods=['POST'])
def analyze():
    file_storage = request.files.get('file')
    result, err = _run_model_analysis(file_storage)
    if err:
        return err
    return jsonify(result)


@app.route('/pointer_overlay', methods=['POST'])
def pointer_overlay():
    global LAST_DISPLAY_IMAGE
    if LAST_DISPLAY_IMAGE is None:
        return jsonify({'success': False, 'error': 'No cached image; run analyze first.'}), 400
    focus_x = request.form.get('focus_x')
    focus_y = request.form.get('focus_y')
    if focus_x is None or focus_y is None:
        return jsonify({'success': False, 'error': 'Missing focus_x or focus_y.'}), 400
    try:
        fx = float(focus_x)
        fy = float(focus_y)
        cp = float(request.form.get('crop_pct') or 20.0)
        focus_points = [{'x': fx, 'y': fy, 'score': 1.0}]
        try:
            W, H = LAST_DISPLAY_IMAGE.size
            frac = max(0.05, min(0.95, cp / 100.0))
            box_w = int(min(W, H) * frac)
            box_h = box_w
            cx = int((fx / 100.0) * W)
            cy = int((fy / 100.0) * H)
            left = max(0, cx - box_w // 2)
            upper = max(0, cy - box_h // 2)
            right = min(W, left + box_w)
            lower = min(H, upper + box_h)
            cropped = LAST_DISPLAY_IMAGE.crop((left, upper, right, lower)).resize((224,224), Image.Resampling.BILINEAR)
        except Exception:
            cropped = LAST_DISPLAY_IMAGE.resize((224,224), Image.Resampling.BILINEAR)
        try:
            val = validate_dragon_fruit(cropped)
            valid = bool(val.get('ok'))
        except Exception:
            valid = True
        overlay = _render_pointer_overlay(LAST_DISPLAY_IMAGE.size, [{'x': fx, 'y': fy, 'score': 1.0}])
        return jsonify({ 'success': True, 'pointer_overlay': _to_data_url(overlay), 'focus_points': focus_points, 'validation_ok': valid })
    except Exception as exc:
        return jsonify({ 'success': False, 'error': str(exc) }), 500


@app.route('/predict', methods=['POST'])
def predict():
    file_storage = request.files.get('file')
    focus_x = request.form.get('focus_x')
    focus_y = request.form.get('focus_y')
    crop_pct = request.form.get('crop_pct')
    if file_storage is None:
        if LAST_DISPLAY_IMAGE is None:
            return jsonify({'success': False, 'error': 'No cached image; run analyze first.'}), 400
        source_img = LAST_DISPLAY_IMAGE.copy()
    else:
        try:
            file_storage.stream.seek(0)
            source_img = Image.open(file_storage.stream)
            source_img = ImageOps.exif_transpose(source_img)
        except Exception as e:
            return jsonify({'success': False, 'error': f'Could not read image: {e}'}), 400

    if focus_x is not None or focus_y is not None:
        try:
            fx = float(focus_x) if focus_x is not None else 50.0
            fy = float(focus_y) if focus_y is not None else 50.0
            cp = float(crop_pct) if crop_pct is not None else 25.0
            W, H = source_img.size
            frac = max(0.05, min(0.95, cp / 100.0))
            box_w = int(min(W, H) * frac)
            box_h = box_w
            cx = int((fx / 100.0) * W)
            cy = int((fy / 100.0) * H)
            left = max(0, cx - box_w // 2)
            upper = max(0, cy - box_h // 2)
            right = min(W, left + box_w)
            lower = min(H, upper + box_h)
            source_img = source_img.crop((left, upper, right, lower))
        except Exception as e:
            print(f"[WARN] crop failed: {e}")

    file_storage = _image_to_filestorage(source_img, name="crop.png")
    result, err = _run_model_analysis(file_storage)
    if err:
        return err
    return jsonify(result)


if __name__ == '__main__':
    load_inference_model()
    app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)
