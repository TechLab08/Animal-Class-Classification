import base64
import html
import os
import tempfile
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

import numpy as np
from PIL import Image


HOST = "127.0.0.1"
PORT = 8000
MODEL_PATH = Path("Best_Cattle_Breed.h5")
DATA_DIR = Path("data")
LABELS_PATH = Path("labels.txt")

model = None
tf = None
class_names = []
load_error = None


def try_load_model():
    global model, tf, class_names, load_error
    if model is not None or load_error is not None:
        return

    try:
        import tensorflow as tensorflow

        tf = tensorflow
        model = tf.keras.models.load_model(MODEL_PATH)
        class_names = load_class_names()
    except Exception as exc:
        load_error = f"{type(exc).__name__}: {exc}"


def load_class_names():
    if LABELS_PATH.is_file():
        labels = [line.strip() for line in LABELS_PATH.read_text(encoding="utf-8").splitlines()]
        labels = [label for label in labels if label]
        if labels:
            return labels

    if DATA_DIR.is_dir():
        labels = sorted(path.name for path in DATA_DIR.iterdir() if path.is_dir())
        if labels:
            return labels

    output_shape = model.output_shape
    num_classes = output_shape[-1] if isinstance(output_shape, tuple) else output_shape[0][-1]
    return [f"Class {index}" for index in range(num_classes)]


def predict_image(image_path):
    try_load_model()
    if load_error:
        raise RuntimeError(
            "TensorFlow is not available, so the model cannot run. "
            f"Install it first with: python -m pip install tensorflow. Details: {load_error}"
        )

    img = Image.open(image_path).convert("RGB").resize((224, 224))
    img = np.asarray(img, dtype=np.float32)
    img = tf.keras.applications.efficientnet_v2.preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    preds = model.predict(img, verbose=0)
    class_id = int(np.argmax(preds[0]))
    confidence = float(preds[0][class_id] * 100)
    label = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
    return label, confidence


def image_preview_data(image_path):
    image = Image.open(image_path).convert("RGB")
    image.thumbnail((520, 360))
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as preview_file:
        preview_path = Path(preview_file.name)
    try:
        image.save(preview_path, format="JPEG", quality=88)
        encoded = base64.b64encode(preview_path.read_bytes()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"
    finally:
        preview_path.unlink(missing_ok=True)


def parse_upload(body, content_type):
    marker = "boundary="
    if marker not in content_type:
        raise ValueError("Missing multipart boundary.")

    boundary = ("--" + content_type.split(marker, 1)[1].strip().strip('"')).encode("utf-8")
    for part in body.split(boundary):
        if b'name="image"' not in part:
            continue
        if b"\r\n\r\n" not in part:
            continue

        headers, file_data = part.split(b"\r\n\r\n", 1)
        file_data = file_data.rsplit(b"\r\n", 1)[0]
        if not file_data:
            raise ValueError("No image was uploaded.")

        filename = "upload.jpg"
        for header in headers.decode("utf-8", "ignore").splitlines():
            if "filename=" in header:
                filename = unquote(header.split("filename=", 1)[1].strip().strip('"')) or filename

        suffix = Path(filename).suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png"}:
            suffix = ".jpg"

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as upload_file:
            upload_file.write(file_data)
            return Path(upload_file.name)

    raise ValueError("Image field was not found in the upload.")


def render_page(result=None, error=None, preview=None):
    result_html = ""
    if result:
        label, confidence = result
        result_html = f"""
        <section class="result ok">
          <span>Prediction</span>
          <strong>{html.escape(label)}</strong>
          <small>{confidence:.2f}% confidence</small>
        </section>
        """
    elif error:
        result_html = f"""
        <section class="result error">
          <span>Cannot predict yet</span>
          <strong>{html.escape(error)}</strong>
        </section>
        """

    preview_html = f'<img src="{preview}" alt="Selected cattle image preview">' if preview else ""
    status = "Model ready" if model is not None else "TensorFlow/model loads on prediction"

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Cattle Breed Classifier</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: Arial, Helvetica, sans-serif;
      background: #f5f7f6;
      color: #17211b;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; min-height: 100vh; }}
    main {{ width: min(940px, calc(100% - 32px)); margin: 0 auto; padding: 36px 0; }}
    header {{ display: flex; justify-content: space-between; gap: 18px; align-items: end; margin-bottom: 24px; }}
    h1 {{ font-size: 30px; margin: 0 0 8px; letter-spacing: 0; }}
    p {{ margin: 0; color: #516158; line-height: 1.5; }}
    .status {{ font-size: 13px; color: #3b6f52; background: #e5f4eb; padding: 8px 10px; border-radius: 6px; white-space: nowrap; }}
    .panel {{ background: #ffffff; border: 1px solid #dce4df; border-radius: 8px; padding: 22px; }}
    form {{ display: grid; grid-template-columns: 1fr auto; gap: 12px; align-items: center; }}
    input[type=file] {{ width: 100%; border: 1px solid #cbd7d0; border-radius: 6px; padding: 10px; background: #fbfcfb; }}
    button {{ border: 0; border-radius: 6px; padding: 12px 18px; background: #1f6b46; color: white; font-weight: 700; cursor: pointer; }}
    button:hover {{ background: #185638; }}
    .content {{ display: grid; grid-template-columns: minmax(0, 1.15fr) minmax(260px, .85fr); gap: 18px; margin-top: 18px; }}
    .preview {{ min-height: 280px; display: grid; place-items: center; background: #eef3f0; border: 1px dashed #b8c7be; border-radius: 8px; overflow: hidden; }}
    .preview img {{ max-width: 100%; max-height: 420px; display: block; }}
    .result {{ border-radius: 8px; padding: 18px; border: 1px solid; }}
    .result span {{ display: block; font-size: 12px; text-transform: uppercase; color: #65726b; margin-bottom: 8px; }}
    .result strong {{ display: block; font-size: 20px; line-height: 1.35; overflow-wrap: anywhere; }}
    .result small {{ display: block; margin-top: 8px; color: #526259; }}
    .ok {{ background: #eef8f2; border-color: #b9dec8; }}
    .error {{ background: #fff2f0; border-color: #f0c0b8; }}
    @media (max-width: 720px) {{
      header, form, .content {{ grid-template-columns: 1fr; display: grid; }}
      .status {{ white-space: normal; }}
      button {{ width: 100%; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>Cattle Breed Classifier</h1>
        <p>Upload a JPG or PNG cattle image to classify it with the saved model.</p>
      </div>
      <div class="status">{html.escape(status)}</div>
    </header>
    <section class="panel">
      <form action="/predict" method="post" enctype="multipart/form-data">
        <input name="image" type="file" accept=".jpg,.jpeg,.png,image/jpeg,image/png" required>
        <button type="submit">Predict</button>
      </form>
      <div class="content">
        <div class="preview">{preview_html or "No image selected"}</div>
        <div>{result_html or ""}</div>
      </div>
    </section>
  </main>
</body>
</html>"""


class AppHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.respond(render_page())

    def do_POST(self):
        if self.path != "/predict":
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        upload_path = None
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            upload_path = parse_upload(body, self.headers.get("Content-Type", ""))
            preview = image_preview_data(upload_path)
            result = predict_image(upload_path)
            self.respond(render_page(result=result, preview=preview))
        except Exception as exc:
            details = str(exc) or traceback.format_exc(limit=1)
            preview = image_preview_data(upload_path) if upload_path and upload_path.exists() else None
            self.respond(render_page(error=details, preview=preview), HTTPStatus.BAD_REQUEST)
        finally:
            if upload_path:
                upload_path.unlink(missing_ok=True)

    def respond(self, content, status=HTTPStatus.OK):
        encoded = content.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format, *args):
        print(f"{self.address_string()} - {format % args}")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    server = ThreadingHTTPServer((HOST, PORT), AppHandler)
    print(f"Open http://{HOST}:{PORT}")
    server.serve_forever()
