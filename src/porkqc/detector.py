"""Thin wrappers around Ultralytics YOLOv8 for training, validation and prediction.

Why YOLOv8:
    * single-stage detector -> one forward pass per image, suitable for
      real-time conveyor inspection;
    * transfer learning from COCO-pretrained weights works well with the
      ~1,000 distinct frames available here;
    * the Roboflow export is already in YOLOv8 format.

``ultralytics`` is imported inside each function so the rest of the package
(and its unit tests) work without PyTorch installed.
"""

from __future__ import annotations

from pathlib import Path

from porkqc.inspection import Detection


def resolve_device(device: str = "auto") -> str:
    """Choose the compute device.

    Args:
        device: ``"auto"``, ``"cpu"``, ``"mps"`` or a CUDA index such as ``"0"``.

    Returns:
        A device string accepted by Ultralytics.
    """
    if device != "auto":
        return device
    import torch

    if torch.cuda.is_available():
        return "0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train(weights: str, data_yaml: str | Path, out_dir: str | Path, name: str,
          epochs: int = 50, imgsz: int = 640, batch: int = 16, patience: int = 15,
          device: str = "auto", workers: int = 2, seed: int = 42, fraction: float = 1.0) -> Path:
    """Fine-tune a pretrained YOLOv8 model.

    Ultralytics applies its own online augmentation (mosaic, flips, HSV jitter)
    on top of the Roboflow exposure augmentation. Early stopping (``patience``)
    monitors validation mAP so the number of epochs need not be tuned by hand.

    Args:
        weights: Pretrained checkpoint, e.g. ``"yolov8n.pt"`` (downloaded automatically).
        data_yaml: Dataset YAML (use the absolute-path one from `porkqc.data.write_data_yaml`).
        out_dir: Folder for training runs.
        name: Run name (sub-folder).
        epochs: Maximum epochs.
        imgsz: Square training input size in pixels.
        batch: Batch size (reduce if you run out of memory).
        patience: Early-stopping patience in epochs.
        device: See `resolve_device`.
        workers: Data-loader worker processes (keep low on Windows).
        seed: Random seed for repeatability.
        fraction: Fraction of training images to use (data-requirement study).

    Returns:
        Path to the best checkpoint (``weights/best.pt``).
    """
    from ultralytics import YOLO

    model = YOLO(weights)
    model.train(
        data=str(data_yaml), epochs=epochs, imgsz=imgsz, batch=batch, patience=patience,
        device=resolve_device(device), workers=workers, seed=seed, deterministic=True,
        project=str(Path(out_dir).resolve()), name=name, exist_ok=True, fraction=fraction,
        plots=True, verbose=True,
    )
    return Path(model.trainer.best)


def validate(weights: str | Path, data_yaml: str | Path, split: str = "test",
             imgsz: int = 640, device: str = "auto", out_dir: str | Path | None = None,
             name: str = "val") -> dict:
    """Compute COCO-style detection metrics on one split.

    Args:
        weights: Trained checkpoint.
        data_yaml: Dataset YAML.
        split: ``"val"`` or ``"test"``.
        imgsz: Inference size.
        device: See `resolve_device`.
        out_dir: Where Ultralytics saves plots (PR curves, confusion matrix).
        name: Sub-folder name for this validation run.

    Returns:
        Dict with ``precision``, ``recall``, ``mAP50``, ``mAP50_95``, per-class
        ``AP50`` and inference speed (ms/image).
    """
    from ultralytics import YOLO

    from porkqc.config import CLASS_NAMES

    model = YOLO(str(weights))
    kwargs = dict(data=str(data_yaml), split=split, imgsz=imgsz,
                  device=resolve_device(device), plots=True, verbose=False, exist_ok=True)
    if out_dir:
        kwargs.update(project=str(Path(out_dir).resolve()), name=name)
    m = model.val(**kwargs)
    per_class = {CLASS_NAMES[int(c)]: float(ap) for c, ap in zip(m.box.ap_class_index, m.box.ap50)}
    return {
        "precision": float(m.box.mp),
        "recall": float(m.box.mr),
        "mAP50": float(m.box.map50),
        "mAP50_95": float(m.box.map),
        "AP50_per_class": per_class,
        "speed_ms": {k: float(v) for k, v in m.speed.items()},
    }


def predict(weights: str | Path, images: list[Path], imgsz: int = 640,
            conf: float = 0.01, device: str = "auto") -> dict[str, list[Detection]]:
    """Run the detector on a list of images.

    A low ``conf`` is used on purpose: all candidate boxes are stored once and
    the decision threshold is applied afterwards, so threshold sensitivity can
    be studied without re-running the network.

    Args:
        weights: Trained checkpoint.
        images: Image paths.
        imgsz: Inference size.
        conf: Minimum confidence kept.
        device: See `resolve_device`.

    Returns:
        ``{image_name: [Detection, ...]}``.
    """
    from ultralytics import YOLO

    model = YOLO(str(weights))
    out: dict[str, list[Detection]] = {}
    dev = resolve_device(device)
    # Process in chunks: passing thousands of paths at once can exhaust RAM.
    for i in range(0, len(images), 32):
        chunk = [str(p) for p in images[i:i + 32]]
        for r in model.predict(chunk, imgsz=imgsz, conf=conf, device=dev, verbose=False):
            b = r.boxes
            out[Path(r.path).name] = [
                Detection(int(c), float(s), tuple(float(v) for v in xy))
                for c, s, xy in zip(b.cls.cpu().numpy(), b.conf.cpu().numpy(), b.xyxy.cpu().numpy())
            ]
    return out


def model_complexity(weights: str | Path, imgsz: int = 640) -> dict:
    """Parameter count, GFLOPs and file size of a checkpoint.

    Args:
        weights: Checkpoint path or name.
        imgsz: Input size for the FLOP estimate.

    Returns:
        Dict with ``params_M``, ``gflops`` (at ``imgsz``) and ``file_MB``.
    """
    from ultralytics import YOLO

    from porkqc.profiling import scale_flops

    model = YOLO(str(weights))
    params = sum(p.numel() for p in model.model.parameters())
    gflops = None
    try:
        info = model.info(verbose=False)  # (layers, params, gradients, GFLOPs@640)
        gflops = scale_flops(float(info[3]), imgsz)
    except Exception:  # FLOP counting needs the optional 'thop' package
        pass
    path = Path(weights)
    return {
        "params_M": params / 1e6,
        "gflops": gflops,
        "file_MB": path.stat().st_size / 1e6 if path.exists() else None,
    }
