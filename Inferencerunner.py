"""
inference_runner.py — Vitis AI DPU Inference Interface
=======================================================
Wraps the VART (Vitis AI Runtime) API to:
  1. Load an xmodel (compiled DPU subgraph) from disk.
  2. Accept a raw OpenCV BGR frame.
  3. Feed the frame through the HLS preprocessing kernel (or a software
     fallback when running on a dev machine without the overlay).
  4. Run DPU inference and return softmax class probabilities.

Hardware path  : Camera → HLS Kernel (PL) → DPU (PL) → ARM post-process
Software path  : Camera → OpenCV resize  → NumPy norm → Mock inference

Usage
-----
    runner = DPUInferenceRunner("models/classroom_model.xmodel")
    probs  = runner.infer(bgr_frame)
    top5   = runner.top_k(probs, k=5)
"""

import logging
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger("DPUInferenceRunner")

# ---------------------------------------------------------------------------
# Constants matching preprocessing.h
# ---------------------------------------------------------------------------
DPU_INPUT_H     = 224
DPU_INPUT_W     = 224
DPU_INPUT_C     = 3
DPU_QUANT_SCALE = 64.0   # INT8 Q6 fixed-point scale


# ---------------------------------------------------------------------------
# Soft fallback: pure-NumPy preprocessing (CPU path)
# ---------------------------------------------------------------------------
def _sw_preprocess(bgr_frame: np.ndarray) -> np.ndarray:
    """
    Mirrors the HLS kernel logic in software for dev/testing.
    Returns an INT8 array of shape (1, DST_H, DST_W, 3).
    """
    resized = cv2.resize(bgr_frame, (DPU_INPUT_W, DPU_INPUT_H),
                         interpolation=cv2.INTER_LINEAR)
    # BGR → RGB (most DPU models expect RGB)
    rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    # Normalize to [-1.0, 1.0]
    norm    = rgb.astype(np.float32) / 127.5 - 1.0
    # Quantize to INT8
    quantized = np.clip(norm * DPU_QUANT_SCALE, -128, 127).astype(np.int8)
    return quantized[np.newaxis, ...]   # add batch dimension


# ---------------------------------------------------------------------------
# DPUInferenceRunner
# ---------------------------------------------------------------------------
class DPUInferenceRunner:
    """
    High-level wrapper around the VART RunnerExt API.

    Parameters
    ----------
    xmodel_path : str | Path
        Path to the compiled .xmodel file produced by Vitis AI quantiser.
    use_hw_preprocess : bool
        If True, use the XRT-managed HLS preprocessing kernel via PyXRT.
        Falls back to software if the overlay is not available.
    labels_path : str | Path | None
        Optional path to a newline-separated class label file.
    """

    def __init__(
        self,
        xmodel_path: str | Path,
        use_hw_preprocess: bool = True,
        labels_path: Optional[str | Path] = None,
    ):
        self.xmodel_path      = Path(xmodel_path)
        self.use_hw_preprocess = use_hw_preprocess
        self.labels: list[str] = []

        if labels_path:
            self.labels = Path(labels_path).read_text().strip().splitlines()

        self._runner    = None   # vart.RunnerExt handle
        self._hw_kernel = None   # PyXRT handle for HLS preprocessing

        self._load_dpu_runner()
        if use_hw_preprocess:
            self._load_hw_preprocess_kernel()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def _load_dpu_runner(self):
        """Load the DPU subgraph runner using VART."""
        try:
            import vart                          # Vitis AI Runtime
            import xir                           # XIR graph API

            graph     = xir.Graph.deserialize(str(self.xmodel_path))
            subgraphs = graph.get_root_subgraph().toposort_child_subgraph()

            # Select the first subgraph assigned to the DPU
            dpu_subgraphs = [
                sg for sg in subgraphs
                if sg.has_attr("device") and sg.get_attr("device").upper() == "DPU"
            ]
            if not dpu_subgraphs:
                raise RuntimeError("No DPU subgraph found in xmodel.")

            # VART runner — equivalent to runner.load_subgraph(dpu_subgraphs[0])
            self._runner = vart.Runner.create_runner(dpu_subgraphs[0], "run")
            log.info("✅  DPU runner loaded from %s", self.xmodel_path)

            # Cache input / output tensor shapes
            self._in_tensors  = self._runner.get_input_tensors()
            self._out_tensors = self._runner.get_output_tensors()
            log.info(
                "    Input : %s  |  Output: %s",
                [t.dims for t in self._in_tensors],
                [t.dims for t in self._out_tensors],
            )

        except ImportError:
            log.warning(
                "⚠️  VART/XIR not installed — DPU runner disabled (SW mock active)."
            )
            self._runner = None

    def _load_hw_preprocess_kernel(self):
        """Attach to the HLS preprocessing kernel via PyXRT."""
        try:
            import pyxrt                        # AMD XRT Python binding

            device        = pyxrt.device(0)
            xclbin_path   = os.environ.get("PREPROCESS_XCLBIN", "overlay/preprocess.xclbin")
            xclbin        = pyxrt.xclbin(xclbin_path)
            device.register_xclbin(xclbin)
            ctx           = pyxrt.hw_context(device, xclbin.get_uuid())
            self._hw_kernel = pyxrt.kernel(ctx, "preprocess_image")
            log.info("✅  HLS preprocessing kernel loaded from %s", xclbin_path)

        except Exception as exc:
            log.warning(
                "⚠️  HLS kernel unavailable (%s) — falling back to SW preprocess.", exc
            )
            self._hw_kernel = None

    # ------------------------------------------------------------------
    # Preprocessing dispatch
    # ------------------------------------------------------------------
    def _preprocess(self, bgr_frame: np.ndarray) -> np.ndarray:
        """Return an INT8 tensor (1, H, W, 3) ready for the DPU."""
        if self._hw_kernel is not None:
            return self._hw_preprocess(bgr_frame)
        return _sw_preprocess(bgr_frame)

    def _hw_preprocess(self, bgr_frame: np.ndarray) -> np.ndarray:
        """
        Execute the HLS preprocessing kernel via XRT managed buffers.
        Src buffer → kernel → dst buffer, then wrap dst as a NumPy array.
        """
        import pyxrt

        src_flat = bgr_frame.flatten().astype(np.uint8)
        src_buf  = pyxrt.bo(
            self._hw_kernel.get_device(),
            src_flat.nbytes,
            pyxrt.bo.normal,
            self._hw_kernel.group_id(0),
        )
        dst_size = DPU_INPUT_H * DPU_INPUT_W * DPU_INPUT_C
        dst_buf  = pyxrt.bo(
            self._hw_kernel.get_device(),
            dst_size,
            pyxrt.bo.normal,
            self._hw_kernel.group_id(1),
        )

        src_buf.write(src_flat.tobytes(), 0)
        src_buf.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

        run = self._hw_kernel(src_buf, dst_buf)
        run.wait()   # blocks until kernel completes (typically < 1 ms)

        dst_buf.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        result = np.frombuffer(dst_buf.read(dst_size, 0), dtype=np.int8)
        return result.reshape(1, DPU_INPUT_H, DPU_INPUT_W, DPU_INPUT_C)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def infer(self, bgr_frame: np.ndarray) -> np.ndarray:
        """
        End-to-end inference: BGR frame → class probability array.

        Parameters
        ----------
        bgr_frame : np.ndarray  shape (H, W, 3)  dtype uint8

        Returns
        -------
        np.ndarray  shape (num_classes,)  dtype float32  — softmax probabilities
        """
        input_tensor = self._preprocess(bgr_frame)

        if self._runner is None:
            return self._mock_inference(input_tensor)

        # Allocate output buffer matching the DPU output tensor shape
        out_shape   = tuple(self._out_tensors[0].dims)
        output_data = np.zeros(out_shape, dtype=np.int8)

        # Execute DPU — VART async execute_async + wait pattern
        job_id = self._runner.execute_async([input_tensor], [output_data])
        self._runner.wait(job_id)

        # Dequantize INT8 output and apply softmax
        output_f32 = output_data.astype(np.float32)
        return self._softmax(output_f32.flatten())

    # ------------------------------------------------------------------
    # Post-processing helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        exp = np.exp(logits - logits.max())
        return exp / exp.sum()

    @staticmethod
    def _mock_inference(input_tensor: np.ndarray) -> np.ndarray:
        """Deterministic mock — returns uniform distribution over 1000 classes."""
        log.debug("Mock inference called (DPU not available).")
        probs = np.ones(1000, dtype=np.float32) / 1000.0
        return probs

    def top_k(self, probs: np.ndarray, k: int = 5) -> list[dict]:
        """
        Returns the top-k predictions as a list of dicts:
        [{"label": str, "confidence": float}, ...]
        """
        indices = np.argsort(probs)[::-1][:k]
        results = []
        for idx in indices:
            label = self.labels[idx] if idx < len(self.labels) else f"class_{idx}"
            results.append({"label": label, "confidence": float(probs[idx])})
        return results

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        """Release VART and XRT resources."""
        if self._runner:
            del self._runner
            self._runner = None
        if self._hw_kernel:
            del self._hw_kernel
            self._hw_kernel = None
        log.info("DPU runner closed.")


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    xmodel = sys.argv[1] if len(sys.argv) > 1 else "models/demo.xmodel"

    # Simulate a camera frame
    fake_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

    with DPUInferenceRunner(xmodel, use_hw_preprocess=False) as runner:
        probs = runner.infer(fake_frame)
        top5  = runner.top_k(probs, k=5)
        print("\nTop-5 predictions:")
        for i, pred in enumerate(top5, 1):
            print(f"  {i}. {pred['label']:<30} {pred['confidence']*100:5.2f}%")