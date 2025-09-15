import math
import torch
import torch.nn.functional as F
import base64
import json
import hashlib
from pathlib import Path
from collections import OrderedDict

import gradio as gr
import numpy as np
from PIL import Image, ImageFilter

from modules import scripts, shared, processing, sd_schedulers, sd_samplers, sd_samplers_common, script_callbacks, rng
from contextlib import contextmanager, nullcontext
from modules import images, devices, prompt_parser, sd_models, extra_networks
from typing import Optional

# Ensure we call the upscaler path when resizing (A1111 compat)
# --- Compatibility notes ---
# Recommended runtime:
# - Python 3.10–3.11
# - PyTorch 2.1+ (fallback for F.interpolate antialias is handled)
# - Gradio 4.x
# - Automatic1111 WebUI >= 1.9
# - k-diffusion (optional; a minimal stub is used if absent)
# - kornia (optional)

RESIZE_WITH_UPSCALER = getattr(images, "RESIZE_WITH_UPSCALER", None)
if RESIZE_WITH_UPSCALER is None:
    print("[Custom Hires Fix] Warning: images.RESIZE_WITH_UPSCALER not found; using fallback value 2. Please update Automatic1111 WebUI to a recent version.")
    RESIZE_WITH_UPSCALER = 2

# Optional deps (best-effort)
def _safe_import(modname, pipname=None):
    try:
        __import__(modname)
        return True
    except Exception as e:
        print(f"[Custom Hires Fix] Warning: {e}")
        return False

_safe_import("omegaconf")
_safe_import("kornia")
_safe_import("k_diffusion", "k-diffusion")
_safe_import("skimage")
_safe_import("cv2")

try:
    from omegaconf import OmegaConf, DictConfig  # type: ignore
except Exception:  # graceful fallback if OmegaConf not available
    class DictConfig(dict):  # minimal stub
        pass
    class OmegaConf:  # minimal stub
        @staticmethod
        def load(path):
            return DictConfig()
        @staticmethod
        def create(obj):
            return DictConfig(obj)
try:
    import kornia  # type: ignore
    _HAS_KORNIA = True
except Exception as e:
    print(f"[Custom Hires Fix] Warning: {e}")
    kornia = None
    _HAS_KORNIA = False

try:
    import k_diffusion as K  # type: ignore
    _HAS_KDIFF = True
except Exception as e:
    print(f"[Custom Hires Fix] Warning: {e}")
    _HAS_KDIFF = False

    class _KStub:
        class sampling:
            @staticmethod
            def get_sigmas_karras(n, sigma_min, sigma_max, rho=7.0, device=None):
                """
                Простейшая замена: логарифмическое пространство от sigma_max до sigma_min
                + финальный нулевой шаг, как ожидает WebUI.
                """
                import math
                import torch
                n = int(n)
                sigma_min = float(sigma_min)
                sigma_max = float(sigma_max)
                if n <= 0:
                    return torch.tensor([0.0], dtype=torch.float32, device=device)
                if sigma_max <= 0 or sigma_min <= 0:
                    sigma_max, sigma_min = 1.0, 0.01
                sig = torch.logspace(
                    math.log10(sigma_max),
                    math.log10(sigma_min),
                    steps=n,
                    device=device,
                    dtype=torch.float32,
                )
                return torch.cat([sig, torch.tensor([0.0], device=device, dtype=torch.float32)], dim=0)

            @staticmethod
            def get_sigmas_exponential(n, sigma_min, sigma_max, device=None):
                """
                Экспоненциальный (геометрический) ряд между sigma_max и sigma_min + 0.0.
                """
                import torch
                n = int(n)
                sigma_min = float(sigma_min)
                sigma_max = float(sigma_max)
                if n <= 0:
                    return torch.tensor([0.0], dtype=torch.float32, device=device)
                if sigma_max <= 0 or sigma_min <= 0:
                    sigma_max, sigma_min = 1.0, 0.01
                r = (sigma_min / sigma_max) ** (1.0 / max(1, n - 1))
                vals = [sigma_max * (r ** i) for i in range(n)]
                sig = torch.tensor(vals, dtype=torch.float32, device=device)
                return torch.cat([sig, torch.tensor([0.0], device=device, dtype=torch.float32)], dim=0)

            @staticmethod
            def get_sigmas_polyexponential(n, sigma_min, sigma_max, rho=0.5, device=None):
                """
                Упрощённая polyexponential: используем karras-подобное лог-пространство,
                параметр rho для совместимости (на форму не влияет в этой заглушке).
                """
                return _KStub.sampling.get_sigmas_karras(n, sigma_min, sigma_max, rho=rho, device=device)

    K = _KStub()
# skimage helpers (optional)

# ---- Latent resampling helpers (with anti-alias) ----
def _parse_upscale_method(upscale_method: str):
    """
    Поддерживает 'bilinear-antialiased' / 'bicubic-antialiased' (флаг сглаживания)
    + обычные 'nearest'/'bilinear'/'bicubic'.
    """
    if upscale_method in ("bilinear-antialiased", "bicubic-antialiased"):
        base = upscale_method.split("-")[0]
        return base, True
    return upscale_method, False

def _interpolate_latent(tensor: torch.Tensor, size_hw: tuple[int, int], mode_name: str) -> torch.Tensor:
    """
    Унифицированная интерполяция латента с безопасным использованием antialias и align_corners.
    """
    mode, aa = _parse_upscale_method(mode_name)
    kwargs = {"mode": mode}
    if mode in ("bilinear", "bicubic"):
        kwargs["align_corners"] = False
    try:
        # PyTorch >= 2.0: есть antialias
        return F.interpolate(tensor, size=size_hw, antialias=aa, **kwargs)
    except TypeError:
        # Старые версии — без параметра antialias
        return F.interpolate(tensor, size=size_hw, **kwargs)

try:
    from skimage.exposure import match_histograms, equalize_adapthist  # type: ignore
    from skimage import color as skcolor  # type: ignore
    _SKIMAGE_OK = True
except Exception as e:
    print(f"[Custom Hires Fix] Warning: {e}")
    _SKIMAGE_OK = False

# OpenCV (optional)
try:
    import cv2  # type: ignore
    _CV2_OK = True
except Exception as e:
    print(f"[Custom Hires Fix] Warning: {e}")
    _CV2_OK = False

QUOTE_SINGLE_TO_DOUBLE = {ord("'"): ord('"')}
config_path = (Path(__file__).parent.resolve() / "../config.yaml").resolve()


class CustomHiresFix(scripts.Script):
    def _get_free_vram_bytes(self):
        """Return free VRAM bytes if available; else None."""
        try:
            import torch
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                return int(free)
        except Exception:
            pass
        return None

    def _interp(self, *args, _default_mode: str = "bicubic", **kwargs):
        """Config-driven interpolate with runtime fallbacks."""
        import torch.nn.functional as F
        if self._latent_resample_enabled():
            label = str(self.config.get("latent_resample_mode", _default_mode))
            mode, antialias = self._parse_interpolate_mode(label)
            kwargs["mode"] = mode
            # Set align_corners only for linear/bilinear/bicubic/trilinear families
            if mode in {"linear", "bilinear", "bicubic", "trilinear"}:
                kwargs.setdefault("align_corners", False)
            else:
                kwargs.pop("align_corners", None)
            # antialias valid only for linear/bilinear/bicubic/trilinear
            if mode in {"linear", "bilinear", "bicubic", "trilinear"}:
                kwargs["antialias"] = antialias
            else:
                kwargs.pop("antialias", None)
        try:
            return F.interpolate(*args, **kwargs)
        except Exception as e:
            # 1) PyTorch<2.0: попробуем без antialias
            if "antialias" in kwargs:
                aa = kwargs.pop("antialias")
                try:
                    return F.interpolate(*args, **kwargs)
                except Exception:
                    kwargs["antialias"] = aa  # восстановим для следующих веток

            # 2) Старые torch: 'nearest-exact' не поддерживается
            if kwargs.get("mode") == "nearest-exact":
                try:
                    kwargs["mode"] = "nearest"
                    kwargs.pop("antialias", None)
                    return F.interpolate(*args, **kwargs)
                except Exception:
                    pass

            # 3) Жёсткий фолбэк: форсируем 'nearest' без align_corners/antialias
            try:
                kwargs["mode"] = "nearest"
                kwargs.pop("antialias", None)
                kwargs.pop("align_corners", None)
                return F.interpolate(*args, **kwargs)
            except Exception:
                pass

            print(f"[Custom Hires Fix] _interp fallback: {e}")
            try:
                return args[0]  # мягкий фолбэк без ресайза
            except Exception:
                return None


    """Two-stage img2img upscaling with optional latent mixing and prompt overrides.

    Features:
    - Ratio/width/height or Megapixels target (+ quick MP buttons)
    - Compact preset panel (global presets)
    - Separate steps for 1st/2nd pass
    - Per-pass sampler + scheduler
    - CFG base + optional delta on 2nd pass
    - Reuse seed/noise on 2nd pass
    - Conditioning cache (LRU) with capacity
    - Second-pass prompt (append/replace)
    - Per-pass LoRA weight scaling
    - Seamless tiling (+ overlap)
    - VAE tiling toggle (low VRAM)
    - Color match to original (strength) with presets
    - Post-FX presets: CLAHE (local contrast), Unsharp Mask
    - PNG-info serialization + paste support
    - Final ×4 upscale (optional), with its own upscaler and tiling overlap for seam-safe stitching.
    """
    def __init__(self):
        super().__init__()
        # Load or init config
        if config_path.exists():
            try:
                self.config: DictConfig = OmegaConf.load(str(config_path)) or OmegaConf.create({})  # type: ignore
            except Exception as e:
                print(f"[Custom Hires Fix] Warning: {e}")
                self.config = OmegaConf.create({})  # type: ignore
        else:
            self.config = OmegaConf.create({})  # type: ignore

        # Runtime state
        self.p = None
        self.pp = None
        self.cfg = 0.0
        self.cond = None
        self.uncond = None
        self.width = None
        self.height = None
        self._orig_clip_skip = None
        self._cn_units = []
        self._use_cn = False

        # Reuse state
        self._saved_seeds = None
        # subseeds may not exist in all pipelines; keep best-effort
        self._saved_subseeds = None
        self._saved_subseed_strength = None
        self._saved_seed_resize_from_h = None
        self._saved_seed_resize_from_w = None
        self._first_noise = None
        self._first_noise_shape = None

        # Conditioning cache (LRU)
        self._cond_cache: OrderedDict[str, tuple] = OrderedDict()

        # VAE tiling state restore
        self._orig_opt_vae_tiling = None

        # Seamless tiling restore
        self._orig_tiling = None
        self._orig_tile_overlap = None

        # Prompt override for second pass
        self._override_prompt_second = None

        # LoRA scaling factor per pass (used during _prepare_conditioning by pass context)
        self._current_lora_factor = 1.0
        # Scheduler restore state
        self._orig_scheduler = None
        self._orig_size = (None, None)


        # NEW: флаг семейства модели (SDXL/SD3) и LRU-кэш шума
        self.is_sdxl = False
        # Небольшой LRU на тензоры шума: ключ (pass|shape|seed) -> Tensor
        # Важно держать размер маленьким, чтобы не проедать VRAM
        self._noise_cache: OrderedDict[str, torch.Tensor] = OrderedDict()

        # Cache fingerprint
        self._last_cache_fp = None

    def _get_model_fingerprint(self) -> str | None:
        """Build a fingerprint of current model/checkpoint + clip_skip to detect changes."""
        parts = []
        try:
            sd_model = getattr(shared, "sd_model", None)
            if sd_model is not None:
                for attr in ("sd_model_hash", "model_hash", "hash", "ckpt_hash"):
                    val = getattr(sd_model, attr, None)
                    if val:
                        parts.append(str(val))
                ci = getattr(sd_model, "sd_checkpoint_info", None)
                if ci is not None:
                    for attr in ("model_hash", "hash", "name", "title", "filename"):
                        val = getattr(ci, attr, None)
                        if val:
                            parts.append(str(val))
        except Exception as e:
            print(f"[Custom Hires Fix] fingerprint build warning: {e}")
        try:
            cs = getattr(getattr(shared, "opts", None), "CLIP_stop_at_last_layers", None)
            if cs is not None:
                parts.append(f"clip_skip={cs}")
        except Exception as e:
            print(f"[Custom Hires Fix] fingerprint clip_skip warning: {e}")
        if not parts:
            return None
        try:
            import hashlib as _hl
            return _hl.md5("|".join(parts).encode("utf-8")).hexdigest()
        except Exception as e:
            print(f"[Custom Hires Fix] Warning: {e}")
            return "|".join(parts)

    def _clear_runtime_caches(self):
        """Clear LRU caches safely."""
        try:
            if hasattr(self, "_cond_cache"):
                self._cond_cache.clear()
            if hasattr(self, "_noise_cache"):
                self._noise_cache.clear()
            print("[Custom Hires Fix] Runtime caches cleared.")
        except Exception as e:
            print(f"[Custom Hires Fix] cache clear warning: {e}")

    def _maybe_reset_caches(self):
        """Reset caches on model change."""
        fp = self._get_model_fingerprint()
        if fp is None:
            return
        if getattr(self, "_last_cache_fp", None) != fp:
            self._clear_runtime_caches()
            self._last_cache_fp = fp
            print("[Custom Hires Fix] Model change detected -> caches reset.")



    def _apply_token_merging(self, *, for_hr: bool = False, halve: bool = False):
        """Safely apply token merging ratio across webui versions."""
        ratio_fn = getattr(self.p, "get_token_merging_ratio", None)
        r: float = 0.0
        if callable(ratio_fn):
            try:
                r = float(ratio_fn(for_hr=for_hr))
            except TypeError:
                r = float(ratio_fn())
            except Exception as e:
                print(f"[Custom Hires Fix] Warning: {e}")
                r = 0.0
        if halve:
            r = r / 2.0
        try:
            # используем p.sd_model если есть, иначе shared.sd_model
            model = getattr(self.p, "sd_model", None) or shared.sd_model
            sd_models.apply_token_merging(model, r)
        except Exception as e:
            print(f"[Custom Hires Fix] Warning: {e}")
            pass

    def _set_scheduler_by_label(self, label_or_obj):
        """
        Всегда устанавливаем p.scheduler как СТРОКУ (label).
        Это совместимо с путём k-diffusion A1111, где ожидается строковый ключ
        для sd_schedulers.schedulers_map.get(...). На новых ветках это тоже безопасно.
        """
        if not label_or_obj or label_or_obj == "Use same scheduler":
            return

        if isinstance(label_or_obj, str):
            label = label_or_obj
        else:
            label = getattr(label_or_obj, "label", getattr(label_or_obj, "name", str(label_or_obj)))

        # Всегда строка:
        try:
            self.p.scheduler = label
        except Exception as e:
            print(f"[Custom Hires Fix] Warning: {e}")
            try:
                setattr(self.p, "scheduler", label)
            except Exception as e:
                print(f"[Custom Hires Fix] Warning: {e}")
                pass
    def _coerce_scheduler_to_string(self):
        """
        Если p.scheduler по каким-то причинам оказался объектом,
        приводим к строке (label|name), чтобы sd_samplers_kdiffusion не падал.
        """
        try:
            sch = getattr(self.p, "scheduler", None)
            if sch is not None and not isinstance(sch, str):
                label = getattr(sch, "label", getattr(sch, "name", None))
                if isinstance(label, str) and label:
                    self.p.scheduler = label
        except Exception as e:
            print(f"[Custom Hires Fix] Warning: {e}")
            pass


    # ---- A1111 Script API ----
    def title(self):
        return "Custom Hires Fix"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        visible_names = [x.name for x in sd_samplers.visible_samplers()]
        sampler_names = ["Restart + DPM++ 3M SDE"] + visible_names
        _scheds = getattr(sd_schedulers, "schedulers", [])
        scheduler_names = ["Use same scheduler"] + [
            getattr(x, "label", getattr(x, "name", str(x))) for x in _scheds]

        with gr.Accordion(label="Custom Hires Fix", open=False) as enable_box:
            enable = gr.Checkbox(label="Enable extension", value=bool(self.config.get("enable", False)))

            # ---------- Compact preset panel ----------
            with gr.Row():
                quick_preset = gr.Dropdown(
                    ["None", "Hi-Res Portrait", "Hi-Res Texture", "Hi-Res Illustration", "Hi-Res Product Shot"],
                    label="Quick preset",
                    value="None"
                )
                btn_apply_preset = gr.Button(value="Apply preset", variant="primary")

                btn_mp_1 = gr.Button(value="MP 1.0")
                btn_mp_2 = gr.Button(value="MP 2.0")
                btn_mp_4 = gr.Button(value="MP 4.0")
                btn_mp_8 = gr.Button(value="MP 8.0")

            with gr.Row():
                ratio = gr.Slider(minimum=0.0, maximum=4.0, step=0.05, label="Upscale by (ratio)",
                                  value=float(self.config.get("ratio", 0.0)))
                width = gr.Slider(minimum=0, maximum=4096, step=8, label="Resize width to",
                                  value=int(self.config.get("width", 0)))
                height = gr.Slider(minimum=0, maximum=4096, step=8, label="Resize height to",
                                   value=int(self.config.get("height", 0)))
            # --------- Size helpers ---------
            with gr.Row():
                long_edge = gr.Slider(minimum=0, maximum=8192, step=8,
                                      label="Resize by long edge (0 = off)",
                                      value=int(self.config.get("long_edge", 0)))
                btn_swap_wh = gr.Button(value="Swap W↔H")

            # NEW: кастомный размер для 2-го прохода
            with gr.Row():
                second_custom_size_enable = gr.Checkbox(
                    label="Second pass: override size",
                    value=bool(self.config.get("second_custom_size_enable", False))
                )
                second_width = gr.Slider(minimum=0, maximum=4096, step=8,
                                         label="Second pass width",
                                         value=int(self.config.get("second_width", 0)))
                second_height = gr.Slider(minimum=0, maximum=4096, step=8,
                                          label="Second pass height",
                                          value=int(self.config.get("second_height", 0)))

            with gr.Row():
                steps_first = gr.Slider(minimum=1, maximum=100, step=1, label="Hires steps — 1st pass",
                                        value=int(self.config.get("steps_first", max(1, int(self.config.get("steps", 20))))))
                steps_second = gr.Slider(minimum=1, maximum=100, step=1, label="Hires steps — 2nd pass",
                                         value=int(self.config.get("steps_second", int(self.config.get("steps", 20)))))

            # --------- Per-pass denoising ---------
            with gr.Row():
                denoise_first = gr.Slider(minimum=0.0, maximum=1.0, step=0.01,
                                          label="Denoising strength — 1st pass",
                                          value=float(self.config.get("denoise_first", 0.33)))
                denoise_second = gr.Slider(minimum=0.0, maximum=1.0, step=0.01,
                                           label="Denoising strength — 2nd pass",
                                           value=float(self.config.get("denoise_second", 0.45)))
            with gr.Row():
                first_upscaler = gr.Dropdown([x.name for x in shared.sd_upscalers],
                                             label="First upscaler", value=self.config.get("first_upscaler", "R-ESRGAN 4x+"))
                second_upscaler = gr.Dropdown([x.name for x in shared.sd_upscalers],
                                              label="Second upscaler", value=self.config.get("second_upscaler", "R-ESRGAN 4x+"))

            with gr.Row():
                first_latent = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="Latent mix (first stage)",
                                         value=float(self.config.get("first_latent", 0.3)))
                second_latent = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="Latent mix (second stage)",
                                          value=float(self.config.get("second_latent", 0.1)))
            # рядом с латент-слайдами
            with gr.Row():
                            first_latent_invert = gr.Checkbox(
                                            label="Invert 1st-pass latent mix (slider = original image latent weight)",
                                            value=bool(self.config.get("first_latent_invert", False))
                            )  # NEW

            # Режим ресемпла латента
            with gr.Row():
                latent_resample_enable = gr.Checkbox(label="Latent resample enable", value=bool(self.config.get("latent_resample_enable", True)))
                latent_resample_mode = gr.Dropdown(["Disabled", "nearest", "nearest-exact", "bilinear", "bicubic", "area", "bilinear-antialiased", "bicubic-antialiased"],
                    label="Latent resample mode",
                    value=self.config.get("latent_resample_mode", "nearest")
                )

            with gr.Row():
                filter_mode = gr.Dropdown(["Noise sync (sharp)", "Morphological (smooth)", "Combined (balanced)"],
                                          label="Filter mode", value=self.config.get("filter_mode", "Noise sync (sharp)"))
                strength = gr.Slider(minimum=0.5, maximum=4.0, step=0.1, label="Generation strength",
                                     value=float(self.config.get("strength", 2.0)))
                denoise_offset = gr.Slider(minimum=-0.1, maximum=0.2, step=0.01, label="Denoise offset",
                                           value=float(self.config.get("denoise_offset", 0.05)))
                # NEW: чекбокс включения адаптивной формы сигм
                adaptive_sigma_enable = gr.Checkbox(label="Adaptive denoiser shaping (uses Filter/Strength)",
                                                    value=bool(self.config.get("adaptive_sigma_enable", False)))

            with gr.Row():
                prompt = gr.Textbox(label="Prompt override (1st pass)", placeholder="Leave empty to use main UI prompt",
                                    value=self.config.get("prompt", ""))
                negative_prompt = gr.Textbox(label="Negative prompt override", placeholder="Leave empty to use main UI negative prompt",
                                             value=self.config.get("negative_prompt", ""))

            with gr.Row():
                second_pass_prompt = gr.Textbox(label="Second-pass prompt", placeholder="Append or replace on 2nd pass",
                                                value=self.config.get("second_pass_prompt", ""))
                second_pass_prompt_append = gr.Checkbox(label="Append instead of replace",
                                                        value=bool(self.config.get("second_pass_prompt_append", True)))

            with gr.Accordion(label="Extra", open=False):
                with gr.Row():
                    filter_offset = gr.Slider(minimum=-1.0, maximum=1.0, step=0.1, label="Filter offset",
                                              value=float(self.config.get("filter_offset", 0.0)))
                    clip_skip = gr.Slider(minimum=0, maximum=12, step=1, label="CLIP skip (0 = keep)",
                                          value=int(self.config.get("clip_skip", 0)))

                # Режим расписания сигм
                with gr.Row():
                    noise_schedule_mode = gr.Dropdown(
                        [
                            "Use sampler default", "Adaptive (filter/strength)", "Karras", "Exponential",
                            "Polyexponential", "DDIM uniform", "Normal", "Simple"
                        ],
                        label="Noise schedule override",
                        value=self.config.get("noise_schedule_mode", "Use sampler default"))

                # Per-pass sampler/scheduler
                with gr.Row():
                    sampler_first = gr.Dropdown(sampler_names, label="Sampler — 1st pass",
                                                value=self.config.get("sampler_first", sampler_names[0]))
                    sampler_second = gr.Dropdown(sampler_names, label="Sampler — 2nd pass",
                                                 value=self.config.get("sampler_second", self.config.get("sampler", sampler_names[0])))
                with gr.Row():
                    scheduler_first = gr.Dropdown(
                        choices=scheduler_names, label="Schedule type — 1st pass",
                        value=self.config.get("scheduler_first", self.config.get("scheduler", scheduler_names[0]))
                    )
                    scheduler_second = gr.Dropdown(
                        choices=scheduler_names, label="Schedule type — 2nd pass",
                        value=self.config.get("scheduler_second", self.config.get("scheduler", scheduler_names[0]))
                    )

                # Restore scheduler toggle
                restore_scheduler_after = gr.Checkbox(
                    label="Restore scheduler after run",
                    value=bool(self.config.get("restore_scheduler_after", True))
                )
 
                with gr.Row():
                    cfg = gr.Slider(minimum=0, maximum=30, step=0.5, label="CFG Scale (base)",
                                    value=float(self.config.get("cfg", 7.0)))
                    cfg_second_pass_boost = gr.Checkbox(label="Enable CFG delta on 2nd pass",
                                                        value=bool(self.config.get("cfg_second_pass_boost", True)))
                    cfg_second_pass_delta = gr.Slider(minimum=-5.0, maximum=5.0, step=0.5, label="CFG delta (2nd pass)",
                                                      value=float(self.config.get("cfg_second_pass_delta", 3.0)))

                # Reuse seed/noise + Megapixels target
                with gr.Row():
                    reuse_seed_noise = gr.Checkbox(label="Reuse seed/noise on 2nd pass",
                                                   value=bool(self.config.get("reuse_seed_noise", False)))
                    # ВКЛ/ВЫКЛ кэш шума по ключу (pass|sampler|scheduler|shape|seed)
                    reuse_noise_cache = gr.Checkbox(
                        label="Noise cache (reuse by key)",
                        value=bool(self.config.get("reuse_noise_cache", True))
                    )


                    mp_target_enabled = gr.Checkbox(label="Enable Megapixels target",
                                                    value=bool(self.config.get("mp_target_enabled", False)))
                    mp_target = gr.Slider(minimum=0.3, maximum=16.0, step=0.1, label="Megapixels",
                                          value=float(self.config.get("mp_target", 2.0)))

                # Conditioning cache controls
                with gr.Row():
                    cond_cache_enabled = gr.Checkbox(label="Enable conditioning cache (LRU)",
                                                     value=bool(self.config.get("cond_cache_enabled", True)))
                    cond_cache_max = gr.Slider(minimum=8, maximum=256, step=8, label="Conditioning cache size",
                                               value=int(self.config.get("cond_cache_max", 64)))


                # Noise cache controls (NEW)
                # Отдельный контрол для лимита LRU-кэша шума (CPU/host memory).
                # Ставим небольшой верхний предел, чтобы не проедать память при больших сессиях.
                with gr.Row():
                    noise_cache_max = gr.Slider(
                        minimum=4, maximum=64, step=1,
                        label="Noise cache size",
                        value=int(self.config.get("noise_cache_max", 16)))

                # VAE tiling
                with gr.Row():
                    vae_tiling_enabled = gr.Checkbox(label="Enable VAE tiling (low VRAM)",
                                                     value=bool(self.config.get("vae_tiling_enabled", False)))

                # Seamless tiling
                with gr.Row():
                    seamless_tiling_enabled = gr.Checkbox(label="Seamless tiling (texture)",
                                                          value=bool(self.config.get("seamless_tiling_enabled", False)))
                    tile_overlap = gr.Slider(minimum=0, maximum=64, step=1, label="Tile overlap (px)",
                                             value=int(self.config.get("tile_overlap", 12)))

                # LoRA scaling
                with gr.Row():
                    lora_weight_first_factor = gr.Slider(minimum=0.0, maximum=2.0, step=0.05, label="LoRA weight × (1st pass)",
                                                         value=float(self.config.get("lora_weight_first_factor", 1.0)))
                    lora_weight_second_factor = gr.Slider(minimum=0.0, maximum=2.0, step=0.05, label="LoRA weight × (2nd pass)",
                                                          value=float(self.config.get("lora_weight_second_factor", 1.0)))

                # Match colors presets & controls
                with gr.Row():
                    match_colors_preset = gr.Dropdown(
                        ["Off", "Subtle (0.3)", "Natural (0.5)", "Strong (0.8)"],
                        label="Match colors preset",
                        value=self.config.get("match_colors_preset", "Off")
                    )
                    match_colors_enabled = gr.Checkbox(label="Match colors to original",
                                                       value=bool(self.config.get("match_colors_enabled", False)))
                    match_colors_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, label="Match strength",
                                                      value=float(self.config.get("match_colors_strength", 0.5)))

                # Post-processing presets & controls
                with gr.Row():
                    postfx_preset = gr.Dropdown(
                        ["Off", "Soft clarity", "Portrait safe", "Texture boost", "Crisp detail"],
                        label="Post-FX preset",
                        value=self.config.get("postfx_preset", "Off")
                    )
                    clahe_enabled = gr.Checkbox(label="CLAHE (local contrast)",
                                                value=bool(self.config.get("clahe_enabled", False)))
                    clahe_clip = gr.Slider(minimum=1.0, maximum=5.0, step=0.1, label="CLAHE clip limit",
                                           value=float(self.config.get("clahe_clip", 2.0)))
                    clahe_tile_grid = gr.Slider(minimum=4, maximum=16, step=2, label="CLAHE tile grid",
                                                value=int(self.config.get("clahe_tile_grid", 8)))

                with gr.Row():
                    unsharp_enabled = gr.Checkbox(label="Unsharp Mask (sharpen)",
                                                  value=bool(self.config.get("unsharp_enabled", False)))
                    unsharp_radius = gr.Slider(minimum=0.5, maximum=5.0, step=0.1, label="Unsharp radius",
                                               value=float(self.config.get("unsharp_radius", 1.5)))
                    unsharp_amount = gr.Slider(minimum=0.0, maximum=2.0, step=0.05, label="Unsharp amount",
                                               value=float(self.config.get("unsharp_amount", 0.75)))
                    unsharp_threshold = gr.Slider(minimum=0, maximum=10, step=1, label="Unsharp threshold",
                                                  value=int(self.config.get("unsharp_threshold", 0)))

                with gr.Row():
                    cn_ref = gr.Checkbox(label="Use last image as ControlNet reference", value=bool(self.config.get("cn_ref", False)))
                    start_control_at = gr.Slider(minimum=0.0, maximum=0.7, step=0.01, label="CN start (enabled units)",
                                                 value=float(self.config.get("start_control_at", 0.0)))
                    cn_proc_res_cap = gr.Slider(minimum=256, maximum=2048, step=64,
                                                label="ControlNet processor_res cap",
                                                value=int(self.config.get("cn_proc_res_cap", 1024)))

                # --- Final upscale controls (configurable scale) ---
                with gr.Row():
                                final_upscale_enable = gr.Checkbox(
                                                label="Final upscale (after 2nd pass)",
                                                value=bool(self.config.get("final_upscale_enable", False))
                                )
                                final_upscaler = gr.Dropdown(
                                                [x.name for x in shared.sd_upscalers],
                                                label="Final upscaler",
                                                value=self.config.get("final_upscaler", "R-ESRGAN 4x+")
                                )
                                # NEW: настраиваемый масштаб
                                final_scale = gr.Slider(
                                                minimum=1.0, maximum=8.0, step=0.5,
                                                label="Final upscale scale (×)",
                                                value=float(self.config.get("final_scale", 4.0))
                                )

                with gr.Row():
                                final_tile = gr.Slider(
                                                minimum=128, maximum=2048, step=32,
                                                label="Final tile size (px, pre-scale)",
                                                value=int(self.config.get("final_tile", 512))
                                )
                                final_tile_overlap = gr.Slider(
                                                minimum=0, maximum=256, step=2,
                                                label="Final tile overlap (px, pre-scale)",
                                                value=int(self.config.get("final_tile_overlap", 16))
                                )

                # NEW: дополнительные переключатели поведения
                with gr.Row():
                    legacy_cfg_handling = gr.Checkbox(
                        label="Legacy CFG handling (treat 0 as unset)",
                        value=bool(self.config.get("legacy_cfg_handling", False))
                    )
                    cleanup_noise_overrides = gr.Checkbox(
                        label="Cleanup noise-scheduler overrides",
                        value=bool(self.config.get("cleanup_noise_overrides", True))
                    )

                # --- Stop-at-step controls ---
                with gr.Row():
                    stop_mode = gr.Dropdown(
                        ["Off", "Clamp steps (built-in)", "Hook & interrupt (Anti-Burn style)"],
                        label="Stop-at-step mode",
                        value=self.config.get("stop_mode", "Off")
                    )
                with gr.Row():
                    stop_first_at = gr.Slider(0, 100, step=1,
                                              label="Stop 1st pass at step (0 = off)",
                                              value=int(self.config.get("stop_first_at", 0)))
                    stop_second_at = gr.Slider(0, 100, step=1,
                                               label="Stop 2nd pass at step (0 = off)",
                                               value=int(self.config.get("stop_second_at", 0)))

            # NEW: Anti-twinning (latent deep shrink) и SDXL-подсказки
            with gr.Row():
                deep_shrink_enable = gr.Checkbox(
                    label="Enable Deep Shrink (anti-twinning)",
                    value=bool(self.config.get("deep_shrink_enable", False))
                )
                deep_shrink_strength = gr.Slider(
                    minimum=0.1, maximum=0.9, step=0.05,
                    label="Shrink strength",
                    value=float(self.config.get("deep_shrink_strength", 0.5)),
                    visible=bool(self.config.get("deep_shrink_enable", False))
                 )
            # toggle visibility
            deep_shrink_enable.change(
                fn=lambda v: gr.update(visible=bool(v)),
                inputs=deep_shrink_enable,
                outputs=deep_shrink_strength
            )

            with gr.Row():
                sdxl_mode = gr.Checkbox(
                    label="SDXL/SD3 mode (gentle denoise boost)",
                    value=bool(self.config.get("sdxl_mode", False))
                )
                sdxl_denoise_boost = gr.Slider(
                    minimum=0.0, maximum=0.9, step=0.01,
                    label="SDXL denoise boost (+)",
                    value=float(self.config.get("sdxl_denoise_boost", 0.1))
                )

            # ---------- Preset logic (UI events) ----------
            def _apply_match_preset(preset_name):
                if preset_name == "Off":
                    return (gr.update(value=False), gr.update(value=0.5))
                if preset_name == "Subtle (0.3)":
                    return (gr.update(value=True), gr.update(value=0.3))
                if preset_name == "Natural (0.5)":
                    return (gr.update(value=True), gr.update(value=0.5))
                if preset_name == "Strong (0.8)":
                    return (gr.update(value=True), gr.update(value=0.8))
                return (gr.update(), gr.update())

            def _apply_postfx_preset(preset_name):
                # Returns: clahe_enabled, clahe_clip, clahe_tile_grid, unsharp_enabled, unsharp_radius, unsharp_amount, unsharp_threshold
                if preset_name == "Off":
                    return (gr.update(value=False), gr.update(value=2.0), gr.update(value=8),
                            gr.update(value=False), gr.update(value=1.5), gr.update(value=0.75), gr.update(value=0))
                if preset_name == "Soft clarity":
                    return (gr.update(value=True), gr.update(value=1.8), gr.update(value=8),
                            gr.update(value=True), gr.update(value=1.2), gr.update(value=0.6), gr.update(value=0))
                if preset_name == "Portrait safe":
                    return (gr.update(value=True), gr.update(value=1.6), gr.update(value=8),
                            gr.update(value=True), gr.update(value=1.4), gr.update(value=0.8), gr.update(value=2))
                if preset_name == "Texture boost":
                    return (gr.update(value=True), gr.update(value=2.4), gr.update(value=8),
                            gr.update(value=True), gr.update(value=1.6), gr.update(value=1.0), gr.update(value=0))
                if preset_name == "Crisp detail":
                    return (gr.update(value=True), gr.update(value=2.1), gr.update(value=8),
                            gr.update(value=True), gr.update(value=1.3), gr.update(value=0.9), gr.update(value=0))
                return (gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update())

            def _apply_quick_preset(name):
                # Returns a large tuple of updates for several controls
                out = [
                    gr.update(),  # steps_first
                    gr.update(),  # steps_second
                    gr.update(),  # cfg_second_pass_boost
                    gr.update(),  # cfg_second_pass_delta
                    gr.update(),  # sampler_first
                    gr.update(),  # sampler_second
                    gr.update(),  # scheduler_first
                    gr.update(),  # scheduler_second
                    gr.update(),  # vae_tiling_enabled
                    gr.update(),  # seamless_tiling_enabled
                    gr.update(),  # tile_overlap
                    gr.update(),  # match_colors_preset
                    gr.update(),  # match_colors_enabled
                    gr.update(),  # match_colors_strength
                    gr.update(),  # postfx_preset
                    gr.update(),  # clahe_enabled
                    gr.update(),  # clahe_clip
                    gr.update(),  # clahe_tile_grid
                    gr.update(),  # unsharp_enabled
                    gr.update(),  # unsharp_radius
                    gr.update(),  # unsharp_amount
                    gr.update(),  # unsharp_threshold
                    gr.update(),  # reuse_seed_noise
                    gr.update(),  # cond_cache_max
                    gr.update(),  # lora_weight_first_factor
                    gr.update(),  # lora_weight_second_factor
                    gr.update(),  # mp_target_enabled
                    gr.update(),  # mp_target
                ]
                if name == "Hi-Res Portrait":
                    out = [
                        gr.update(value=18), gr.update(value=28),
                        gr.update(value=True), gr.update(value=2.5),
                        gr.update(value="DPM++ 2M Karras"), gr.update(value="DPM++ 3M SDE"),
                        gr.update(value="Use same scheduler"), gr.update(value="Use same scheduler"),
                        gr.update(value=True), gr.update(value=False), gr.update(value=12),
                        gr.update(value="Subtle (0.3)"), gr.update(value=True), gr.update(value=0.3),
                        gr.update(value="Portrait safe"), gr.update(value=True), gr.update(value=1.6), gr.update(value=8),
                        gr.update(value=True), gr.update(value=1.4), gr.update(value=0.8), gr.update(value=2),
                        gr.update(value=True), gr.update(value=64),
                        gr.update(value=1.0), gr.update(value=1.1),
                        gr.update(value=True), gr.update(value=2.0),
                    ]
                elif name == "Hi-Res Texture":
                    out = [
                        gr.update(value=14), gr.update(value=22),
                        gr.update(value=True), gr.update(value=3.0),
                        gr.update(value="DPM++ 2M Karras"), gr.update(value="DPM++ SDE Karras"),
                        gr.update(value="Use same scheduler"), gr.update(value="Use same scheduler"),
                        gr.update(value=True), gr.update(value=True), gr.update(value=12),
                        gr.update(value="Off"), gr.update(value=False), gr.update(value=0.5),
                        gr.update(value="Texture boost"), gr.update(value=True), gr.update(value=2.2), gr.update(value=8),
                        gr.update(value=True), gr.update(value=1.6), gr.update(value=1.0), gr.update(value=0),
                        gr.update(value=True), gr.update(value=128),
                        gr.update(value=0.9), gr.update(value=1.25),
                        gr.update(value=True), gr.update(value=4.0),
                    ]
                elif name == "Hi-Res Illustration":
                    out = [
                        gr.update(value=16), gr.update(value=24),
                        gr.update(value=True), gr.update(value=2.0),
                        gr.update(value="DPM++ 2M Karras"), gr.update(value="DPM++ 3M SDE"),
                        gr.update(value="Use same scheduler"), gr.update(value="Use same scheduler"),
                        gr.update(value=True), gr.update(value=False), gr.update(value=8),
                        gr.update(value="Off"), gr.update(value=False), gr.update(value=0.5),
                        gr.update(value="Crisp detail"), gr.update(value=True), gr.update(value=2.0), gr.update(value=8),
                        gr.update(value=True), gr.update(value=1.2), gr.update(value=0.9), gr.update(value=0),
                        gr.update(value=True), gr.update(value=64),
                        gr.update(value=0.85), gr.update(value=1.2),
                        gr.update(value=True), gr.update(value=2.0),
                    ]
                elif name == "Hi-Res Product Shot":
                    out = [
                        gr.update(value=18), gr.update(value=26),
                        gr.update(value=True), gr.update(value=2.0),
                        gr.update(value="DPM++ 2M Karras"), gr.update(value="DPM++ SDE Karras"),
                        gr.update(value="Use same scheduler"), gr.update(value="Use same scheduler"),
                        gr.update(value=True), gr.update(value=False), gr.update(value=8),
                        gr.update(value="Natural (0.5)"), gr.update(value=True), gr.update(value=0.5),
                        gr.update(value="Crisp detail"), gr.update(value=True), gr.update(value=2.1), gr.update(value=8),
                        gr.update(value=True), gr.update(value=1.3), gr.update(value=0.9), gr.update(value=0),
                        gr.update(value=True), gr.update(value=96),
                        gr.update(value=1.0), gr.update(value=1.15),
                        gr.update(value=True), gr.update(value=2.0),
                    ]
                return tuple(out)
                
            # --- Привязка событий пресетов ---
            match_colors_preset.change(
                fn=_apply_match_preset,
                inputs=[match_colors_preset],
                outputs=[match_colors_enabled, match_colors_strength]
            )
            postfx_preset.change(
                fn=_apply_postfx_preset,
                inputs=[postfx_preset],
                outputs=[clahe_enabled, clahe_clip, clahe_tile_grid, unsharp_enabled, unsharp_radius, unsharp_amount, unsharp_threshold]
            )
            btn_apply_preset.click(
                fn=_apply_quick_preset,
                inputs=[quick_preset],
                outputs=[
                    steps_first, steps_second,
                    cfg_second_pass_boost, cfg_second_pass_delta,
                    sampler_first, sampler_second,
                    scheduler_first, scheduler_second,
                    vae_tiling_enabled, seamless_tiling_enabled, tile_overlap,
                    match_colors_preset, match_colors_enabled, match_colors_strength,
                    postfx_preset, clahe_enabled, clahe_clip, clahe_tile_grid,
                    unsharp_enabled, unsharp_radius, unsharp_amount, unsharp_threshold,
                    reuse_seed_noise, cond_cache_max,
                    lora_weight_first_factor, lora_weight_second_factor,
                    mp_target_enabled, mp_target
                ]
            )
            # MP buttons
            btn_mp_1.click(fn=lambda: (gr.update(value=True), gr.update(value=1.0)), inputs=[], outputs=[mp_target_enabled, mp_target])
            btn_mp_2.click(fn=lambda: (gr.update(value=True), gr.update(value=2.0)), inputs=[], outputs=[mp_target_enabled, mp_target])
            btn_mp_4.click(fn=lambda: (gr.update(value=True), gr.update(value=4.0)), inputs=[], outputs=[mp_target_enabled, mp_target])
            btn_mp_8.click(fn=lambda: (gr.update(value=True), gr.update(value=8.0)), inputs=[], outputs=[mp_target_enabled, mp_target])

        # Exclusivity helpers (STRICT two-mode: exact W×H OR pure ratio)
        # Изменение любой стороны отключает ratio, НО не обнуляет вторую сторону — можно задать обе.
        width.change(fn=lambda _: (gr.update(value=0.0), gr.update(value=False)),
                     inputs=width, outputs=[ratio, mp_target_enabled])
        height.change(fn=lambda _: (gr.update(value=0.0), gr.update(value=False)),
                      inputs=height, outputs=[ratio, mp_target_enabled])
        ratio.change(fn=lambda _: (gr.update(value=0), gr.update(value=0), gr.update(value=False)),
                     inputs=ratio, outputs=[width, height, mp_target_enabled])
        # Long edge excludes ratio and explicit W/H
        long_edge.change(fn=lambda _: (gr.update(value=0), gr.update(value=0), gr.update(value=0.0), gr.update(value=False)),
                         inputs=long_edge, outputs=[width, height, ratio, mp_target_enabled])
        # Swap button
        btn_swap_wh.click(fn=lambda w, h: (h, w), inputs=[width, height], outputs=[width, height])

        # infotext paste support
        def read_params(d, key, default=None):
            try:
                return d["Custom Hires Fix"].get(key, default)
            except Exception as e:
                print(f"[Custom Hires Fix] Warning: {e}")
                return default

        self.infotext_fields = [
            (enable, lambda d: "Custom Hires Fix" in d),
            (ratio, lambda d: read_params(d, "ratio", 0.0)),
            (width, lambda d: read_params(d, "width", 0)),
            (height, lambda d: read_params(d, "height", 0)),
            (long_edge, lambda d: read_params(d, "long_edge", 0)),
            (steps_first, lambda d: read_params(d, "steps_first", read_params(d, "steps", 20))),
            (steps_second, lambda d: read_params(d, "steps_second", read_params(d, "steps", 20))),
            (denoise_first, lambda d: read_params(d, "denoise_first", 0.33)),
            (denoise_second, lambda d: read_params(d, "denoise_second", 0.45)),
            (first_upscaler, lambda d: read_params(d, "first_upscaler")),
            (second_upscaler, lambda d: read_params(d, "second_upscaler")),
            (first_latent, lambda d: read_params(d, "first_latent", 0.0)),
            (second_latent, lambda d: read_params(d, "second_latent", 0.0)),
            (latent_resample_mode, lambda d: read_params(d, "latent_resample_mode", "nearest")),
            (latent_resample_enable, lambda d: read_params(d, "latent_resample_enable", True)),
            
            (prompt, lambda d: read_params(d, "prompt", "")),
            (negative_prompt, lambda d: read_params(d, "negative_prompt", "")),
            (second_pass_prompt, lambda d: read_params(d, "second_pass_prompt", "")),
            (second_pass_prompt_append, lambda d: read_params(d, "second_pass_prompt_append", True)),
            (strength, lambda d: read_params(d, "strength", 0.0)),
            (filter_mode, lambda d: read_params(d, "filter_mode")),
            (denoise_offset, lambda d: read_params(d, "denoise_offset", 0.0)),
            (filter_offset, lambda d: read_params(d, "filter_offset", 0.0)),
            (adaptive_sigma_enable, lambda d: read_params(d, "adaptive_sigma_enable", False)),
            (noise_schedule_mode, lambda d: read_params(d, "noise_schedule_mode", "Use sampler default")),
            (clip_skip, lambda d: read_params(d, "clip_skip", 0)),
            # per-pass samplers/schedulers + legacy fallbacks
            (sampler_first, lambda d: read_params(d, "sampler_first", read_params(d, "sampler", sampler_names[0]))),
            (sampler_second, lambda d: read_params(d, "sampler_second", read_params(d, "sampler", sampler_names[0]))),
            (scheduler_first, lambda d: read_params(d, "scheduler_first", read_params(d, "scheduler", scheduler_names[0]))),
            (scheduler_second, lambda d: read_params(d, "scheduler_second", read_params(d, "scheduler", scheduler_names[0]))),
            (restore_scheduler_after, lambda d: read_params(d, "restore_scheduler_after", True)),
            # cfg/delta
            (cfg, lambda d: read_params(d, "cfg", 7.0)),
            (cfg_second_pass_boost, lambda d: read_params(d, "cfg_second_pass_boost", True)),
            (cfg_second_pass_delta, lambda d: read_params(d, "cfg_second_pass_delta", 3.0)),
            # flags
            (reuse_seed_noise, lambda d: read_params(d, "reuse_seed_noise", False)),
            (reuse_noise_cache, lambda d: read_params(d, "reuse_noise_cache", True)),
            (mp_target_enabled, lambda d: read_params(d, "mp_target_enabled", False)),
            (mp_target, lambda d: read_params(d, "mp_target", 2.0)),
            (cond_cache_enabled, lambda d: read_params(d, "cond_cache_enabled", True)),
            (cond_cache_max, lambda d: read_params(d, "cond_cache_max", 64)),
            (noise_cache_max, lambda d: read_params(d, "noise_cache_max", 16)),
            (vae_tiling_enabled, lambda d: read_params(d, "vae_tiling_enabled", False)),
            (seamless_tiling_enabled, lambda d: read_params(d, "seamless_tiling_enabled", False)),
            (tile_overlap, lambda d: read_params(d, "tile_overlap", 12)),
            (lora_weight_first_factor, lambda d: read_params(d, "lora_weight_first_factor", 1.0)),
            (lora_weight_second_factor, lambda d: read_params(d, "lora_weight_second_factor", 1.0)),
            (match_colors_preset, lambda d: read_params(d, "match_colors_preset", "Off")),
            (match_colors_enabled, lambda d: read_params(d, "match_colors_enabled", False)),
            (match_colors_strength, lambda d: read_params(d, "match_colors_strength", 0.5)),
            (postfx_preset, lambda d: read_params(d, "postfx_preset", "Off")),
            (clahe_enabled, lambda d: read_params(d, "clahe_enabled", False)),
            (clahe_clip, lambda d: read_params(d, "clahe_clip", 2.0)),
            (clahe_tile_grid, lambda d: read_params(d, "clahe_tile_grid", 8)),
            (unsharp_enabled, lambda d: read_params(d, "unsharp_enabled", False)),
            (unsharp_radius, lambda d: read_params(d, "unsharp_radius", 1.5)),
            (unsharp_amount, lambda d: read_params(d, "unsharp_amount", 0.75)),
            (unsharp_threshold, lambda d: read_params(d, "unsharp_threshold", 0)),
            (cn_ref, lambda d: read_params(d, "cn_ref", False)),
            (start_control_at, lambda d: read_params(d, "start_control_at", 0.0)),
            (cn_proc_res_cap, lambda d: read_params(d, "cn_proc_res_cap", 1024)),
            # final upscale
            (final_upscale_enable, lambda d: read_params(d, "final_upscale_enable", False)),
            (final_upscaler,      lambda d: read_params(d, "final_upscaler", "R-ESRGAN 4x+")),
            (final_scale,         lambda d: read_params(d, "final_scale", 4.0)),        # NEW
            (final_tile,          lambda d: read_params(d, "final_tile", 512)),
            (final_tile_overlap,  lambda d: read_params(d, "final_tile_overlap", 16)),
            # NEW
            (legacy_cfg_handling,     lambda d: read_params(d, "legacy_cfg_handling", False)),
            (cleanup_noise_overrides, lambda d: read_params(d, "cleanup_noise_overrides", True)),
            (deep_shrink_enable,   lambda d: read_params(d, "deep_shrink_enable", False)),
            (deep_shrink_strength, lambda d: read_params(d, "deep_shrink_strength", 0.5)),
            (sdxl_mode,            lambda d: read_params(d, "sdxl_mode", False)),
            (sdxl_denoise_boost,   lambda d: read_params(d, "sdxl_denoise_boost", 0.1)),
            (first_latent_invert,        lambda d: read_params(d, "first_latent_invert", False)),
            (second_custom_size_enable,  lambda d: read_params(d, "second_custom_size_enable", False)),
            (second_width,               lambda d: read_params(d, "second_width", 0)),
            (second_height,              lambda d: read_params(d, "second_height", 0)),
            # Stop-at-step paste support
            (stop_mode,                  lambda d: read_params(d, "stop_mode", "Off")),
            (stop_first_at,              lambda d: read_params(d, "stop_first_at", 0)),
            (stop_second_at,             lambda d: read_params(d, "stop_second_at", 0)),
 
        ]

        return [
            enable, quick_preset,
            ratio, width, height, long_edge,
            steps_first, steps_second, denoise_first, denoise_second,
            first_upscaler, second_upscaler, first_latent, second_latent,
            latent_resample_mode,
            latent_resample_enable,
            prompt, negative_prompt, second_pass_prompt, second_pass_prompt_append,
            strength, filter_mode, filter_offset, denoise_offset, adaptive_sigma_enable,
            noise_schedule_mode,
            sampler_first, sampler_second, scheduler_first, scheduler_second,
            restore_scheduler_after,
            cfg, cfg_second_pass_boost, cfg_second_pass_delta,
            reuse_seed_noise, reuse_noise_cache, mp_target_enabled, mp_target,
            cond_cache_enabled, cond_cache_max,
            vae_tiling_enabled,
            seamless_tiling_enabled, tile_overlap,
            lora_weight_first_factor, lora_weight_second_factor,
            match_colors_preset, match_colors_enabled, match_colors_strength,
            postfx_preset, clahe_enabled, clahe_clip, clahe_tile_grid,
            unsharp_enabled, unsharp_radius, unsharp_amount, unsharp_threshold,
            cn_ref, start_control_at, cn_proc_res_cap,
            # final upscale
            final_upscale_enable, final_upscaler, final_scale, final_tile, final_tile_overlap,
            # NEW (UI → сигнатура, добавлены элементы ниже)
            deep_shrink_enable, deep_shrink_strength,
            sdxl_mode, sdxl_denoise_boost,
            # существующие NEW
            first_latent_invert,
            second_custom_size_enable, second_width, second_height,
            # NEW (возврат контролов галочек)
            legacy_cfg_handling,
            cleanup_noise_overrides,
            # Stop-at-step controls
            stop_mode, stop_first_at, stop_second_at,
            # NEW: noise cache control
            noise_cache_max
        ]

    # Capture base processing object and optional ControlNet state
    
    def _parse_interpolate_mode(self, label: str) -> tuple[str, bool]:
        """
        Преобразует подпись из UI в аргументы torch.nn.functional.interpolate:
        (mode, antialias). Поддерживает: "nearest", "nearest-exact", "bilinear",
        "bicubic", "area", а также "bilinear-antialiased"/"bicubic-antialiased".
        Для старых torch, где "nearest-exact" недоступен, _interp уже содержит фолбэк.
        """
        try:
            lab = str(label or "").strip().lower()
            if lab.endswith("-antialiased"):
                base = lab.replace("-antialiased", "")
                if base in ("bilinear", "bicubic"):
                    return base, True
            if lab in ("nearest", "nearest-exact", "bilinear", "bicubic", "area"):
                return lab, False
            return "bicubic", True
        except Exception:
            return "bicubic", True

    def _latent_resample_enabled(self) -> bool:
        """
        Включено ли ресемплирование латента согласно конфигу.
        Отключается, если выбран режим 'Disabled'.
        """
        mode = str(self.config.get("latent_resample_mode", "bicubic")).lower()
        return bool(self.config.get("latent_resample_enable", True)) and mode != "disabled"


    def process(self, p, *args, **kwargs):
        self.p = p
        # Reset caches if model/clip_skip changed
        self._maybe_reset_caches()
        # Запомним исходные W/H, если есть
        self._orig_size = (getattr(p, "width", None), getattr(p, "height", None))
        self._cn_units = []
        self._use_cn = False
        self._first_noise = None
        self._first_noise_shape = None
        self._saved_seeds = None
        self._saved_subseeds = None
        self._saved_subseed_strength = None
        self._saved_seed_resize_from_h = None
        self._saved_seed_resize_from_w = None
        self._override_prompt_second = None

        # Try detect ControlNet (best-effort; path may vary across installs)
        ext_candidates = [
            "extensions.sd_webui_controlnet.scripts.external_code",
            "extensions.sd-webui-controlnet.scripts.external_code",
            "extensions-builtin.sd-webui-controlnet.scripts.external_code",
        ]
        self._cn_ext = None
        for mod in ext_candidates:
            try:
                self._cn_ext = __import__(mod, fromlist=["external_code"])
                break
            except Exception as e:
                print(f"[Custom Hires Fix] Warning: {e}")
                continue
        if self._cn_ext:
            try:
                units = self._cn_ext.get_all_units_in_processing(p)
                self._cn_units = list(units) if units else []
                self._use_cn = len(self._cn_units) > 0
            except Exception as e:
                print(f"[Custom Hires Fix] Warning: {e}")
                self._use_cn = False
            # Post-detection notice
            try:
                cn_flags = []
                for k, v in getattr(self, "config", {}).items():
                    if isinstance(k, str) and k.startswith("cn_") and bool(v):
                        cn_flags.append(k)
                if (cn_flags or bool(self.config.get("cn_ref", False))) and not self._use_cn:
                    print("[Custom Hires Fix] ControlNet not detected, but CN-related options are set:", cn_flags or ["cn_ref"])
            except Exception as e:
                print(f"[Custom Hires Fix] ControlNet warning logic failed: {e}")


    # Log settings into PNG-info (single JSON block)
    def before_process_batch(self, p, *args, **kwargs):
        if not bool(self.config.get("enable", False)):
            return
        # сохраняем информ-блок (json64, чтобы не ломать кавычки)
        p.extra_generation_params["Custom Hires Fix"] = self.create_infotext(p)

    def create_infotext(self, p, *args, **kwargs):
        scale_val = 0
        if int(self.config.get("width", 0)) and int(self.config.get("height", 0)):
            scale_val = f"{int(self.config.get('width'))}x{int(self.config.get('height'))}"
        elif float(self.config.get("ratio", 0)):
            scale_val = float(self.config.get("ratio"))

        payload = {
            "scale": scale_val,
            "ratio": float(self.config.get("ratio", 0.0)),
            "width": int(self.config.get("width", 0) or 0),
            "height": int(self.config.get("height", 0) or 0),
            "long_edge": int(self.config.get("long_edge", 0)),
            # NEW
            "second_custom_size_enable": bool(self.config.get("second_custom_size_enable", False)),
            "second_width": int(self.config.get("second_width", 0)),
            "second_height": int(self.config.get("second_height", 0)),
            "denoise_first": float(self.config.get("denoise_first", 0.33)),
            "denoise_second": float(self.config.get("denoise_second", 0.45)),

            "steps_first": int(self.config.get("steps_first", int(self.config.get("steps", 20)))),
            "steps_second": int(self.config.get("steps_second", int(self.config.get("steps", 20)))),
            # legacy aggregate (оставлено для совместимости с paste)
            "steps": int(self.config.get("steps", int(self.config.get("steps_first", 20)))),
            "first_upscaler": self.config.get("first_upscaler", ""),
            "second_upscaler": self.config.get("second_upscaler", ""),
            "first_latent": float(self.config.get("first_latent", 0.3)),
            "first_latent_invert": bool(self.config.get("first_latent_invert", False)),  # NEW
            "second_latent": float(self.config.get("second_latent", 0.1)),
            "prompt": self.config.get("prompt", ""),
            "negative_prompt": self.config.get("negative_prompt", ""),
            "second_pass_prompt": self.config.get("second_pass_prompt", ""),
            "second_pass_prompt_append": bool(self.config.get("second_pass_prompt_append", True)),
            "strength": float(self.config.get("strength", 2.0)),
            "filter_mode": self.config.get("filter_mode", ""),
            "filter_offset": float(self.config.get("filter_offset", 0.0)),
            "denoise_offset": float(self.config.get("denoise_offset", 0.05)),
            "latent_resample_mode": self.config.get("latent_resample_mode", "nearest"),
            "latent_resample_enable": bool(self.config.get("latent_resample_enable", True)),
            "noise_schedule_mode": self.config.get("noise_schedule_mode", "Use sampler default"),
            "adaptive_sigma_enable": bool(self.config.get("adaptive_sigma_enable", False)),
            "clip_skip": int(self.config.get("clip_skip", 0)),
            # per-pass sampler/scheduler (include legacy for context)
            "sampler_first": self.config.get("sampler_first", ""),
            "sampler_second": self.config.get("sampler_second", self.config.get("sampler", "")),
            "scheduler_first": self.config.get("scheduler_first", self.config.get("scheduler", "")),
            "scheduler_second": self.config.get("scheduler_second", self.config.get("scheduler", "")),
            "restore_scheduler_after": bool(self.config.get("restore_scheduler_after", True)),
            # cfg
            "cfg": float(getattr(p, "cfg_scale", self.cfg)),
            "cfg_second_pass_boost": bool(self.config.get("cfg_second_pass_boost", True)),
            "cfg_second_pass_delta": float(self.config.get("cfg_second_pass_delta", 3.0)),
            # flags
            "reuse_seed_noise": bool(self.config.get("reuse_seed_noise", False)),
            "reuse_noise_cache": bool(self.config.get("reuse_noise_cache", True)),
            "mp_target_enabled": bool(self.config.get("mp_target_enabled", False)),
            "mp_target": float(self.config.get("mp_target", 2.0)),
            "cond_cache_enabled": bool(self.config.get("cond_cache_enabled", True)),
            "cond_cache_max": int(self.config.get("cond_cache_max", 64)),
            "noise_cache_max": int(self.config.get("noise_cache_max", 16)),
            "vae_tiling_enabled": bool(self.config.get("vae_tiling_enabled", False)),
            "seamless_tiling_enabled": bool(self.config.get("seamless_tiling_enabled", False)),
            "tile_overlap": int(self.config.get("tile_overlap", 12)),
            "lora_weight_first_factor": float(self.config.get("lora_weight_first_factor", 1.0)),
            "lora_weight_second_factor": float(self.config.get("lora_weight_second_factor", 1.0)),
            "match_colors_preset": self.config.get("match_colors_preset", "Off"),
            "match_colors_enabled": bool(self.config.get("match_colors_enabled", False)),
            "match_colors_strength": float(self.config.get("match_colors_strength", 0.5)),
            "postfx_preset": self.config.get("postfx_preset", "Off"),
            "clahe_enabled": bool(self.config.get("clahe_enabled", False)),
            "clahe_clip": float(self.config.get("clahe_clip", 2.0)),
            "clahe_tile_grid": int(self.config.get("clahe_tile_grid", 8)),
            "unsharp_enabled": bool(self.config.get("unsharp_enabled", False)),
            "unsharp_radius": float(self.config.get("unsharp_radius", 1.5)),
            "unsharp_amount": float(self.config.get("unsharp_amount", 0.75)),
            "unsharp_threshold": int(self.config.get("unsharp_threshold", 0)),
            "cn_ref": bool(self.config.get("cn_ref", False)),
            "start_control_at": float(self.config.get("start_control_at", 0.0)),
            "cn_proc_res_cap": int(self.config.get("cn_proc_res_cap", 1024)),
            # final upscale
            "final_upscale_enable": bool(self.config.get("final_upscale_enable", False)),
            "final_upscaler": self.config.get("final_upscaler", "R-ESRGAN 4x+"),
            "final_scale": float(self.config.get("final_scale", 4.0)), 
            "final_tile": int(self.config.get("final_tile", 512)),
            "final_tile_overlap": int(self.config.get("final_tile_overlap", 16)),
            # NEW: anti-twinning & SDXL
            "legacy_cfg_handling": bool(self.config.get("legacy_cfg_handling", False)),
            "cleanup_noise_overrides": bool(self.config.get("cleanup_noise_overrides", True)),
            "deep_shrink_enable": bool(self.config.get("deep_shrink_enable", False)),
            "deep_shrink_strength": float(self.config.get("deep_shrink_strength", 0.5)),
            "sdxl_mode": bool(self.config.get("sdxl_mode", False)),
            "sdxl_denoise_boost": float(self.config.get("sdxl_denoise_boost", 0.1)),
            # Stop-at-step
            "stop_mode": self.config.get("stop_mode", "Off"),
            "stop_first_at": int(self.config.get("stop_first_at", 0)),
            "stop_second_at": int(self.config.get("stop_second_at", 0)),
            
        }
        # кодируем надёжно — в base64
        payload_str = json.dumps(payload, ensure_ascii=False)
        return "json64:" + base64.b64encode(payload_str.encode("utf-8")).decode("ascii")

    # --- Main postprocess hook ---
    def postprocess_image(self, p, pp,
                          enable, quick_preset,
                          ratio, width, height, long_edge,
                          steps_first, steps_second, denoise_first, denoise_second,
                          first_upscaler, second_upscaler, first_latent, second_latent,
                          latent_resample_mode,
                          latent_resample_enable,
                          prompt, negative_prompt, second_pass_prompt, second_pass_prompt_append,
                          strength, filter_mode, filter_offset, denoise_offset, adaptive_sigma_enable,
                          noise_schedule_mode,
                          sampler_first, sampler_second, scheduler_first, scheduler_second,
                          restore_scheduler_after,
                          cfg, cfg_second_pass_boost, cfg_second_pass_delta,
                          reuse_seed_noise, reuse_noise_cache, mp_target_enabled, mp_target,
                          cond_cache_enabled, cond_cache_max,
                          vae_tiling_enabled,
                          seamless_tiling_enabled, tile_overlap,
                          lora_weight_first_factor, lora_weight_second_factor,
                          match_colors_preset, match_colors_enabled, match_colors_strength,
                          postfx_preset, clahe_enabled, clahe_clip, clahe_tile_grid,
                          unsharp_enabled, unsharp_radius, unsharp_amount, unsharp_threshold,
                          cn_ref, start_control_at, cn_proc_res_cap,
                          final_upscale_enable, final_upscaler, final_scale, final_tile, final_tile_overlap,
                          # NEW из UI
                          deep_shrink_enable, deep_shrink_strength,
                          sdxl_mode, sdxl_denoise_boost,
                          # NEW ↓
                          first_latent_invert,
                          second_custom_size_enable, second_width, second_height,
                          # NEW галочки
                          legacy_cfg_handling, cleanup_noise_overrides,
                          # NEW: stop-at-step
                          stop_mode, stop_first_at, stop_second_at,
                          # NEW
                          noise_cache_max):
        if not enable:
            return

        # Save config chosen in UI
        self.pp = pp
        self.config["enable"] = bool(enable)
        self.config["ratio"] = float(ratio)
        self.config["width"] = int(width)
        self.config["height"] = int(height)
        self.config["long_edge"] = int(long_edge)
        self.config["steps_first"] = int(steps_first)
        self.config["steps_second"] = int(steps_second)
        self.config["denoise_first"] = float(denoise_first)
        self.config["denoise_second"] = float(denoise_second)
        self.config["latent_resample_enable"] = bool(latent_resample_enable)
        self.config["steps"] = int(steps_second)  # legacy aggregate
        self.config["first_upscaler"] = first_upscaler
        self.config["second_upscaler"] = second_upscaler
        self.config["first_latent"] = float(first_latent)
        self.config["second_latent"] = float(second_latent)
        self.config["prompt"] = prompt.strip()
        self.config["negative_prompt"] = negative_prompt.strip()
        self.config["second_pass_prompt"] = second_pass_prompt.strip()
        self.config["second_pass_prompt_append"] = bool(second_pass_prompt_append)
        self.config["strength"] = float(strength)
        self.config["filter_mode"] = filter_mode
        self.config["filter_offset"] = float(filter_offset)
        self.config["denoise_offset"] = float(denoise_offset)
        self.config["latent_resample_mode"] = str(latent_resample_mode)
        self.config["noise_schedule_mode"] = str(noise_schedule_mode)
        self.config["adaptive_sigma_enable"] = bool(adaptive_sigma_enable)
        # per-pass sampler/scheduler
        self.config["sampler_first"] = sampler_first
        self.config["sampler_second"] = sampler_second
        self.config["scheduler_first"] = scheduler_first
        self.config["scheduler_second"] = scheduler_second
        self.config["restore_scheduler_after"] = bool(restore_scheduler_after)
        # cfg/delta
        self.config["cfg"] = float(cfg)
        self.config["cfg_second_pass_boost"] = bool(cfg_second_pass_boost)
        self.config["cfg_second_pass_delta"] = float(cfg_second_pass_delta)
        # flags & extras
        self.config["reuse_seed_noise"] = bool(reuse_seed_noise)
        self.config["reuse_noise_cache"] = bool(reuse_noise_cache)
        self.config["mp_target_enabled"] = bool(mp_target_enabled)
        self.config["mp_target"] = float(mp_target)
        self.config["cond_cache_enabled"] = bool(cond_cache_enabled)
        self.config["cond_cache_max"] = int(cond_cache_max)
        # NEW
        self.config["noise_cache_max"] = int(noise_cache_max)
        self.config["vae_tiling_enabled"] = bool(vae_tiling_enabled)
        self.config["seamless_tiling_enabled"] = bool(seamless_tiling_enabled)
        self.config["tile_overlap"] = int(tile_overlap)
        self.config["lora_weight_first_factor"] = float(lora_weight_first_factor)
        self.config["lora_weight_second_factor"] = float(lora_weight_second_factor)
        self.config["match_colors_preset"] = match_colors_preset
        self.config["match_colors_enabled"] = bool(match_colors_enabled)
        self.config["match_colors_strength"] = float(match_colors_strength)
        self.config["postfx_preset"] = postfx_preset
        self.config["clahe_enabled"] = bool(clahe_enabled)
        self.config["clahe_clip"] = float(clahe_clip)
        self.config["clahe_tile_grid"] = int(clahe_tile_grid)
        self.config["unsharp_enabled"] = bool(unsharp_enabled)
        self.config["unsharp_radius"] = float(unsharp_radius)
        self.config["unsharp_amount"] = float(unsharp_amount)
        self.config["unsharp_threshold"] = int(unsharp_threshold)
        self.config["cn_ref"] = bool(cn_ref)
        self.config["start_control_at"] = float(start_control_at)
        self.config["cn_proc_res_cap"] = int(cn_proc_res_cap)
        # final upscale
        self.config["final_upscale_enable"] = bool(final_upscale_enable)
        self.config["final_upscaler"] = final_upscaler
        self.config["final_scale"] = float(final_scale)
        self.config["final_tile"] = int(final_tile)
        self.config["final_tile_overlap"] = int(final_tile_overlap)
        # NEW
        self.config["first_latent_invert"] = bool(first_latent_invert)
        # Применяем текущий лимит кэша сразу (если пользователь его уменьшил)
        try:
            max_items = int(self.config.get("noise_cache_max", 16))
            while len(self._noise_cache) > max_items:
                self._noise_cache.popitem(last=False)
        except Exception:
            pass

        # NEW (anti-twinning & SDXL)
        self.config["deep_shrink_enable"] = bool(deep_shrink_enable)
        self.config["deep_shrink_strength"] = float(deep_shrink_strength)
        self.config["sdxl_mode"] = bool(sdxl_mode)
        self.config["sdxl_denoise_boost"] = float(sdxl_denoise_boost)

        self.config["second_custom_size_enable"] = bool(second_custom_size_enable)
        self.config["second_width"] = int(second_width)
        self.config["second_height"] = int(second_height)
        # NEW: флаги поведения
        self.config["legacy_cfg_handling"] = bool(legacy_cfg_handling)
        self.config["cleanup_noise_overrides"] = bool(cleanup_noise_overrides)
        # Stop-at-step
        self.config["stop_mode"] = str(stop_mode)
        self.config["stop_first_at"] = int(stop_first_at)
        self.config["stop_second_at"] = int(stop_second_at)
        # NEW: CFG ветвление по галочке
        if bool(self.config.get("legacy_cfg_handling", False)):
            # старое поведение: 0.0 трактуется как «не задано»
            self.cfg = float(cfg) if cfg else float(getattr(p, "cfg_scale", 7.0))
        else:
            # новое поведение: 0.0 сохраняется как есть
            self.cfg = float(cfg) if cfg is not None else float(getattr(p, "cfg_scale", 7.0))
 
        # Глобальная очистка старых override до любых проходов (на случай чужих экстеншенов)
        try:
            if hasattr(self.p, "sampler_noise_scheduler_override") and \
               bool(self.config.get("cleanup_noise_overrides", True)):
                self.p.sampler_noise_scheduler_override = None
        except Exception as _e:
            print(f"[Custom Hires Fix] Warning: {_e}")
        # Обновить PNG-info уже с актуальным self.config
        p.extra_generation_params["Custom Hires Fix"] = self.create_infotext(p)

        # Короткий сводный лог запуска — удобно для багрепортов
        try:
            size_mode = "MP" if self.config["mp_target_enabled"] else ("long_edge" if self.config["long_edge"] else ("ratio" if self.config["ratio"] else "WxH"))
            print(f"[Custom Hires Fix] Run | size={size_mode} | steps={self.config['steps_first']}/{self.config['steps_second']} | "
                  f"samplers={self.config.get('sampler_first')}/{self.config.get('sampler_second')} | "
                  f"schedulers={self.config.get('scheduler_first')}/{self.config.get('scheduler_second')} | "
                  f"final_upscale={self.config['final_upscale_enable']}×{self.config['final_scale']}")
        except Exception: pass

        # Validate sizing:
        # Если MP target выключен и long_edge=0 — строго ДВА режима:
        #   1) точные W×H (ratio обязан быть 0),
        #   2) или чистый ratio>0 при width=height=0.
        if (not self.config["mp_target_enabled"]
            and int(self.config.get("long_edge", 0)) == 0
            and not bool(self.config.get("second_custom_size_enable", False))):
            ok = (
                (int(width) > 0 and int(height) > 0 and float(ratio) == 0.0)
                or
                (int(width) == 0 and int(height) == 0 and float(ratio) > 0.0)
            )
            if not ok:
                # Soft normalization instead of hard error
                w, h, r = int(width), int(height), float(ratio)
                changed = False
                if r > 0.0 and (w > 0 or h > 0):
                    # Ratio mode takes precedence: clear W/H
                    w, h = 0, 0
                    changed = True
                elif r == 0.0 and ((w > 0) ^ (h > 0)):
                    # Only one side given: make it square
                    if w == 0:
                        w = h
                    else:
                        h = w
                    changed = True
                if changed:
                    print("[Custom Hires Fix] Sizing inputs normalized (auto-fix applied).")
                    self.config["width"] = int(w)
                    self.config["height"] = int(h)
                    self.config["ratio"] = float(r)
                    width, height, ratio = w, h, r
                else:
                    # As a last resort, default to ratio mode 1.0
                    self.config["width"] = 0
                    self.config["height"] = 0
                    self.config["ratio"] = 1.0
                    width, height, ratio = 0, 0, 1.0
                    print("[Custom Hires Fix] Sizing inputs defaulted to ratio=1.0 (auto-fix).")

        # Track extras activated during conditioning
        self._activated_extras = []

        # Preserve original batch size
        self._orig_batch_size = getattr(self.p, 'batch_size', None)

        # Apply CLIP skip for the run
        self._orig_clip_skip = shared.opts.CLIP_stop_at_last_layers
        if int(self.config.get("clip_skip", 0)) > 0:
            shared.opts.CLIP_stop_at_last_layers = int(self.config.get("clip_skip", 0))

        # Toggle VAE tiling for the run
        self._set_vae_tiling(self.config["vae_tiling_enabled"])

        # Toggle seamless tiling
        # Save original scheduler (before first/second pass may change it)
        self._orig_scheduler = getattr(self.p, "scheduler", None)

        self._orig_tiling = getattr(self.p, "tiling", None)
        self._orig_tile_overlap = getattr(self.p, "tile_overlap", None)
        if bool(self.config.get("seamless_tiling_enabled", False)):
            try:
                self.p.tiling = True
                if hasattr(self.p, "tile_overlap"):
                    self.p.tile_overlap = int(self.config.get("tile_overlap", 12))
            except Exception as e:
                print(f"[Custom Hires Fix] Warning: {e}")
                pass
        try:
            with devices.autocast():
                shared.state.nextjob()
                x = self._first_pass(pp.image)
                shared.state.nextjob()
                x = self._second_pass(x)
                # Final ×4 upscale (optional)
                if bool(self.config.get("final_upscale_enable", False)):
                    scale_val = float(self.config.get("final_scale", 4.0))
                    if scale_val > 1.0:
                        x = self._final_upscale_tiled(x, scale_val)  # NEW
            # ВАЖНО: пробрасываем результат обратно в WebUI
            pp.image = x

            self._apply_token_merging(for_hr=False)
            # Post-FX chain is inside _second_pass; final upscale is pure upscaler
            # сохранить актуальный конфиг на диск
            self._save_config()
        finally:
            # Restore options
            shared.opts.CLIP_stop_at_last_layers = self._orig_clip_skip
            self._restore_vae_tiling()
            # Restore scheduler if requested (independent of tiling)
            try:
                if bool(self.config.get("restore_scheduler_after", True)) and getattr(self, "_orig_scheduler", None) is not None:
                    self.p.scheduler = self._orig_scheduler
            except Exception as e:
                print(f"[Custom Hires Fix] Warning: {e}")
                pass
            finally:
                self._orig_scheduler = None

            # Restore tiling if it existed
            if self._orig_tiling is not None:
                try:
                    self.p.tiling = self._orig_tiling
                    if hasattr(self.p, "tile_overlap") and self._orig_tile_overlap is not None:
                        self.p.tile_overlap = self._orig_tile_overlap
                except Exception as e:
                    print(f"[Custom Hires Fix] Warning: {e}")
                    pass
            try:
                if getattr(self, "_orig_batch_size", None) is not None:
                    self.p.batch_size = self._orig_batch_size
            except Exception as e:
                print(f"[Custom Hires Fix] Warning: {e}")
                pass
            try:
                # Deactivate any extras we activated during conditioning
                for _extra in getattr(self, "_activated_extras", []) or []:
                    try:
                        extra_networks.deactivate(self.p, _extra)
                    except Exception as e:
                        print(f"[Custom Hires Fix] Warning: {e}")
                        pass
            finally:
                self._activated_extras = []
            try:
                # Сбросить override сигм для последующих шагов
                if hasattr(self.p, "sampler_noise_scheduler_override") and \
                   bool(self.config.get("cleanup_noise_overrides", True)):
                    self.p.sampler_noise_scheduler_override = None
            except Exception as e:
                print(f"[Custom Hires Fix] Warning: {e}")
                pass
            # Восстановить исходные размеры после возможного _enable_controlnet
            try:
                ow, oh = getattr(self, "_orig_size", (None, None))
                if ow is not None:
                    self.p.width = ow
                if oh is not None:
                    self.p.height = oh
            except Exception as e:
                print(f"[Custom Hires Fix] Warning: {e}")
                pass
            self._orig_size = (None, None)

    # ---- Helpers ----
    def _save_config(self):
        """Сохранение конфигурации: YAML (если доступно) или JSON с явными предупреждениями."""
        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"[Custom Hires Fix] Warning: cannot create config dir: {e}")
        saved = False
        try:
            from omegaconf import OmegaConf as _OC
            _OC.save(self.config, str(config_path))
            print(f"[Custom Hires Fix] Config saved to YAML: {config_path}")
            saved = True
        except Exception as e:
            print(f"[Custom Hires Fix] YAML save failed: {e}")
        if not saved:
            try:
                import json as _json
                json_path = config_path.with_suffix(".json")
                try:
                    from omegaconf import OmegaConf as _OC2
                    data = _OC2.to_container(self.config, resolve=True)
                except Exception as e:
                    print(f"[Custom Hires Fix] Warning: {e}")
                    try:
                        data = dict(self.config)
                    except Exception as e:
                        print(f"[Custom Hires Fix] Warning: {e}")
                        data = {}
                with open(json_path, "w", encoding="utf-8") as f:
                    _json.dump(data, f, ensure_ascii=False, indent=2)
                print(f"[Custom Hires Fix] Config saved to JSON: {json_path}")
            except Exception as e:
                print(f"[Custom Hires Fix] Config save failed: {e}")


    def _vae_down_factor(self) -> int:
        """Возвращает фактический коэффициент даунсемпла VAE (по умолчанию 8)."""
        try:
            f = getattr(getattr(shared.sd_model, "first_stage_model", None), "downsample_factor", None)
            return int(f) if f else 8
        except Exception as e:
            print(f"[Custom Hires Fix] Warning: {e}")
            return 8

    # ---- Helpers ----
    def _maybe_mp_resize(self, base_w, base_h, target_mp: float):
        """Compute size from megapixels while keeping aspect ratio; quantize to multiple of 8."""
        aspect = base_w / base_h if base_h else 1.0
        total_px = max(0.01, target_mp) * 1_000_000.0
        w_float = math.sqrt(total_px * aspect)
        h_float = w_float / aspect
        w = max(8, int(round(w_float / 8) * 8))
        h = max(8, int(round(h_float / 8) * 8))
        return w, h

    def _compute_denoise(self, base_key: str) -> float:
        """
        Returns clamped denoising strength in [0,1] as (config[base_key] + denoise_offset).
        Keeps backwards-compatibility with the previous "denoise_offset" knob.
        """
        try:
            base = float(self.config.get(base_key, 0.5))
        except Exception as e:
            print(f"[Custom Hires Fix] Warning: {e}")
            base = 0.5
        try:
            off = float(self.config.get("denoise_offset", 0.0))
        except Exception as e:
            print(f"[Custom Hires Fix] Warning: {e}")
            off = 0.0
        val = max(0.0, min(1.0, base + off))
        return val

    # ───────────── NEW: SDXL/SD3 detection & denoise boost ─────────────
    def _detect_is_sdxl(self) -> bool:
        """Detect SDXL/SD3/XL+; returns bool for SDXL-like and sets self._model_flavor."""
        flavor = "unknown"
        is_xl_like = False
        try:
            sd_model = getattr(shared, "sd_model", None)
            if sd_model is not None:
                # Direct flags first
                for attr in ("is_sdxl", "is_sd_xl", "is_xl", "is_sd3", "is_sd_xl_plus"):
                    val = getattr(sd_model, attr, None)
                    if isinstance(val, bool) and val:
                        if attr in ("is_sd3",):
                            flavor = "sd3"
                            is_xl_like = True
                            break
                        else:
                            flavor = "sdxl"
                            is_xl_like = True
                            break
                # If not set, infer from checkpoint info/name
                if flavor == "unknown":
                    ci = getattr(sd_model, "sd_checkpoint_info", None)
                    text = ""
                    for a in ("model_hash", "hash", "title", "name", "filename"):
                        v = getattr(ci, a, None) if ci is not None else None
                        if v:
                            text += f" {v}".lower()
                    # Heuristics for names in 2025
                    if any(k in text for k in ("sd3", "sd-3", "stable diffusion 3")):
                        flavor = "sd3"; is_xl_like = True
                    elif any(k in text for k in ("sdxl", "sd-xl", "xl-refiner", "refiner-xl")):
                        flavor = "sdxl"; is_xl_like = True
        except Exception as e:
            print(f"[Custom Hires Fix] model detection warning: {e}")
        self._model_flavor = flavor
        return bool(is_xl_like)
    def _maybe_apply_sdxl_denoise_boost(self):
        """
        Мягко увеличивает denoising_strength для SDXL/SD3 (по желанию).
        Контролируется чекбоксом в UI; безопасно для SD1.5/SD2.x.
        """
        try:
            if bool(self.config.get("sdxl_mode", False)):
                boost = float(self.config.get("sdxl_denoise_boost", 0.1))
                self.p.denoising_strength = float(min(1.0, max(0.0, self.p.denoising_strength + boost)))
        except Exception as e:
            print(f"[Custom Hires Fix] Warning: {e}")
            pass
    
    
    def _model_hash_for_cache(self):
        # best-effort model hash + однократный детект SDXL/SD3
        try:
            h = getattr(shared.sd_model, "sd_model_hash", None) or getattr(shared.sd_model, "hash", None)
            self.is_sdxl = self._detect_is_sdxl()
            return h or str(id(shared.sd_model))
        except Exception as e:
            print(f"[Custom Hires Fix] Warning: {e}")
            return str(id(shared.sd_model))
    
    def _cond_key(self, width, height, steps_for_cond, prompt: str, negative: str, clip_skip: int):
        h = hashlib.sha256()
        h.update((prompt or "").encode("utf-8"))
        h.update(b"::")
        h.update((negative or "").encode("utf-8"))
        # NEW: учитываем семейство модели (SDXL/SD3) в ключе кэша
        xl_flag = int(bool(self.config.get("sdxl_mode", False)) or self.is_sdxl)
        key = f"{self._model_hash_for_cache()}|xl={xl_flag}|{width}x{height}|{steps_for_cond}|cs={clip_skip}|{h.hexdigest()}"
        return key
    
    def _cond_cache_get(self, key: str):
        if not bool(self.config.get("cond_cache_enabled", True)):
            return None
        item = self._cond_cache.get(key)
        if item is not None:
            self._cond_cache.move_to_end(key)
        return item
    
    def _cond_cache_put(self, key: str, value: tuple):
        if not bool(self.config.get("cond_cache_enabled", True)):
            return
        self._cond_cache[key] = value
        self._cond_cache.move_to_end(key)
        max_items = int(self.config.get("cond_cache_max", 64))
        while len(self._cond_cache) > max_items:
            self._cond_cache.popitem(last=False)
    
    def _set_vae_tiling(self, enabled: bool):
        # Save original state if we have not yet
        if self._orig_opt_vae_tiling is None and hasattr(shared.opts, "sd_vae_tiling"):
            self._orig_opt_vae_tiling = bool(shared.opts.sd_vae_tiling)
        # Toggle option
        if hasattr(shared.opts, "sd_vae_tiling"):
            shared.opts.sd_vae_tiling = bool(enabled)
        # Try model-level toggle
        vae = getattr(shared.sd_model, "first_stage_model", None)
        if vae is not None:
            try:
                if enabled and hasattr(vae, "enable_tiling"):
                    vae.enable_tiling()
                if not enabled and hasattr(vae, "disable_tiling"):
                    vae.disable_tiling()
            except Exception as e:
                print(f"[Custom Hires Fix] Warning: {e}")
                pass
    
    def _restore_vae_tiling(self):
        if self._orig_opt_vae_tiling is not None and hasattr(shared.opts, "sd_vae_tiling"):
            shared.opts.sd_vae_tiling = self._orig_opt_vae_tiling
        vae = getattr(shared.sd_model, "first_stage_model", None)
        if vae is not None:
            try:
                if self._orig_opt_vae_tiling and hasattr(vae, "enable_tiling"):
                    vae.enable_tiling()
                elif not self._orig_opt_vae_tiling and hasattr(vae, "disable_tiling"):
                    vae.disable_tiling()
            except Exception as e:
                print(f"[Custom Hires Fix] Warning: {e}")
                pass
        self._orig_opt_vae_tiling = None


    # --- safe upscaler wrapper (PIL fallback) ---
    def _resize_with_upscaler(self, img: Image.Image, w: int, h: int, upscaler_name: str) -> Image.Image:
        try:
            return images.resize_image(RESIZE_WITH_UPSCALER, img, int(w), int(h), upscaler_name=upscaler_name)
        except Exception as e:
            print(f"[Custom Hires Fix] resize_image fallback ({upscaler_name}): {e}")
            try:
                lanczos = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.BICUBIC)
                return img.resize((int(w), int(h)), lanczos)
            except Exception:
                return img.resize((int(w), int(h)))

    def _scale_lora_in_prompt(self, text: str, factor: float) -> str:
            """
            Устойчивое масштабирование весов токенов вида <lora:имя:вес> в строке prompt.
            - Поддерживает пробелы: <lora : name : 0.7 >
            - Вес: десятичные числа, знак, экспонента (e.g. -1.2e-1)
            - Если вес отсутствует — добавляет его (= factor)
            - Никакие другие токены/текст не трогаем
            """
            if factor is None or abs(factor - 1.0) < 1e-6 or not isinstance(text, str) or "<lora" not in text:
                return text
    
            import re
            lora_re = re.compile(
                r"""
                <\s*lora\s*:\s*
                (?P<name>[^:>]+?)          # имя (без ':' и '>')
                (?:\s*:\s*
                   (?P<weight>             # необязательный вес
                       [+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?
                   )
                )?
                \s*>
                """,
                re.VERBOSE
            )
    
            def _repl(m: re.Match) -> str:
                name = m.group("name").strip()
                w = m.group("weight")
                if w is None or w == "":
                    new_w = max(0.0, float(factor))
                else:
                    try:
                        cur = float(w)
                    except Exception as e:
                        print(f"[Custom Hires Fix] LoRA weight parse warning for '{name}': {e}")
                        cur = 1.0
                    new_w = max(0.0, cur * float(factor))
                return f"<lora:{name}:{new_w:.4g}>"
    
            try:
                return lora_re.sub(_repl, text)
            except Exception as e:
                print(f"[Custom Hires Fix] LoRA scaling failed: {e}")
                return text
    def _prepare_conditioning(self, width, height, steps_for_cond: int, prompt_override: str = None):
        """Build (cond, uncond) with optional LRU caching and LoRA scaling."""
        base_prompt = self.config.get("prompt", "").strip() or self.p.prompt.strip()
        negative_base = self.config.get("negative_prompt", "").strip() or (getattr(self.p, "negative_prompt", "") or "").strip()
    
        if prompt_override:
            base_prompt = prompt_override.strip()
    
        # Apply LoRA scaling for this pass
        scaled_prompt = self._scale_lora_in_prompt(base_prompt, self._current_lora_factor)
    
        clip_skip = int(self.config.get("clip_skip", 0))
    
        # Cache lookup
        cache_key = self._cond_key(width, height, steps_for_cond, scaled_prompt, negative_base, clip_skip)
        cached = self._cond_cache_get(cache_key)
        if cached is not None:
            self.cond, self.uncond = cached
            return
    
        # Parse extra networks and build cond
        prompt_text = scaled_prompt
        if not getattr(self.p, "disable_extra_networks", False):
            try:
                prompt_text, extra = extra_networks.parse_prompt(prompt_text)
                if extra:
                    extra_networks.activate(self.p, extra)
                    try:
                        self._activated_extras.append(extra)
                    except Exception as e:
                        print(f"[Custom Hires Fix] Warning: {e}")
                        pass
            except Exception as e:
                print(f"[Custom Hires Fix] Warning: {e}")
                pass
    
        if width and height and hasattr(prompt_parser, "SdConditioning"):
            c = prompt_parser.SdConditioning([prompt_text], False, width, height)
            uc = prompt_parser.SdConditioning([negative_base], False, width, height)
        else:
            c, uc = [prompt_text], [negative_base]
    
        try:
            cond = prompt_parser.get_multicond_learned_conditioning(shared.sd_model, c, steps_for_cond)
        except TypeError:
            cond = prompt_parser.get_multicond_learned_conditioning(shared.sd_model, c)
        try:
            uncond = prompt_parser.get_learned_conditioning(shared.sd_model, uc, steps_for_cond)
        except TypeError:
            uncond = prompt_parser.get_learned_conditioning(shared.sd_model, uc)
        self.cond, self.uncond = cond, uncond
    
        # Store in cache
        self._cond_cache_put(cache_key, (cond, uncond))
    
    def _to_sample(self, x_img: Image.Image):
        image = np.array(x_img).astype(np.float32) / 255.0
        image = np.moveaxis(image, 2, 0)
        decoded = torch.from_numpy(image).to(devices.device).to(devices.dtype_vae)
        decoded = 2.0 * decoded - 1.0
        with torch.inference_mode():
            encoded = shared.sd_model.encode_first_stage(decoded.unsqueeze(0))
            sample = shared.sd_model.get_first_stage_encoding(encoded)
        return decoded, sample
    
    def _create_sampler(self, sampler_name: str):
        # Поддержка синтаксиса "Restart + <InnerSampler>"
        if isinstance(sampler_name, str) and sampler_name.startswith("Restart + "):
            inner = sampler_name.replace("Restart + ", "", 1).strip()
            try:
                s = sd_samplers.create_sampler("Restart", shared.sd_model)
                # если у Restart есть поле для внутреннего сэмплера — зададим
                try:
                    setattr(s, "inner_sampler_name", inner)
                except Exception as e:
                    print(f"[Custom Hires Fix] Warning: {e}")
                    pass
                return s
            except Exception as e:
                print(f"[Custom Hires Fix] Warning: {e}")
                print(f"[Custom Hires Fix] Restart sampler not available; falling back to inner sampler: {inner}")
                # fallback: напрямую создать указанный «внутренний» сэмплер
                try:
                    return sd_samplers.create_sampler(inner, shared.sd_model)
                except Exception as e:
                    print(f"[Custom Hires Fix] Warning: {e}")
                    return sd_samplers.create_sampler("DPM++ 2M Karras", shared.sd_model)
        s = sd_samplers.create_sampler(sampler_name, shared.sd_model)
        # Жёсткая очистка возможного чужого override на сэмплере
        try:
            if hasattr(s, "noise_scheduler_override") and \
               bool(self.config.get("cleanup_noise_overrides", True)):
                s.noise_scheduler_override = None
        except Exception as _e:
            print(f"[Custom Hires Fix] Warning: {_e}")
        return s
    
    def _apply_clahe(self, img: Image.Image) -> Image.Image:
        if not bool(self.config.get("clahe_enabled", False)):
            return img
        np_img = np.array(img)
        if _CV2_OK:
            lab = cv2.cvtColor(np_img, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clip = float(self.config.get("clahe_clip", 2.0))
            tiles = int(self.config.get("clahe_tile_grid", 8))
            clahe = cv2.createCLAHE(clipLimit=max(0.1, clip), tileGridSize=(tiles, tiles))
            l2 = clahe.apply(l)
            lab2 = cv2.merge((l2, a, b))
            rgb = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)
            return Image.fromarray(rgb)
        if _SKIMAGE_OK:
            # skimage fallback on L channel in LAB
            lab = skcolor.rgb2lab(np_img / 255.0)
            l = lab[..., 0] / 100.0
            # skimage expects clip_limit ~= [0..1]; map UI [1..5] -> [0.005..0.1]
            clip = float(self.config.get("clahe_clip", 2.0))
            tiles = int(self.config.get("clahe_tile_grid", 8))
            clip_ski = max(0.005, min(0.1, clip / 20.0))
            l2 = equalize_adapthist(l, clip_limit=clip_ski, kernel_size=(tiles, tiles))
            lab[..., 0] = np.clip(l2 * 100.0, 0, 100.0)
            rgb = skcolor.lab2rgb(lab)
            rgb8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
            return Image.fromarray(rgb8)
        # No-op fallback
        return img
    
    def _apply_unsharp(self, img: Image.Image) -> Image.Image:
        if not bool(self.config.get("unsharp_enabled", False)):
            return img
        radius = float(self.config.get("unsharp_radius", 1.5))
        amount = float(self.config.get("unsharp_amount", 0.75))
        threshold = int(self.config.get("unsharp_threshold", 0))
        return img.filter(ImageFilter.UnsharpMask(radius=radius, percent=int(amount * 100), threshold=threshold))
    
    def _apply_match_colors(self, img: Image.Image, ref: Image.Image) -> Image.Image:
        if not bool(self.config.get("match_colors_enabled", False)):
            return img
        strength = float(self.config.get("match_colors_strength", 0.5))
        strength = max(0.0, min(1.0, strength))
        if strength <= 0.0:
            return img
    
        arr = np.array(img).astype(np.float32)
        ref_arr = np.array(ref).astype(np.float32)
    
        matched = None
        if _SKIMAGE_OK:
            try:
                matched = match_histograms(arr, ref_arr, channel_axis=-1).astype(np.float32)
            except TypeError:
                # older skimage
                matched = match_histograms(arr, ref_arr, multichannel=True).astype(np.float32)
        else:
            # simple mean-std per channel fallback
            eps = 1e-6
            for c in range(arr.shape[2]):
                src = arr[..., c]
                dst = ref_arr[..., c]
                src_m, src_s = src.mean(), src.std() + eps
                dst_m, dst_s = dst.mean(), dst.std() + eps
                arr[..., c] = np.clip((src - src_m) * (dst_s / src_s) + dst_m, 0, 255)
            matched = arr
    
        out = (1.0 - strength) * arr + strength * matched
        out = np.clip(out, 0, 255).astype(np.uint8)
        return Image.fromarray(out)
    
    # ----- Sigma schedule builder (multiple modes) -----
    def _build_sigma_override(self, which_pass: str, schedule_mode: str):
        """
        Возвращает callable: f(n_steps) -> Tensor(sigmas) ИЛИ None (оставить дефолт сэмплера).
        which_pass: "first" | "second"
        schedule_mode: "Use sampler default" | "Adaptive (filter/strength)" | fixed family.
        """
        # 0) Ничего не переопределяем
        if not schedule_mode or schedule_mode == "Use sampler default":
            return None
    
        # 1) Адаптивная форма polyexponential (с учётом filter_mode/strength)
        def _adaptive_builder():
            if which_pass == "first":
                base_min, base_max, base_rho = 0.005, 20.0, 0.6
            else:
                base_min, base_max, base_rho = 0.01, 15.0, 0.5
    
            # Если adaptive выключен — используем фикс. базу, как раньше
            if not bool(self.config.get("adaptive_sigma_enable", False)):
                def _no_adapt(n):
                    return K.sampling.get_sigmas_polyexponential(n, base_min, base_max, base_rho, devices.device)
                return _no_adapt
    
            raw_strength = float(self.config.get("strength", 2.0))
            s = (raw_strength - 0.5) / 3.5
            s = max(0.0, min(1.0, s))
    
            filt = self.config.get("filter_mode", "Noise sync (sharp)") or "Noise sync (sharp)"
            f_off = float(self.config.get("filter_offset", 0.0))  # -1..1 мягкий сдвиг
    
            if "Morphological" in filt:
                sigma_min = base_min * (1.0 + 0.5 * s + 0.25 * f_off)
                sigma_max = base_max * (1.0 - 0.2 * s)
                rho = base_rho - 0.2 * s
            elif "Combined" in filt:
                sigma_min = base_min * (1.0 + 0.20 * (0.5 - s) + 0.10 * f_off)
                sigma_max = base_max
                rho = base_rho
            else:
                sigma_min = base_min * (1.0 - 0.5 * s - 0.25 * f_off)
                sigma_max = base_max * (1.0 + 0.10 * s)
                rho = base_rho + 0.20 * s
    
            sigma_min = max(1e-4, sigma_min)
            sigma_max = max(sigma_min * 1.01, sigma_max)
            rho = max(0.1, min(1.5, rho))
    
            def _f(n):
                return K.sampling.get_sigmas_polyexponential(n, sigma_min, sigma_max, rho, devices.device)
            return _f
    
        if schedule_mode == "Adaptive (filter/strength)":
            return _adaptive_builder()

        # Если k-diffusion недоступен, сразу откатываемся на адаптивную логику
        if not _HAS_KDIFF and schedule_mode in {"Karras", "Exponential", "Polyexponential", "Normal", "Simple", "DDIM uniform"}:
            return _adaptive_builder()

        # 2) Фиксированные режимы (через k-diffusion external wrapper)
        try:
            quantize = bool(getattr(shared.opts, "enable_quantization", False))
            if getattr(shared.sd_model, "parameterization", "eps") == "v":
                denoiser = K.external.CompVisVDenoiser
            else:
                denoiser = K.external.CompVisDenoiser
            model_wrap = denoiser(shared.sd_model, quantize=quantize)
    
            def _simple(n):
                # простая выборка из доступных сигм модели
                sigmas_all = model_wrap.sigmas
                step = max(1, int(len(sigmas_all) / max(1, n)))
                picked = [float(sigmas_all[-(1 + i * step)].item()) for i in range(n)]
                return torch.tensor(picked + [0.0], dtype=torch.float32, device=devices.device)
    
            def _ddim_uniform(n):
                # Униформные DDIM-тиктаймы -> сигмы
                try:
                    num_ddpm = model_wrap.inner_model.inner_model.num_timesteps
                except Exception as e:
                    print(f"[Custom Hires Fix] Warning: {e}")
                    num_ddpm = 1000
                c = max(1, num_ddpm // max(1, n))
                ddim_ts = np.asarray(list(range(0, num_ddpm, c)))
                steps_out = (ddim_ts + 1)[::-1]
                sigs = []
                for ts in steps_out:
                    ts = min(int(ts), 999)
                    sigs.append(model_wrap.t_to_sigma(torch.tensor(ts, device=devices.device)))
                sigs.append(torch.tensor(0.0, device=devices.device))
                return torch.stack([s if torch.is_tensor(s) else torch.tensor(float(s), device=devices.device) for s in sigs]).float()
    
            use_old = bool(getattr(shared.opts, "use_old_karras_scheduler_sigmas", False))
            if use_old:
                sigma_min, sigma_max = (0.1, 10.0)
            else:
                # sigmas обычно убывают: [sigma_max ... sigma_min]
                sigma_max = float(model_wrap.sigmas[0].item())
                sigma_min = float(model_wrap.sigmas[-1].item())
    
            def _karras(n):
                return K.sampling.get_sigmas_karras(n=n, sigma_min=sigma_min, sigma_max=sigma_max, device=devices.device)
    
            def _exp(n):
                return K.sampling.get_sigmas_exponential(n=n, sigma_min=sigma_min, sigma_max=sigma_max, device=devices.device)
    
            def _poly(n):
                rho = 0.5
                return K.sampling.get_sigmas_polyexponential(n=n, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho, device=devices.device)
    
            def _normal(n):
                return model_wrap.get_sigmas(n).to(devices.device)
    
            table = {
                "Karras": _karras,
                "Exponential": _exp,
                "Polyexponential": _poly,
                "Normal": _normal,
                "Simple": _simple,
                "DDIM uniform": _ddim_uniform,
            }
            fn = table.get(schedule_mode, None)
            return fn or _adaptive_builder()
        except Exception as e:
            print(f"[Custom Hires Fix] Warning: {e}")
            # мягкий откат к адаптивной логике
            return _adaptive_builder()


    # ───────────── Stop-at-step hook (Anti-Burn style) ─────────────
    @contextmanager
    def _stop_hook_context(self, limit_steps: int):
        """
        Вешает временный колбэк на on_cfg_denoiser и снимает его по выходу из контекста.
        На/после limit_steps:
          - выставляет shared.state.interrupt = True (мягко)
          - кидает sd_samplers_common.InterruptedException() (жёстко, как Anti-Burn)
        """
        cb_ref = None
        limit_steps = int(max(1, limit_steps))
        try:
            def _cb(params):
                try:
                    step = getattr(params, "sampling_step", None)
                    if step is None:
                        return
                    # params.sampling_step обычно 0-based → останавливаем на «человеческом» шаге
                    if (int(step) + 1) >= limit_steps:
                        try:
                            shared.state.interrupt = True  # на всякий случай — мягкий флаг
                        except Exception:
                            pass
                        # жёсткий способ, как в Anti-Burn:
                        raise sd_samplers_common.InterruptedException()
                except sd_samplers_common.InterruptedException:
                    # пробрасываем дальше — поймает сэмплер
                    raise
                except Exception as e:
                    print(f"[Custom Hires Fix] stop-hook callback warning: {e}")
                    return

            cb_ref = _cb
            script_callbacks.on_cfg_denoiser(cb_ref)
        except Exception as e:
            print(f"[Custom Hires Fix] stop-hook install failed, fallback to clamp: {e}")
            cb_ref = None
        try:
            yield
        finally:
            if cb_ref is not None:
                try:
                    # В новых WebUI есть явная отписка; если нет — игнор
                    remove = getattr(script_callbacks, "remove_callbacks_for_function", None)
                    if callable(remove):
                        remove(cb_ref)
                except Exception as e:
                    print(f"[Custom Hires Fix] stop-hook remove warning: {e}")

    def _first_pass(self, x: Image.Image) -> Image.Image:
        # Determine target size
        if bool(self.config.get("mp_target_enabled", False)):
            w, h = self._maybe_mp_resize(x.width, x.height, float(self.config.get("mp_target", 2.0)))
        else:
            aspect = x.width / x.height if x.height else 1.0
    
            le = int(self.config.get("long_edge", 0))
            if le > 0:
                if x.width >= x.height:
                    w = int(max(8, round(le / 8) * 8))
                    h = int(max(8, round((le / aspect) / 8) * 8))
                else:
                    h = int(max(8, round(le / 8) * 8))
                    w = int(max(8, round((le * aspect) / 8) * 8))
            elif int(self.config.get("width", 0)) == 0 and int(self.config.get("height", 0)) == 0 and float(self.config.get("ratio", 0)) > 0:
                w = int(max(8, round(x.width * float(self.config["ratio"]) / 8) * 8))
                h = int(max(8, round(x.height * float(self.config["ratio"]) / 8) * 8))
            else:
                if int(self.config.get("width", 0)) > 0 and int(self.config.get("height", 0)) > 0:
                    w, h = int(self.config["width"]), int(self.config["height"])
                else:
                    # Fallback (не должен сработать при строгой валидации; оставлен для совместимости)
                    w, h = x.width, x.height
    
        self.width, self.height = w, h
    
        self._apply_token_merging(for_hr=True, halve=True)
    
        # Per-pass scheduler
        sched_first = self.config.get("scheduler_first", self.config.get("scheduler", "Use same scheduler"))
        self._set_scheduler_by_label(sched_first)
        self._coerce_scheduler_to_string()
    
        # Optional ControlNet
        if self._use_cn:
            try:
                cn_np = np.array(x.resize((self.width, self.height)))
                self._enable_controlnet(cn_np)
            except Exception as e:
                print(f"[Custom Hires Fix] Warning: {e}")
                pass
    
        # Build override prompt for first pass (none; base prompt)
        self._current_lora_factor = float(self.config.get("lora_weight_first_factor", 1.0))
        with devices.autocast(), torch.inference_mode():
            self._prepare_conditioning(self.width, self.height, int(self.config.get("steps_first", 20)))
    
        # Upscale (image domain) then (optionally) blend latent
        x_img = self._resize_with_upscaler(x, self.width, self.height, self.config.get("first_upscaler", "R-ESRGAN 4x+"))
        decoded, sample = self._to_sample(x_img)
        # Латент из исходника (до апскейла) для осмысленного смешивания
        _, sample_orig = self._to_sample(x)
        factor = self._vae_down_factor()
        x_latent = self._interp(
            sample_orig, size=(self.height // factor, self.width // factor),
            _default_mode="nearest"
        )
    
        # NEW: Anti-twinning latent shrink/expand (опционально)
        if bool(self.config.get("deep_shrink_enable", False)):
            shrink = float(self.config.get("deep_shrink_strength", 0.5))
            shrink = max(0.1, min(0.9, shrink))
            b, c, h, w = sample.shape
            sh, sw = max(1, int(h * shrink)), max(1, int(w * shrink))
            with torch.no_grad():
                try:
                    sample = F.interpolate(sample, size=(sh, sw), mode="bicubic", align_corners=False, antialias=True)
                    sample = F.interpolate(sample, size=(h, w),  mode="bicubic", align_corners=False, antialias=True)
                except TypeError:
                    sample = self._interp(sample, size=(sh, sw), _default_mode="bicubic")
                    sample = self._interp(sample, size=(h, w), _default_mode="bicubic")
    
        first_latent = float(self.config.get("first_latent", 0.3))
        if 0.0 <= first_latent <= 1.0:
            invert = bool(self.config.get("first_latent_invert", False))  # NEW
            if invert:
                # Новая семантика: слайдер = вес исходного латента, как на 2-м проходе
                sample = sample * (1.0 - first_latent) + x_latent * first_latent
            else:
                # Старая семантика
                sample = sample * first_latent + x_latent * (1.0 - first_latent)
    
        image_conditioning = self.p.img2img_image_conditioning(decoded, sample)
    
        # RNG setup
        self._saved_seeds = list(getattr(self.p, "seeds", [])) or None
        self._saved_subseeds = list(getattr(self.p, "subseeds", [])) or None
        self._saved_subseed_strength = getattr(self.p, "subseed_strength", None)
        self._saved_seed_resize_from_h = getattr(self.p, "seed_resize_from_h", None)
        self._saved_seed_resize_from_w = getattr(self.p, "seed_resize_from_w", None)
    
        self.p.rng = rng.ImageRNG(sample.shape[1:], self.p.seeds, subseeds=self.p.subseeds,
                                  subseed_strength=self.p.subseed_strength,
                                  seed_resize_from_h=self.p.seed_resize_from_h, seed_resize_from_w=self.p.seed_resize_from_w)
    
        # Denoise config for first pass
        steps = int(self.config.get("steps_first", int(self.config.get("steps", 20))))
        # Stop-at-step — режим clamping
        smode = str(self.config.get("stop_mode", "Off"))
        sfirst = int(self.config.get("stop_first_at", 0))
        if smode == "Clamp steps (built-in)" and sfirst > 0:
            steps = min(steps, sfirst)
        # NEW: LRU-кэш шума для первой стадии
        try:
            key_seed = (self.p.seeds[0] if getattr(self.p, "seeds", None) else getattr(self.p, "seed", None))
        except Exception as e:
            print(f"[Custom Hires Fix] Warning: {e}")
            key_seed = None
        use_cache = bool(self.config.get("reuse_noise_cache", True))
        sam_name = str(self.config.get("sampler_first", self.config.get("sampler", "")))
        sch_name = str(self.config.get("scheduler_first", self.config.get("scheduler", "")))
        noise_key = f"first|{sam_name}|{sch_name}|{tuple(sample.shape)}|{key_seed}"
        if use_cache:
            cached_noise = self._noise_cache.get(noise_key)
            if cached_noise is not None:
                self._noise_cache.move_to_end(noise_key)
                noise = cached_noise.to(sample.device, dtype=sample.dtype)
            else:
                noise = torch.randn_like(sample)
                self._noise_cache[noise_key] = noise.detach().to("cpu", copy=True)
                # NEW: лимит берём из конфигурации
                if len(self._noise_cache) > int(self.config.get("noise_cache_max", 16)):
                    self._noise_cache.popitem(last=False)
        else:
            noise = torch.randn_like(sample)
        if bool(self.config.get("reuse_seed_noise", False)):
            self._first_noise = noise.detach().clone()
            self._first_noise_shape = tuple(sample.shape)
    
        self.p.denoising_strength = self._compute_denoise("denoise_first")
        # NEW: мягкий буст денойза для SDXL/SD3 (если включено)
        self._maybe_apply_sdxl_denoise_boost()
        self.p.cfg_scale = float(self.cfg)
    
        # --- sigma schedule override (с адаптацией или без) ---
        schedule_mode = self.config.get("noise_schedule_mode", "Use sampler default")
        fn_sched_override = self._build_sigma_override("first", schedule_mode)
        if hasattr(self.p, "sampler_noise_scheduler_override"):
            self.p.sampler_noise_scheduler_override = fn_sched_override
    
        self.p.batch_size = 1
    
        # Per-pass sampler
        sampler_first = self.config.get("sampler_first", self.config.get("sampler", "DPM++ 2M Karras"))
        sampler = self._create_sampler(sampler_first)

        # На всякий случай очистим поле у сэмплера перед установкой нового override
        try:
            if hasattr(sampler, "noise_scheduler_override") and \
               bool(self.config.get("cleanup_noise_overrides", True)):
                sampler.noise_scheduler_override = None
        except Exception as _e: print(f"[Custom Hires Fix] Warning: {_e}")


        # NEW: если у сэмплера есть поле для оверрайда, выставим и туда
        try:
            if fn_sched_override is not None and hasattr(sampler, "noise_scheduler_override"):
                sampler.noise_scheduler_override = fn_sched_override
        except Exception as e:
            print(f"[Custom Hires Fix] Warning: {e}")

        # Stop-at-step — режим hook & interrupt
        if smode == "Hook & interrupt (Anti-Burn style)" and sfirst > 0:
            ctx = self._stop_hook_context(sfirst)
        else:
            ctx = nullcontext()
        with ctx:
            samples = sampler.sample_img2img(
                self.p, sample.to(devices.dtype), noise, self.cond, self.uncond,
                steps=steps, image_conditioning=image_conditioning
            ).to(devices.dtype_vae)
    
        devices.torch_gc()
        decoded_sample = processing.decode_first_stage(shared.sd_model, samples)
        if torch.isnan(decoded_sample).any().item():
            devices.torch_gc()
            samples = torch.clamp(samples, -3, 3)
            decoded_sample = processing.decode_first_stage(shared.sd_model, samples)
    
        decoded_sample = torch.clamp((decoded_sample + 1.0) / 2.0, min=0.0, max=1.0).squeeze()
        x_np = 255.0 * np.moveaxis(decoded_sample.to(torch.float32).cpu().numpy(), 0, 2)
        return Image.fromarray(x_np.astype(np.uint8))
    
    def _second_pass(self, x: Image.Image) -> Image.Image:
        # Determine target size for second pass
        # NEW: приоритетный кастомный размер для 2-го прохода
        w = h = 0
        if bool(self.config.get("second_custom_size_enable", False)):
            sw = int(self.config.get("second_width", 0))
            sh = int(self.config.get("second_height", 0))
            if sw > 0 and sh > 0:
                w = max(8, int(round(sw / 8) * 8))
                h = max(8, int(round(sh / 8) * 8))
    
        if not (w and h):
            # старая логика выбора размера
            if bool(self.config.get("mp_target_enabled", False)):
                w, h = self._maybe_mp_resize(x.width, x.height, float(self.config.get("mp_target", 2.0)))
            else:
                le = int(self.config.get("long_edge", 0))
                if le > 0:
                    aspect = x.width / x.height if x.height else 1.0
                    if x.width >= x.height:
                        w = int(max(8, round(le / 8) * 8))
                        h = int(max(8, round((le / aspect) / 8) * 8))
                    else:
                        h = int(max(8, round(le / 8) * 8))
                        w = int(max(8, round((le * aspect) / 8) * 8))
                elif (int(self.config.get("width", 0)) == 0 and int(self.config.get("height", 0)) == 0 and
                        float(self.config.get("ratio", 0)) > 0):
                    w = int(max(8, round(x.width * float(self.config["ratio"]) / 8) * 8))
                    h = int(max(8, round(x.height * float(self.config["ratio"]) / 8) * 8))
                else:
                    if int(self.config.get("width", 0)) > 0 and int(self.config.get("height", 0)) > 0:
                        w, h = int(self.config["width"]), int(self.config["height"])
                    else:
                        # Fallback (не должен сработать при строгой валидации; оставлен для совместимости)
                        w, h = x.width, x.height
    
        self._apply_token_merging(for_hr=True)
    
        # Per-pass scheduler
        sched_second = self.config.get("scheduler_second", self.config.get("scheduler", "Use same scheduler"))
        self._set_scheduler_by_label(sched_second)
        self._coerce_scheduler_to_string()
    
        if self._use_cn:
            cn_img = x if bool(self.config.get("cn_ref", False)) else self.pp.image
            try:
                self._enable_controlnet(np.array(cn_img.resize((w, h))))
            except Exception as e:
                print(f"[Custom Hires Fix] Warning: {e}")
                pass
    
        # Build override prompt for second pass
        base_prompt = self.config.get("prompt", "").strip() or self.p.prompt.strip()
        p2 = (self.config.get("second_pass_prompt", "") or "").strip()
        if p2:
            if bool(self.config.get("second_pass_prompt_append", True)):
                prompt_override = (base_prompt + ", " + p2) if base_prompt else p2
            else:
                prompt_override = p2
        else:
            prompt_override = None
    
        # Apply LoRA scaling for second pass
        self._current_lora_factor = float(self.config.get("lora_weight_second_factor", 1.0))
        with devices.autocast(), torch.inference_mode():
            self._prepare_conditioning(w, h, int(self.config.get("steps_second", 20)), prompt_override=prompt_override)
    
        # Optional latent mix
        x_latent = None
        second_latent = float(self.config.get("second_latent", 0.1))
        if second_latent > 0:
            _, sample_from_img = self._to_sample(x)
            factor = self._vae_down_factor()
            x_latent = self._interp(
                sample_from_img, size=(h // factor, w // factor),
                _default_mode="nearest"
            )
    
        # Upscale to target and encode
        if second_latent < 1.0:
            x_up = self._resize_with_upscaler(x, w, h, self.config.get("second_upscaler", "R-ESRGAN 4x+"))
            decoded, sample = self._to_sample(x_up)
        else:
            decoded, sample = self._to_sample(x)
    
        if x_latent is not None and 0.0 <= second_latent <= 1.0:
            sample = (sample * (1.0 - second_latent)) + (x_latent * second_latent)
    
            # Гарантируем, что decoded и sample согласованы по spatial-размеру (латент*8 пикселей)
            factor = self._vae_down_factor()
            tH, tW = sample.shape[-2] * factor, sample.shape[-1] * factor
            if decoded.shape[-2:] != (tH, tW):
                decoded = (self._interp(decoded.unsqueeze(0), size=(tH, tW), _default_mode="bilinear")).squeeze(0)
    
        image_conditioning = self.p.img2img_image_conditioning(decoded, sample)
    
        # RNG: optionally reuse seed/noise
        if bool(self.config.get("reuse_seed_noise", False)) and self._saved_seeds is not None:
            try:
                self.p.seeds = list(self._saved_seeds)
                self.p.subseeds = list(self._saved_subseeds) if self._saved_subseeds is not None else self.p.subseeds
                self.p.subseed_strength = self._saved_subseed_strength if self._saved_subseed_strength is not None else self.p.subseed_strength
                self.p.seed_resize_from_h = self._saved_seed_resize_from_h if self._saved_seed_resize_from_h is not None else self.p.seed_resize_from_h
                self.p.seed_resize_from_w = self._saved_seed_resize_from_w if self._saved_seed_resize_from_w is not None else self.p.seed_resize_from_w
            except Exception as e:
                print(f"[Custom Hires Fix] Warning: {e}")
                pass
    
        self.p.rng = rng.ImageRNG(sample.shape[1:], self.p.seeds, subseeds=self.p.subseeds,
                                  subseed_strength=self.p.subseed_strength,
                                  seed_resize_from_h=self.p.seed_resize_from_h, seed_resize_from_w=self.p.seed_resize_from_w)
    
        # Denoise config for second pass
        steps = int(self.config.get("steps_second", int(self.config.get("steps", 20))))
        # Stop-at-step — режим clamping
        smode = str(self.config.get("stop_mode", "Off"))
        ssecond = int(self.config.get("stop_second_at", 0))
        if smode == "Clamp steps (built-in)" and ssecond > 0:
            steps = min(steps, ssecond)
        if bool(self.config.get("cfg_second_pass_boost", True)):
            self.p.cfg_scale = float(self.cfg) + float(self.config.get("cfg_second_pass_delta", 3.0))
        else:
            self.p.cfg_scale = float(self.cfg)
        self.p.denoising_strength = self._compute_denoise("denoise_second")
        # NEW: мягкий буст денойза для SDXL/SD3 (если включено)
        self._maybe_apply_sdxl_denoise_boost()
    
        # Noise: reuse tensor if shapes match, else fresh noise
        if bool(self.config.get("reuse_seed_noise", False)) and self._first_noise is not None:
            if tuple(sample.shape) == tuple(self._first_noise_shape or ()):
                noise = self._first_noise.to(sample.device, dtype=sample.dtype)
            else:
                # NEW: если размеры не совпали — попробуем кэш
                try:
                    key_seed = (self.p.seeds[0] if getattr(self.p, "seeds", None) else getattr(self.p, "seed", None))
                except Exception as e:
                    print(f"[Custom Hires Fix] Warning: {e}")
                    key_seed = None
                use_cache = bool(self.config.get("reuse_noise_cache", True))
                sam_name = str(self.config.get("sampler_second", self.config.get("sampler", "")))
                sch_name = str(self.config.get("scheduler_second", self.config.get("scheduler", "")))
                noise_key = f"second|{sam_name}|{sch_name}|{tuple(sample.shape)}|{key_seed}"
                if use_cache:
                    cached_noise = self._noise_cache.get(noise_key)
                    if cached_noise is not None:
                        self._noise_cache.move_to_end(noise_key)
                        noise = cached_noise.to(sample.device, dtype=sample.dtype)
                    else:
                        noise = torch.randn_like(sample)
                        self._noise_cache[noise_key] = noise.detach().to("cpu", copy=True)
                        max_items = int(self.config.get("noise_cache_max", 16))
                        while len(self._noise_cache) > max_items:
                            self._noise_cache.popitem(last=False)
                else:
                    noise = torch.randn_like(sample)
        else:
            # NEW: обычный путь — используем LRU-кэш
            try:
                key_seed = (self.p.seeds[0] if getattr(self.p, "seeds", None) else getattr(self.p, "seed", None))
            except Exception as e:
                print(f"[Custom Hires Fix] Warning: {e}")
                key_seed = None
            use_cache = bool(self.config.get("reuse_noise_cache", True))
            sam_name = str(self.config.get("sampler_second", self.config.get("sampler", "")))
            sch_name = str(self.config.get("scheduler_second", self.config.get("scheduler", "")))
            noise_key = f"second|{sam_name}|{sch_name}|{tuple(sample.shape)}|{key_seed}"
            if use_cache:
                cached_noise = self._noise_cache.get(noise_key)
                if cached_noise is not None:
                    self._noise_cache.move_to_end(noise_key)
                    noise = cached_noise.to(sample.device, dtype=sample.dtype)
                else:
                    noise = torch.randn_like(sample)
                    self._noise_cache[noise_key] = noise.detach().to("cpu", copy=True)
                    max_items = int(self.config.get("noise_cache_max", 16))
                    while len(self._noise_cache) > max_items:
                        self._noise_cache.popitem(last=False)
            else:
                noise = torch.randn_like(sample)
    
        # --- sigma schedule override (с адаптацией или без) ---
        schedule_mode = self.config.get("noise_schedule_mode", "Use sampler default")
        fn_sched_override = self._build_sigma_override("second", schedule_mode)
        if hasattr(self.p, "sampler_noise_scheduler_override"):
            self.p.sampler_noise_scheduler_override = fn_sched_override
    
        self.p.batch_size = 1
    
        # Per-pass sampler
        sampler_second = self.config.get("sampler_second", self.config.get("sampler", "DPM++ 2M Karras"))
        sampler = self._create_sampler(sampler_second)

        # На всякий случай очистим поле у сэмплера перед установкой нового override
        try:
            if hasattr(sampler, "noise_scheduler_override") and \
               bool(self.config.get("cleanup_noise_overrides", True)):
                sampler.noise_scheduler_override = None
        except Exception as _e: print(f"[Custom Hires Fix] Warning: {_e}")


        # NEW: зеркально — если поддерживается, прописать прямо на сэмплер
        try:
            if fn_sched_override is not None and hasattr(sampler, "noise_scheduler_override"):
                sampler.noise_scheduler_override = fn_sched_override
        except Exception as e:
            print(f"[Custom Hires Fix] Warning: {e}")

        # Stop-at-step — режим hook & interrupt
        if smode == "Hook & interrupt (Anti-Burn style)" and ssecond > 0:
            ctx = self._stop_hook_context(ssecond)
        else:
            ctx = nullcontext()
        with ctx:
            samples = sampler.sample_img2img(
                self.p, sample.to(devices.dtype), noise, self.cond, self.uncond,
                steps=steps, image_conditioning=image_conditioning
            ).to(devices.dtype_vae)

        devices.torch_gc()
        decoded_sample = processing.decode_first_stage(shared.sd_model, samples)
        if torch.isnan(decoded_sample).any().item():
            devices.torch_gc()
            samples = torch.clamp(samples, -3, 3)
            decoded_sample = processing.decode_first_stage(shared.sd_model, samples)
    
        decoded_sample = torch.clamp((decoded_sample + 1.0) / 2.0, min=0.0, max=1.0).squeeze()
        x_np = 255.0 * np.moveaxis(decoded_sample.to(torch.float32).cpu().numpy(), 0, 2)
        out_img = Image.fromarray(x_np.astype(np.uint8))
    
        # Post-FX (более предсказуемый порядок):
        # Match colors → CLAHE → Unsharp
        if bool(self.config.get("match_colors_enabled", False)):
            out_img = self._apply_match_colors(out_img, self.pp.image)
        out_img = self._apply_clahe(out_img)
        out_img = self._apply_unsharp(out_img)
    
        return out_img
    
    def _final_upscale_tiled(self, img: Image.Image, scale: float) -> Image.Image:
        # (удалён неиспользуемый локальный хелпер _diag_mem_estimate)
        """
        Улучшенный тайловый апскейл с:
          - Hann (half-cosine) пером для смешивания оверлапа (лучше линейного);
          - опциональным reflect-padding на границах кадра для устойчивости апскейлеров;
          - разложением целевого масштаба на «нативные» множители (2×/4×) + остаток.
    
        Параметры берём из self.config:
          - final_upscaler (str)      — имя апскейлера;
          - final_tile (int)          — базовый размер тайла (до скейла);
          - final_tile_overlap (int)  — перекрытие тайлов (до скейла).
        Дополнительно распознаём НЕобязательные ключи (авто-дефолты):
          - final_blend_window: "hann" | "linear" (по умолчанию "hann")
          - final_reflect_pad:  bool (по умолчанию True)
        """
        upscaler_name = self.config.get("final_upscaler", "R-ESRGAN 4x+")
        tile        = int(self.config.get("final_tile", 512))
        overlap     = int(self.config.get("final_tile_overlap", 16))
        window_mode = (self.config.get("final_blend_window", "hann") or "hann").lower()
        use_reflect = bool(self.config.get("final_reflect_pad", True))
    
        tile = max(64, tile)
        overlap = max(0, min(overlap, tile // 2))
        scale = max(1.0, float(scale))
    
        # Если масштаб 1× — возвращаем как есть
        if scale <= 1.0:
            return img
    
        # --- 1) Разложение масштаба на факторы (напр. [2,2,1.5] для 6×) ---
        def _parse_native_factor(name: str) -> int:
            # Ищем 2x/4x в названии апскейлера; если нет — считаем 2x «нативным»
            import re
            m = re.search(r"(\d+)\s*[xX]|[xX]\s*(\d+)", (name or "").strip())
            if not m:
                return 2
            return max(2, int((m.group(1) or m.group(2)).strip()))
    
        native = max(2, _parse_native_factor(upscaler_name))
        factors = []
        remaining = scale
        while remaining / native >= 1.0 + 1e-6:
            factors.append(float(native))
            remaining /= native
        if remaining > 1.0 + 1e-6:
            factors.append(float(remaining))
        elif not factors:  # если scale < native и нет факторов — просто остаток
            factors.append(float(scale))
    
        # --- 2) Хелперы для весов и апскейла ---
        def _hann_ramp(length: int, left_ovl: int, right_ovl: int) -> np.ndarray:
            """
            Half-cosine окно: на левой/правой зоне плавно 0→1/1→0, в центре — 1.
            """
            w = np.ones(int(length), dtype=np.float32)
            if left_ovl > 0:
                t = np.linspace(0.0, np.pi / 2.0, left_ovl, endpoint=False, dtype=np.float32)
                w[:left_ovl] = np.sin(t) ** 2  # 0..1
            if right_ovl > 0:
                t = np.linspace(0.0, np.pi / 2.0, right_ovl, endpoint=False, dtype=np.float32)
                w[-right_ovl:] = (np.cos(t) ** 2)  # 1..0
            return w
    
        def _linear_ramp(length: int, left_ovl: int, right_ovl: int) -> np.ndarray:
            w = np.ones(int(length), dtype=np.float32)
            if left_ovl > 0:
                w[:left_ovl] = np.linspace(0.0, 1.0, left_ovl, endpoint=False, dtype=np.float32)
            if right_ovl > 0:
                w[-right_ovl:] = np.minimum(w[-right_ovl:], np.linspace(1.0, 0.0, right_ovl, endpoint=False, dtype=np.float32))
            return w
    
        ramp1d = _hann_ramp if window_mode == "hann" else _linear_ramp
    
        def _upscale(img_pil: Image.Image, s: float) -> Image.Image:
            new_w = int(round(img_pil.width  * s))
            new_h = int(round(img_pil.height * s))
            return self._resize_with_upscaler(img_pil, new_w, new_h, upscaler_name)
    
        # --- 3) Чейнится апскейл над всей картинкой (быстро) до ближайшего меньшего масштаба,
        #         а последний этап делаем тайлами для «безопасной» склейки ---
        # Если factors больше одного, последнюю ступень делаем тайлами, предыдущие — сразу на всю картинку.
        if len(factors) > 1:
            for f in factors[:-1]:
                img = _upscale(img, f)
            final_scale = factors[-1]
        else:
            final_scale = factors[0]
    
        W, H = img.width, img.height
        TW, TH = int(round(W * final_scale)), int(round(H * final_scale))
    
        # Memory guard: estimate ~16 bytes per output pixel (accum + weight)
        mem_est_bytes = int(TH) * int(TW) * 16
        budget = None
        # 1) Пытаемся оценить по свободной RAM
        try:
            import psutil  # optional
            budget = int(0.35 * psutil.virtual_memory().available)
        except Exception:
            budget = None
        # 2) Если нет psutil — ориентируемся на свободную VRAM
        if budget is None:
            try:
                free_vram = self._get_free_vram_bytes()
                if free_vram is not None:
                    budget = int(0.45 * free_vram)
            except Exception:
                budget = None
        # 3) Совсем без источников — консервативная константа
        if budget is None:
            budget = int(1.5 * 1024 ** 3)
        if mem_est_bytes > budget:
            print(f"[Custom Hires Fix] Final upscale tiling: estimated {mem_est_bytes/1024/1024/1024:.2f} GB; falling back to single-pass resize.")
            try:
                return self._resize_with_upscaler(img, TW, TH, upscaler_name)
            except Exception:
                return img.resize((TW, TH), Image.LANCZOS)
    
        accum = np.zeros((TH, TW, 3), dtype=np.float32)
        weight = np.zeros((TH, TW), dtype=np.float32)
        step_pre = tile - overlap
        if step_pre <= 0:
            step_pre = tile
    
        # Небольшой helper для reflect-паддинга
        def _reflect_pad_if_needed(arr: np.ndarray, left: int, top: int, right: int, bottom: int) -> np.ndarray:
            if left or top or right or bottom:
                return np.pad(arr, ((top, bottom), (left, right), (0, 0)), mode="reflect")
            return arr

        # Кэш весовых карт для одинаковых размеров, чтобы не пересчитывать
        window_cache: dict[tuple[int, int, int, int, int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    
        for y in range(0, H, step_pre):
            for x in range(0, W, step_pre):
                # Берём тайл с левым оверлапом; правый даём только если будет следующий шаг
                x0 = max(0, x - overlap)
                y0 = max(0, y - overlap)
                x1 = min(W, x + tile)
                y1 = min(H, y + tile)
                crop = img.crop((x0, y0, x1, y1))
    
                # Reflect-padding по краям кадра, если запрошено
                if use_reflect and (x0 == 0 or y0 == 0 or x1 == W or y1 == H):
                    arr = np.array(crop)
                    # считаем недостающий «выход» за границы слева/сверху
                    pad_left   = max(0, x0 - (x - overlap))
                    pad_top    = max(0, y0 - (y - overlap))
                    pad_right  = max(0, (x + tile) - x1)
                    pad_bottom = max(0, (y + tile) - y1)
                    arr = _reflect_pad_if_needed(arr, pad_left, pad_top, pad_right, pad_bottom)
                    crop = Image.fromarray(arr, mode="RGB")
    
                # Апскейлим тайл
                up = _upscale(crop, final_scale)
                up_np = np.array(up).astype(np.float32)
                uh, uw = up_np.shape[0], up_np.shape[1]
    
                # Координаты вставки на апскейленном полотне
                ox = int(round(x0 * final_scale))
                oy = int(round(y0 * final_scale))
    
                # Сколько пикселей апскейленного тайла относится к зонам оверлапа
                left_ovl   = int(round((x - x0) * final_scale))
                top_ovl    = int(round((y - y0) * final_scale))
                right_ovl  = 0
                bottom_ovl = 0
                if (x + step_pre) < W:
                    right_ovl = int(round(((min(W, x + tile) - x) - step_pre) * final_scale))
                if (y + step_pre) < H:
                    bottom_ovl = int(round(((min(H, y + tile) - y) - step_pre) * final_scale))
    
                left_ovl   = max(0, min(left_ovl, uw // 2))
                right_ovl  = max(0, min(right_ovl, uw // 2))
                top_ovl    = max(0, min(top_ovl,  uh // 2))
                bottom_ovl = max(0, min(bottom_ovl, uh // 2))
    
                # Достаём/строим 2D-веса
                key = (uw, uh, left_ovl, right_ovl, top_ovl, bottom_ovl)
                cached = window_cache.get(key)
                if cached is None:
                    wx = ramp1d(uw, left_ovl, right_ovl)
                    wy = ramp1d(uh, top_ovl, bottom_ovl)
                    w2d = (wy[:, None] * wx[None, :]).astype(np.float32)
                    window_cache[key] = (wx, wy, w2d)
                else:
                    _, _, w2d = cached
    
                # Клауним координаты в пределах полотна и при необходимости режем тайл/окно
                y_to = min(oy + uh, TH)
                x_to = min(ox + uw, TW)
                crop_h = y_to - oy
                crop_w = x_to - ox
                if crop_h <= 0 or crop_w <= 0:
                    # Тайл полностью вне холста (защита от редких погрешностей округления)
                    continue
                # Аккумулируем только видимую часть тайла и соответствующее окно весов
                accum[oy:y_to, ox:x_to, :] += up_np[:crop_h, :crop_w, :] * w2d[:crop_h, :crop_w][..., None]
                weight[oy:y_to, ox:x_to]   += w2d[:crop_h, :crop_w]
    
        weight = np.clip(weight, 1e-6, None)
        out = (accum / weight[..., None]).astype(np.uint8)
        return Image.fromarray(out, mode="RGB")
    
    
    def _enable_controlnet(self, image_np: np.ndarray):
        if not getattr(self, "_cn_ext", None):
            return
        for unit in self._cn_units:
            try:
                if getattr(unit, "model", "None") != "None":
                    if getattr(unit, "enabled", True):
                        unit.guidance_start = float(self.config.get("start_control_at", 0.0))
                        # безопасный предел для VRAM (теперь настраиваемый)
                        min_side = min(image_np.shape[0], image_np.shape[1])
                        cap = int(self.config.get("cn_proc_res_cap", 1024))
                        cap = max(256, min(4096, cap))
                        unit.processor_res = max(256, min(cap, min_side))
                        if getattr(unit, "image", None) is None:
                            unit.image = image_np
                self.p.width = image_np.shape[1]
                self.p.height = image_np.shape[0]
            except Exception as e:
                print(f"[Custom Hires Fix] Warning: {e}")
                continue
        try:
            self._cn_ext.update_cn_script_in_processing(self.p, self._cn_units)
            for script in self.p.scripts.alwayson_scripts:
                if script.title().lower() == "controlnet":
                    script.controlnet_hack(self.p)
        except Exception as e:
            print(f"[Custom Hires Fix] Warning: {e}")
            pass
    
    
def parse_infotext(infotext, params):
    try:
        block = params.get("Custom Hires Fix")
        if block is None:   # <-- было: if not block
            return
        # поддержка нового формата "json64:<base64(json)>", а также обратная совместимость
        if isinstance(block, str) and block.startswith("json64:"):
            data = json.loads(base64.b64decode(block[7:]).decode("utf-8"))
        elif isinstance(block, str):
            try:
                data = json.loads(block)  # сначала пробуем валидный JSON
            except json.JSONDecodeError:
                # фолбэк для старых строк с одинарными кавычками
                data = json.loads(block.translate(QUOTE_SINGLE_TO_DOUBLE))
        else:
            data = block
        params["Custom Hires Fix"] = data
        scale = data.get("scale", 0)
        scale_str = str(scale).lower()
        if "x" in scale_str:
            w, _, h = scale_str.partition("x")
            data["ratio"] = 0.0
            try:
                data["width"]  = int(w.strip())
                data["height"] = int(h.strip())
            except Exception:
                data["width"], data["height"] = 0, 0
        else:
            try:
                r = float(scale)
            except Exception as e:
                print(f"[Custom Hires Fix] Warning: {e}")
                r = 0.0
            data["ratio"] = r
            data["width"] = int(data.get("width", 0) or 0)
            data["height"] = int(data.get("height", 0) or 0)


        # Defaults for new/legacy fields
        if "steps_first" not in data:
            data["steps_first"] = int(data.get("steps", 20))
        if "steps_second" not in data:
            data["steps_second"] = int(data.get("steps", 20))

        # per-pass sampler/scheduler defaults from legacy single values
        data.setdefault("sampler_first", data.get("sampler", ""))
        data.setdefault("sampler_second", data.get("sampler", ""))
        data.setdefault("scheduler_first", data.get("scheduler", "Use same scheduler"))
        data.setdefault("scheduler_second", data.get("scheduler", "Use same scheduler"))

        # CFG delta defaults
        data.setdefault("cfg_second_pass_boost", True)
        data.setdefault("cfg_second_pass_delta", 3.0)

        # Flags defaults
        data.setdefault("reuse_seed_noise", False)
        data.setdefault("reuse_noise_cache", True)
        data.setdefault("mp_target_enabled", False)
        data.setdefault("mp_target", 2.0)
        data.setdefault("cond_cache_enabled", True)
        data.setdefault("cond_cache_max", 64)
        data.setdefault("noise_cache_max", 16)
        data.setdefault("vae_tiling_enabled", False)
        data.setdefault("seamless_tiling_enabled", False)
        data.setdefault("tile_overlap", 12)
        data.setdefault("lora_weight_first_factor", 1.0)
        data.setdefault("lora_weight_second_factor", 1.0)
        data.setdefault("match_colors_preset", "Off")
        data.setdefault("match_colors_enabled", False)
        data.setdefault("match_colors_strength", 0.5)
        data.setdefault("postfx_preset", "Off")
        data.setdefault("clahe_enabled", False)
        data.setdefault("clahe_clip", 2.0)
        data.setdefault("clahe_tile_grid", 8)
        data.setdefault("unsharp_enabled", False)
        data.setdefault("unsharp_radius", 1.5)
        data.setdefault("unsharp_amount", 0.75)
        data.setdefault("unsharp_threshold", 0)
        data.setdefault("second_pass_prompt", "")
        data.setdefault("second_pass_prompt_append", True)
        data.setdefault("cn_proc_res_cap", 1024)

        # final upscale defaults
        data.setdefault("final_upscale_enable", False)
        data.setdefault("final_upscaler", "R-ESRGAN 4x+")
        data.setdefault("final_scale", 4.0)      # NEW
        data.setdefault("final_tile", 512)
        data.setdefault("final_tile_overlap", 16)
        # NEW: anti-twinning & SDXL defaults
        data.setdefault("legacy_cfg_handling", False)
        data.setdefault("cleanup_noise_overrides", True)
        data.setdefault("deep_shrink_enable", False)
        data.setdefault("deep_shrink_strength", 0.5)
        data.setdefault("sdxl_mode", False)
        data.setdefault("sdxl_denoise_boost", 0.1)
        # Stop-at-step defaults
        data.setdefault("stop_mode", "Off")
        data.setdefault("stop_first_at", 0)
        data.setdefault("stop_second_at", 0)
        # NEW (инверсия латента и размер 2-го прохода)
        data.setdefault("first_latent_invert", False)
        data.setdefault("second_custom_size_enable", False)
        data.setdefault("second_width", 0)
        data.setdefault("second_height", 0)

        # Новое поле по умолчанию — добавляем внутри try
        data.setdefault("adaptive_sigma_enable", False)
        data.setdefault("restore_scheduler_after", True)
        data.setdefault("latent_resample_enable", True)
        data.setdefault("latent_resample_mode", "nearest")
        data.setdefault("noise_schedule_mode", "Use sampler default")

        # безопасные дефолты для старых инфотекстов
        data.setdefault("prompt", "")
        data.setdefault("negative_prompt", "")
        data.setdefault("first_upscaler", "R-ESRGAN 4x+")
        data.setdefault("second_upscaler", "R-ESRGAN 4x+")
        data.setdefault("first_latent", 0.3)
        data.setdefault("second_latent", 0.1)
        data.setdefault("denoise_first", 0.33)
        data.setdefault("denoise_second", 0.45)
        data.setdefault("denoise_offset", 0.05)
        data.setdefault("filter_offset", 0.0)
        data.setdefault("strength", 2.0)
        data.setdefault("filter_mode", "Noise sync (sharp)")
        data.setdefault("clip_skip", 0)
        data.setdefault("cn_ref", False)
        data.setdefault("start_control_at", 0.0)
        data.setdefault("final_blend_window", "hann")
        data.setdefault("final_reflect_pad", True)
        
    except Exception as e:
        print(f"[Custom Hires Fix] Warning: {e}")
        return
# Register paste-params hook (guard against double registration)
try:
    _INFOTEXT_HOOK_REGISTERED
except NameError:
    _INFOTEXT_HOOK_REGISTERED = False
if not _INFOTEXT_HOOK_REGISTERED:
    script_callbacks.on_infotext_pasted(parse_infotext)
    _INFOTEXT_HOOK_REGISTERED = True
