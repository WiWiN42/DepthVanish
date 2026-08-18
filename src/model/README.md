
This file documents how to setup attack target models for our stereo patch attack.

First, go clone all the official repos into local then follow the setup provided at <THIS_REPO>/README.md for the environment.




As some evaluated models are out-of-date, the setup would be problematic on new version of torch. This file documents what we encodered during setup the models for evaluation.

NOTE: you should first  this file is only for reference in case of some typical failures occurs

## Why patches are needed

`PSMNet/`, `DeepPruner/`, and `aanet/` here are official upstream repos, freshly cloned.
Each needs a small set of source patches to work with this project's wrapper classes
(`psmnet_model.py`, `deeppruner_model.py`, `aanet_model.py`) and checkpoints
(`_checkpoints/`). Without these patches:

- **PSMNet and DeepPruner fail immediately** with `TypeError: __init__() got an unexpected
  keyword argument ...` when their wrapper constructs the model, because the wrappers pass
  constructor arguments (`num_deform_layers`, `method`) that the vanilla official classes
  don't accept.
- **Even past that, both then break in eval mode** — the wrappers assume a specific output
  shape (a tuple of predictions) that the vanilla official `forward()` no longer returns once
  real `self.training`-based branching is restored.
- **AANet imports and runs as-is**, but silently risks autograd errors/wrong gradients under
  this project's optimization loop unless patched (see AANet section).

Reference implementations with these patches already applied live in the sibling `*_legacy/`
folders (`PSMNet_legacy/`, `DeepPruner_legacy/`, `aanet_legacy/`) — those are the
pre-existing, working copies this project ran against before. **Keep those folders around**;
if you ever need to re-verify a patch below, `diff -ru <legacy_dir> <official_dir>` (with
`__pycache__`, `*.so`, `*.pth`, `*.tar` excluded) is the fastest way to see it directly.

Every patch below is a **source-only** change and doesn't require rebuilding anything except
where noted (AANet's CUDA extension, if you need to rebuild it at all — the `.so` files are
already prebuilt and vendored).

**Checkpoint compatibility**, verified by inspecting `state_dict` keys directly:

| Checkpoint | Deform-conv keys present? | Matches |
|---|---|---|
| `_checkpoints/aanet_kitti15-fb2a0d23.pth` | Yes (15 keys, e.g. `feature_extractor.layer3.0.conv2.deform_conv.weight`) | Deform-conv is core to AANet's own architecture — unaffected by these patches |
| `_checkpoints/pretrained_model_KITTI2015.tar` (PSMNet) | **No** (0 of 429 keys) | Vanilla PSMNet architecture — but the wrapper's constructor call still requires the `num_deform_layers` kwarg to exist, regardless of which value (0) ends up passed |
| `_checkpoints/DeepPruner-best-kitti.tar` | **No** (0 of 656 keys), and 0 `patch_match`-named keys | Matches the `method="best"` / `patch_match` sampler config restored below |

None of the three checkpoints actually need the *custom* deform layers active by default —
`num_deform_layers=0` (PSMNet/DeepPruner's own default) selects the vanilla blocks. The
patches are required because the wrapper classes' **call signatures** (constructor kwargs,
expected output shape) don't match the official code, independent of which architecture
variant ends up selected at runtime.

## AANet (build deform_conv with torch==2.5.1)

### Source patches needed

1. **`nets/deform_conv/src/deform_conv_cuda.cpp`: `AT_CHECK` → `TORCH_CHECK`.** `AT_CHECK`
   is a deprecated PyTorch C++ macro; `TORCH_CHECK` replaced it. Both take identical
   `(condition, message)` arguments — pure rename, safe as a blanket find/replace. Doesn't
   affect already-compiled `.so` files, but the official source **won't compile** against a
   modern PyTorch as-shipped. (Checked: no other file under `deform_conv/src/` — including
   the `.cu` kernel — uses `AT_CHECK`, so the `.cpp` file is the only one that needs it.)

    ```
    cd /data3/luqi/yxing/stereo_PhysicalAttack/src/model/aanet/nets/deform_conv

    # 先看一遍有哪些文件受影响
    grep -rn 'AT_CHECK' src/

    # 备份后替换
    cp -r src src.bak
    sed -i 's/\bAT_CHECK\b/TORCH_CHECK/g' src/*.cpp src/*.cu src/*.h 2>/dev/null

    # 确认替换干净
    grep -rn 'AT_CHECK' src/ || echo "已全部替换"
    ```

2. **`nets/refinement.py`: `inplace=True` → `inplace=False` (4 occurrences).** This is the
   one patch here that actually matters for correctness, not just for building. The official
   code uses `inplace=True` for the LeakyReLU/ReLU ops in the refinement head. In-place
   tensor ops are a classic source of `RuntimeError: a leaf Variable that requires grad has
   been used in an in-place operation`, or silently incorrect gradients, in autograd graphs
   more complex than a plain forward/backward pass — exactly the situation here, where AANet
   sits inside a larger differentiable rendering + patch-optimization graph. Change all 4:
    ```diff
    -                         nn.LeakyReLU(0.2, inplace=True))
    +                         nn.LeakyReLU(0.2, inplace=False))
    ```
    and (3x, identical each time):
    ```diff
    -        disp = F.relu(disp + residual_disp, inplace=True)  # [B, 1, H, W]
    +        disp = F.relu(disp + residual_disp, inplace=False)  # [B, 1, H, W]
    ```

3. **`nets/aanet.py`: restore `return_features` parameter on `forward()`.** Not used
   anywhere in this repo's own code (grepped: the only `return_features` hits are in the
   unrelated SoM/SAM integration under `src/tools/SoM/`, a different model entirely).
   Restored purely for interface parity in case future code passes it — functionally inert
   either way, lowest-priority of the three.
    ```diff
    -    def forward(self, left_img, right_img):
    +    def forward(self, left_img, right_img, return_features=False):
    ```

Not restored (confirmed cosmetic / unused, left as official): a handful of blank-line and
commented-out `#print(...)` debug statements in `nets/aanet.py`/`nets/feature.py` (zero
behavioral difference); `README.md`, `train.py`, `requirements.txt`,
`run_aanet_inference.sh`, `yml2pip.py` (legacy-only demo/training/packaging scripts —
grepped, nothing in this project's pipeline references any of them).

### Rebuild the CUDA extension

```
rm -rf build/ *.so
bash build.sh 2>&1 | tee build2.txt
echo "exit: ${PIPESTATUS[0]}"
```

## PSMNet

### 1. Restore `models/submodule_deform.py` (missing entirely from the official clone)
This file doesn't exist upstream at all — copy it from
`PSMNet_legacy/models/submodule_deform.py`. It's self-contained (its own `BasicBlock`,
`DeformBlock`, `matchshifted`, `disparityregression`, `feature_extraction`) — nothing else in
`PSMNet/models/` needs to change to support it.

The legacy copy already includes a `sys.path` fix for its `from nets.deform import
DeformConv2d` line (it reuses AANet's deform-conv implementation rather than vendoring its
own copy — the only `nets/` directory in the whole repo is under `aanet/`). The **raw
original line** `sys.path.insert(0, os.path.join('model', 'aanet'))` is relative to the
process's cwd and silently fails whenever this script is invoked from anywhere other than
`src/` (this was the root cause of an earlier `ModuleNotFoundError: No module named 'nets'`
bug). The version to copy already resolves the path from `__file__` instead:
```python
_SRC_MODEL_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # .../src/model
sys.path.insert(0, os.path.join(_SRC_MODEL_DIR, 'aanet'))
from nets.deform import DeformConv2d
```
If starting from a truly pristine official clone (no pre-fixed legacy copy available), use
this `__file__`-based version rather than the original cwd-relative one.

### 2. `models/stackhourglass.py` — 3 changes

**a) Import the restored deform variant instead of vanilla `submodule`:**
```diff
-from .submodule import *
+from .submodule_deform import *
```

**b) Restore `num_deform_layers` constructor parameter** (the wrapper always passes it —
without this, model construction raises `TypeError`):
```diff
-    def __init__(self, maxdisp):
+    def __init__(self, maxdisp, num_deform_layers=0):
         super(PSMNet, self).__init__()
         self.maxdisp = maxdisp
+        self.num_deform_layers = num_deform_layers

-        self.feature_extraction = feature_extraction()
+        self.feature_extraction = feature_extraction(self.num_deform_layers)
```

**c) Force the 3-prediction-tuple return path regardless of train/eval mode** (two spots).
`psmnet_model.py`'s own `forward()` unconditionally does `outputs[2]` when the *wrapper's*
mode is `'eval'` — it has no branch for a single-tensor output. The official code correctly
gates the tuple return behind `if self.training:`, but since `PSMNetModel.__init__` always
calls `.eval()` (→ `self.training = False`), that means the official code would return a
single tensor in exactly the situation where the wrapper expects a 3-tuple, crashing with
`IndexError: index 2 is out of bounds for dimension 0 with size 1` (batch size is always 1 in
this pipeline). Restore the override:
```diff
-        if self.training:
+        if True: #self.training:
+        #if self.training:
                cost1 = F.upsample(cost1, ...)
                ...
```
(both the multi-scale cost-volume block, and the final return — see
`PSMNet_legacy/models/stackhourglass.py` lines 136 and 157 for the exact indentation, since
indentation shifts by one level in the "always execute" version.)

Not restored (confirmed cosmetic, left as official): `models/submodule.py`,
`models/Test_img.py`, `main.py`, `finetune.py`, all `dataloader/*.py` — differ, but none are
imported by this project (only `models/stackhourglass.py` is, via `psmnet_model.py`).
`models/submodule.py` in particular is only used by `models/basic.py` (the
non-stackhourglass variant), which this project never imports either. Also various
trailing-whitespace / blank-line / tab-vs-space diffs throughout `stackhourglass.py` and
`submodule_deform.py` — zero behavioral difference.

## DeepPruner

### 1. Restore `models/feature_extractor_deform.py` (missing entirely)
Copy from `DeepPruner_legacy/deeppruner/models/feature_extractor_deform.py` as-is — it has
no `sys.path` issues of its own (`from models.submodules import BasicBlock, convbn_relu,
DeformBlock` resolves fine once `DeepPruner/deeppruner` is already on `sys.path`, which
`deeppruner_model.py` already handles).

### 2. `models/submodules.py` — restore `DeformBlock` + `convbn_deform`
Both are entirely absent from the official file. Add, right after the `import math` line:
```python
import os
import sys
_SRC_MODEL_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # .../src/model
sys.path.insert(0, os.path.join(_SRC_MODEL_DIR, 'aanet'))
from nets.deform import DeformConv2d


def convbn_deform(in_planes, out_planes, kernel_size, stride, pad, dilation):
    return nn.Sequential(DeformConv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation, bias=False),
                         nn.BatchNorm2d(out_planes))
```
(same cwd-independence rationale as PSMNet step 1 above — this file is one directory level
deeper, hence 4 `dirname()` calls instead of 3.) And a `DeformBlock` class, placed right
before `class BasicBlock` — see `DeepPruner_legacy/deeppruner/models/submodules.py` lines
71-94 for the exact class body (mirrors `BasicBlock` but routes `conv2` through
`convbn_deform` instead of `convbn`).

### 3. `models/deeppruner.py` — 4 changes

**a) Restore `method`/`num_deform_layers` constructor parameters** (same TypeError risk as
PSMNet — the wrapper always passes both):
```diff
-    def __init__(self):
+    def __init__(self,method="best", num_deform_layers=0):
         super(DeepPruner, self).__init__()
-        self.scale = args.cost_aggregator_scale
+        if (method == "fast"):
+            self.scale = 8
+            self.feature_extractor_refinement_level_outplanes = 64
+        else:
+            self.scale = args.cost_aggregator_scale
+            self.feature_extractor_refinement_level_outplanes = args.feature_extractor_refinement_level_outplanes
         self.max_disp = args.max_disp // self.scale
         self.mode = args.mode
+        self.num_deform_layers = num_deform_layers
```

**b) Import the restored deform feature extractor:**
```diff
         else:
-            from models.feature_extractor_best import feature_extraction
+            from models.feature_extractor_deform import feature_extraction
```

**c) Use the per-method attribute instead of the raw config value**, and pass
`num_deform_layers` through:
```diff
-        refinement_inplanes = args.feature_extractor_refinement_level_outplanes + self.post_CRP_sample_count + 2 + 1
+        refinement_inplanes = self.feature_extractor_refinement_level_outplanes + self.post_CRP_sample_count + 2 + 1
...
-        self.feature_extraction = feature_extraction()
+        self.feature_extraction = feature_extraction(num_deform_layers=self.num_deform_layers)
```

**d) Remove the official code's added `if self.mode == 'evaluation':` early return in
`forward()`.** `DeepPrunerModel.eval()` sets `self.model.mode = 'evaluation'` directly — with
the official code's added branch, that now returns a *single tensor*
(`refined_disparity.squeeze(1)`), but `deeppruner_model.py`'s wrapper unconditionally does
`outputs[0]` expecting the first element of a *tuple* of predictions. On a single tensor,
`outputs[0]` silently slices the batch dimension instead of erroring — this is a **shape
corruption bug**, worse than a clean crash, since it doesn't fail loudly. Legacy's
`forward()` has no `self.mode` branching at all in its return logic — it always returns a
tuple (4 elements for `scale==4`/`"best"`, 5 for `scale==8`/`"fast"`), which is exactly what
the wrapper expects regardless of `self.mode`. Delete the added block entirely:
```diff
-        if self.mode == 'evaluation':
-            if self.scale == 8:
-                refined_disparity_1 = F.interpolate(refined_disparity_1 * 2, scale_factor=(2, 2),
-                                                    mode='bilinear').squeeze(1)
-                return refined_disparity_1
-            return refined_disparity.squeeze(1)
-
         min_disparity = F.interpolate(min_disparity * self.scale, ...
```

### 4. `models/config.py` — restore `patch_match` sampler default
```diff
+    # "post_CRP_sampler_type": "uniform", #change to patch_match for Sceneflow model.
-    "post_CRP_sampler_type": "uniform", #change to patch_match for Sceneflow model.
+    "post_CRP_sampler_type": "patch_match",
```
(Checkpoint inspection found zero `patch_match`-named parameters in
`DeepPruner-best-kitti.tar`, consistent with — though not absolute proof of — this being the
sampler config the "best" checkpoint was paired with. `PatchMatch` is a differentiable,
non-parametric sampling operation, so this setting doesn't change what keys exist in the
checkpoint either way; it only changes *how* disparity candidates are sampled during the
forward pass.)

Not restored (confirmed cosmetic, left as official): `config.py`'s commented-out `# "mode":
"evaluation", ...` alternative line — the *active* value (`"mode": "training"`) is identical
in both versions, and is overridden immediately anyway by `DeepPrunerModel.eval()` setting
`self.model.mode = 'evaluation'` at runtime, so this line never has any effect regardless.
`README.md` (both locations), all `dataloader/*.py`, `finetune_kitti.py`,
`submission_kitti.py`, `train_sceneflow.py`/`.sh`, `run_deeppruner_train.sh` — not referenced
anywhere in this project's own pipeline (only `models/deeppruner.py` and `models/config.py`
are imported, via `deeppruner_model.py`). Also trailing-whitespace / blank-line diffs
throughout — zero behavioral difference.

## Verifying a patch reapplication

After applying all of the above, this should reproduce cleanly with `torch` importable but
no CUDA (e.g. on a machine without the prebuilt `.so`, or without a GPU at all):

```bash
python3 -c "
import sys
sys.path.insert(0, 'src')
from model.aanet_model import *
from model.psmnet_model import *
from model.deeppruner_model import *
print('all three imported OK up to the point of needing the compiled deform_conv_cuda extension')
"
```
All three should fail at the *identical* final step —
`ImportError: cannot import name 'deform_conv_cuda' from partially initialized module
'nets.deform_conv'` inside `aanet/nets/deform_conv/deform_conv.py` — which is expected and
fine: that's the prebuilt CUDA extension boundary, not a patch gap. If any of the three fails
earlier or differently than that, a patch above was missed or mis-applied.
