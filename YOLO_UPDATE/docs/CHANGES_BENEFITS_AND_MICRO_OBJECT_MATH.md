# YOLO UPDATE Changes, Benefits, and Micro-Object Math

Date: 2026-06-22
Scope: YOLO UPDATE architecture and training logic in `YOLO_UPDATE/`

## 1. Executive Summary

YOLO UPDATE is a YOLO-family detector modified for extreme small-object detection. It is not a random architecture. It keeps the YOLO pattern of a staged convolutional backbone, feature pyramid neck, dense one-pass prediction heads, objectness/class branches, and distribution-style box regression. The major changes are targeted at preserving 2x2-pixel evidence and making the training signal stable when IoU becomes mathematically brittle.

The core design change is this:

```text
Standard YOLO:     input -> downsample -> P3/P4/P5 detection
YOLO UPDATE:       input -> preserve P1/P2 detail -> P2/P3/P4/P5 detection
                  optional P1 detector only in accuracy mode
```

For a 2-pixel-wide and 2-pixel-tall object, there is not enough raw information to recover fine semantic detail from the object pixels alone. The model can preserve the evidence that a small object-like signal exists, localize it at native resolution, and use surrounding context, temporal evidence, and semantic feature injection to improve classification. This distinction matters: the architecture improves detection and localization of 2x2 objects; it does not magically create high-resolution texture that the sensor never captured.

## 2. What Changed

### 2.1 Backbone Changes

Standard YOLO backbones normally downsample early and rely on P3/P4/P5 or similar detection scales. A 2x2 object can disappear after early stride-2 and stride-4 reductions. YOLO UPDATE changes the early backbone:

- Adds a stride-1 P1 detail path.
- Uses a narrow `EdgePreserveBlock` detail branch.
- Uses `SpaceToDepthConv` for the first downsample.
- Uses anti-aliased downsampling instead of raw stride-2 convolution.
- Keeps C2/C3/C4/C5/C6 staged features like a YOLO-style backbone.
- Keeps C2f-style blocks and SPPF context.

Why it helps:

- P1 keeps the original pixel grid alive.
- P2 becomes the primary micro-object detection scale.
- Space-to-depth preserves a 2x2 neighborhood as channels instead of averaging or skipping samples.
- Anti-aliased downsampling reduces false texture artifacts and missed tiny objects caused by aliasing.

### 2.2 Neck Changes

Standard YOLO necks fuse features from low-resolution semantic maps and higher-resolution maps. YOLO UPDATE adds explicit micro-object fusion:

- Emits P1/P2/P3/P4/P5 pyramid features.
- Uses weighted top-down and bottom-up fusion.
- Adds `MicroFusion` to inject semantic P2 context into P1 detail features.
- Keeps P1 as a refinement/detail path by default.
- Detects on P2-P5 by default, with P1 detection available only in accuracy mode.

Why it helps:

- Tiny object evidence remains spatially precise.
- Larger-context semantic features help separate real tiny objects from noise.
- P1 detection is expensive, so making it optional keeps the default model deployable.

### 2.3 Head Changes

YOLO UPDATE uses a dense anchor-free head with separate branches:

- quality/objectness logits,
- box distribution logits,
- subpixel offsets,
- box uncertainty,
- closed-set class logits,
- region embeddings,
- mask coefficients.

Why it helps:

- Anchor-free prediction avoids hand-tuned 2x2 or 3x3 anchors.
- Distribution-style box regression is smoother than direct four-number regression.
- Subpixel offsets matter because a one-pixel error is half the width of a 2x2 object.
- Region embeddings and mask coefficients prepare the model for unified multi-task expansion.

### 2.4 Training Changes

YOLO UPDATE adds a trainable criterion that connects YOLO-format labels to model outputs:

- size-aware level assignment,
- P2/P3 assignment for micro and small objects,
- area-weighted tiny-object localization,
- normalized Wasserstein distance loss for micro boxes,
- quality targets based on tiny-box similarity,
- BCE class/objectness terms.

Why it helps:

- IoU is unstable for 2x2 boxes; NWD gives a smoother signal.
- Small-object weighting prevents tiny objects from being drowned out by large objects.
- Size-aware assignment ensures micro objects produce positives on the correct feature levels.

## 3. Major Modifications From Standard YOLO

| Area | Standard YOLO Pattern | YOLO UPDATE Modification | Reason |
| --- | --- | --- | --- |
| First useful detection level | Usually P3 or stride 8 in many variants | P2 default, optional P1 accuracy mode | 2x2 objects are damaged by stride 4/8 maps. |
| Early downsampling | Stride convolution | Space-to-depth plus anti-aliased downsample | Preserve 2x2 pixel neighborhoods. |
| High-res detail | Usually limited or omitted | Dedicated P1 detail branch | Retain native-resolution micro evidence. |
| Feature fusion | PAN/FPN | Weighted PAN/FPN plus MicroFusion | Inject semantics into high-res maps without blurring detail. |
| Box assignment | YOLO-style scale/center assignment | Size-aware P1/P2/P3/P4/P5 routing | Force micro targets onto viable levels. |
| Box loss | IoU/CIoU/DFL dominant | IoU-like term plus NWD micro loss | IoU collapses under tiny coordinate errors. |
| Center precision | Cell-level or normal offset | Explicit subpixel offset branch | 1 pixel error is severe for 2-pixel objects. |
| Multi-task hooks | Usually detect-only or detect+seg variant | Detection, masks, embeddings, image logits in one model | Supports unified roadmap. |

## 4. Mathematical Functions and Equations

This section explains the math used to preserve 2x2 evidence and train stable tiny-object localization.

### 4.1 Space-to-Depth Preservation

For an input feature map:

```text
X in R^(B x C x H x W)
```

space-to-depth with block size 2 rearranges each 2x2 spatial neighborhood into channels:

```text
Y[b, c * 4 + 2i + j, h, w] = X[b, c, 2h + i, 2w + j]
where i, j in {0, 1}
```

Output shape:

```text
Y in R^(B x 4C x H/2 x W/2)
```

Why this helps a 2x2 object:

- A normal stride-2 convolution can discard or average parts of the object.
- Space-to-depth keeps all four pixels of a 2x2 object as four channel values at one lower-resolution cell.
- The model can then learn channel mixing with a 1x1/3x3 convolution instead of losing the evidence during sampling.

### 4.2 Anti-Aliased Downsampling

Downsampling should first reduce high-frequency aliasing:

```text
Y = Conv(LowPass(X))
```

In the current implementation, the low-pass approximation is average pooling:

```text
LowPass(X)[h, w] = (1 / 4) * sum_{i=0..1} sum_{j=0..1} X[2h + i, 2w + j]
```

Why this helps:

- Tiny bright/dark dots can alias into false features.
- Low-pass downsampling makes reduced-resolution features more stable.
- The first reduction uses space-to-depth because preserving a 2x2 object is more important there; later reductions use anti-aliased pooling to reduce noise.

### 4.3 Weighted Feature Fusion

YOLO UPDATE uses learned non-negative fusion weights:

```text
F_out = (sum_i ReLU(w_i) * F_i) / (epsilon + sum_i ReLU(w_i))
```

where:

```text
w_i are learned scalar fusion weights
F_i are aligned feature maps
epsilon prevents division by zero
```

Why this helps:

- The model can learn how much to trust semantic low-resolution maps versus high-resolution detail maps.
- For micro objects, too much low-resolution semantic injection can blur or suppress the target.
- Learned fusion lets the network adapt the balance per fusion node.

### 4.4 MicroFusion Gate

MicroFusion injects semantic context into detail features using a learned gate.

Let:

```text
D = projected P1 detail feature
S = upsample(projected P2 semantic feature)
```

The gate is:

```text
G = sigmoid(Conv(concat(D, S)))
```

The fused output is:

```text
F_micro = Refine(D + G * S)
```

Why this helps:

- `D` preserves exact spatial evidence.
- `S` provides context and class/scene information.
- `G` decides how much semantic context to inject.
- Multiplicative gating prevents semantic features from blindly overwriting the 2-pixel signal.

### 4.5 Anchor-Free Cell Geometry

For a pyramid level with stride `s`, each cell `(x, y)` maps to an image-space center:

```text
C_x = (x + 0.5) * s
C_y = (y + 0.5) * s
```

YOLO UPDATE adds a learned subpixel residual:

```text
C'_x = C_x + s * tanh(delta_x)
C'_y = C_y + s * tanh(delta_y)
```

Why this helps:

- A 2x2 object cannot tolerate coarse cell-center localization.
- The residual lets the model move the predicted center within approximately one stride around the cell center.

### 4.6 Distribution Box Regression

The head predicts a discrete distribution over distances for each side:

```text
z_l, z_t, z_r, z_b in R^(reg_max + 1)
```

Convert logits to probabilities:

```text
p_k = softmax(z)_k
```

Expected distance:

```text
d = sum_{k=0..reg_max} k * p_k
```

Image-space side distances:

```text
D = s * d
```

Decoded box:

```text
x0 = C'_x - D_l
y0 = C'_y - D_t
x1 = C'_x + D_r
y1 = C'_y + D_b
```

Why this helps:

- The model learns a distribution, not only one hard scalar.
- This is more stable around tiny boxes where one-pixel changes cause large relative errors.

### 4.7 Size-Aware Assignment

For a target box:

```text
w = x1 - x0
h = y1 - y0
m = max(w, h)
```

YOLO UPDATE routes target boxes by size:

```text
m <= 5 px       -> P1/P2 candidate levels
5 < m <= 16 px  -> P2/P3
16 < m <= 64 px -> P3/P4
m > 64 px       -> P4/P5
```

The center-positive radius is:

```text
r = clamp(0.75 * min(w, h), 1.0, 4.0)
```

Why this helps:

- Micro objects must be assigned to feature maps that still contain their evidence.
- At least one positive cell is created for tiny targets instead of losing them through rounding.

### 4.8 Small-Object Weighting

For target area:

```text
A = w * h
```

YOLO UPDATE applies positive localization weighting:

```text
w_small(A) = min(w_max, sqrt(A_ref / max(A, 1)))
```

Current defaults:

```text
A_ref = 256
w_max = 6
```

For a 2x2 object:

```text
A = 4
w_small = min(6, sqrt(256 / 4)) = min(6, 8) = 6
```

Why this helps:

- Large objects naturally dominate aggregate loss because they are easier and produce more stable gradients.
- The weighting raises the importance of tiny positive examples without letting them explode unbounded.

### 4.9 IoU and Why It Fails for 2x2 Boxes

Intersection-over-union is:

```text
IoU(B, T) = area(B intersection T) / area(B union T)
```

For a 2x2 target, a one-pixel shift can reduce overlap dramatically:

```text
target = 2 x 2 area = 4
prediction shifted by 1 px horizontally:
intersection = 1 * 2 = 2
union = 4 + 4 - 2 = 6
IoU = 2 / 6 = 0.333
```

A one-pixel shift in both directions:

```text
intersection = 1 * 1 = 1
union = 4 + 4 - 1 = 7
IoU = 1 / 7 = 0.143
```

Why this matters:

- These are small absolute errors but huge IoU penalties.
- Pure IoU loss can produce noisy gradients for micro boxes.

### 4.10 Normalized Wasserstein Distance

YOLO UPDATE uses an approximate box Gaussian representation:

```text
B = (cx, cy, w, h)
T = (cx_t, cy_t, w_t, h_t)
```

Distance:

```text
NWD_distance(B, T) =
  (cx - cx_t)^2
+ (cy - cy_t)^2
+ ((w - w_t)^2 + (h - h_t)^2) / 4
```

Similarity:

```text
NWD_similarity(B, T) = exp(-sqrt(NWD_distance(B, T)) / C)
```

Current constant:

```text
C = 12.8
```

NWD loss:

```text
L_NWD = 1 - NWD_similarity
```

Why this helps:

- NWD changes smoothly under small center and size errors.
- It does not collapse as abruptly as IoU for 2x2 boxes.
- It provides useful gradients even when tiny boxes barely overlap.

### 4.11 Quality Target

For assigned positive cells, the quality/objectness target can combine IoU and NWD similarity:

```text
q = alpha * IoU + (1 - alpha) * NWD_similarity
```

For very small boxes, NWD should dominate:

```text
alpha < 0.5 for micro objects
```

The current training path uses NWD-driven quality for assigned positives in `YOLOUpdateDetectionCriterion`.

Why this helps:

- Objectness is trained to reflect localization quality.
- NWD prevents the quality target from collapsing to near-zero after a one-pixel error.

### 4.12 Binary Cross-Entropy for Objectness and Classes

For logit `z` and target `y`:

```text
BCE(z, y) = -y * log(sigmoid(z)) - (1 - y) * log(1 - sigmoid(z))
```

YOLO UPDATE uses BCE for:

- quality/objectness map,
- closed-set class targets.

Why this helps:

- Dense objectness training suppresses false positives.
- Class BCE supports multi-label extension if needed.

### 4.13 Total Positive Detection Loss

The tiny detection loss combines:

```text
L_positive =
  lambda_box * L_box
+ lambda_nwd * L_NWD_micro
+ lambda_quality * L_quality
+ lambda_cls * L_class
+ lambda_subpixel * L_subpixel
```

Current defaults:

```text
lambda_box = 7.5
lambda_nwd = 2.0
lambda_quality = 1.0
lambda_cls = 0.5
lambda_subpixel = 0.5
```

The trainable criterion also adds dense quality-map supervision:

```text
L_total = L_positive + lambda_neg * L_quality_map
```

Current default:

```text
lambda_neg = 0.25
```

Why this helps:

- Positive loss teaches precise localization and classification.
- Quality-map loss teaches the dense detector not to fire everywhere.
- NWD and small-object weights make 2x2 samples meaningful during optimization.

## 5. How This Gets More Detail From a 2x2 Object

Strictly speaking, the network does not recover "detailed features" from a 2x2 object in the way it can from a 64x64 object. A 2x2 crop contains four pixels per channel. The system improves usable evidence through these mechanisms:

1. Preserve the four pixels:
   Space-to-depth keeps each pixel in the 2x2 neighborhood as channel evidence.

2. Avoid destructive downsampling:
   P1/P2 features prevent the target from vanishing before detection.

3. Add local contrast/detail processing:
   Edge-preserving high-resolution blocks let the model respond to small local discontinuities.

4. Inject context:
   MicroFusion adds semantic context from nearby and larger receptive fields.

5. Use smoother localization math:
   NWD keeps gradients useful when IoU is too harsh.

6. Increase tiny-object training weight:
   Area weighting prevents 2x2 examples from being ignored.

7. Decode with subpixel offsets:
   Center residuals reduce the quantization error of grid-cell predictions.

The practical result is better detection and localization of tiny objects, not true high-resolution reconstruction.

## 6. Current Implementation Locations

| Functionality | File |
| --- | --- |
| Backbone with P1/P2 preservation | `YOLO_UPDATE/yolo_update/models/backbone/yolo_update.py` |
| Space-to-depth, anti-alias, C2f, SPPF blocks | `YOLO_UPDATE/yolo_update/models/blocks.py` |
| Weighted FPN/PAN and MicroFusion | `YOLO_UPDATE/yolo_update/models/neck/micro_bifpn.py` |
| Unified anchor-free head | `YOLO_UPDATE/yolo_update/models/heads/unified_anchor_free.py` |
| Box decoding and DFL expectation | `YOLO_UPDATE/yolo_update/inference/decoder.py` |
| Size-aware assignment | `YOLO_UPDATE/yolo_update/losses/assignment.py` |
| NWD math | `YOLO_UPDATE/yolo_update/losses/nwd_loss.py` |
| Tiny-object loss | `YOLO_UPDATE/yolo_update/losses/tiny_box_loss.py` |
| Trainable criterion | `YOLO_UPDATE/yolo_update/training/criterion.py` |
| Trainer | `YOLO_UPDATE/yolo_update/engine/trainer.py` |

## 7. Bottom Line

YOLO UPDATE changes YOLO in the places that matter for 2x2-pixel targets:

- keep native-resolution evidence alive,
- use P2 as the main micro-object detection level,
- add semantic context without erasing detail,
- train tiny boxes with smoother math than IoU alone,
- give tiny positives enough gradient weight,
- keep P1 detection optional for accuracy-mode deployments.

The architecture improves the probability that a 2x2 object is detected and localized. It does not violate sensor physics: class certainty for 2x2 objects still depends heavily on context, repeated frames, priors, and downstream tracking.
