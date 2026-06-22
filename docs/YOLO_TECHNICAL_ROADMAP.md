# YOLO Technical Roadmap: Unified Micro-Object Vision Model

Status: draft architecture brief
Date: 2026-06-22
Audience: vision model engineering, perception platform, edge deployment

## 1. Executive Position

This roadmap defines a proprietary model, codenamed YOLO, that keeps the useful properties of modern YOLO-style systems: dense single-pass prediction, feature pyramid fusion, and edge-friendly deployment. The redesign is aimed at three requirements that usually fight each other:

- Detect objects as small as 2 to 5 pixels at native sensor resolution.
- Run detection, segmentation, and classification in one inference pass.
- Generalize to new categories and domains with zero-shot and few-shot mechanisms.

The practical position is direct: a 2-pixel object has almost no intrinsic class information in a single frame. Reliable operation depends on preserving native-resolution evidence, using context and temporal cues, and separating class-agnostic objectness from category recognition. The model can learn to say "there is a target-like object here" at 2 pixels. Fine-grained class labeling at that scale must come from surrounding context, motion history, prior geometry, higher-resolution crops when available, or exemplar/text embeddings. Any plan that assumes rich classification from two raw pixels alone will fail in the field.

Recommended system shape:

- High-resolution YOLO backbone with stride-1 and stride-2 detail paths.
- Feature pyramid outputs at P1/P2/P3/P4/P5 instead of the usual P3/P4/P5-only detection stack.
- Anchor-free dense heads with size-aware assignment and subpixel box regression.
- Unified decoupled heads for class-agnostic objectness, boxes, masks, closed-set classes, and open-vocabulary embeddings.
- Native-resolution tiled inference with overlap, temporal feature caching, and optional second-pass refinement for high-risk regions.
- Mixed-precision edge deployment where the first stages, high-resolution heads, and box regressors are protected from aggressive quantization.

## 2. Non-Negotiable Engineering Constraints

### 2.1 Pixel Evidence

For 2-pixel objects, downsampling is destructive. If the system resizes a 4K frame to 640 pixels before inference, most 2-pixel targets disappear. The model must operate on native-resolution tiles or high-resolution sensor crops.

Recommended default:

- Use 1280, 1536, or 2048 square tiles depending on edge memory.
- Use 96 to 192 pixels of overlap so small objects near tile borders remain visible with context.
- Preserve a stride-1 or stride-2 path through the backbone.
- Use temporal evidence when the platform has video. Motion turns subpixel or near-pixel signals into repeated evidence.

### 2.2 Real-Time Edge Meaning

"Real time" needs to be defined per product. For autonomous and industrial use, accuracy usually matters more than absolute frame rate. A 12 ms detector that misses a 3-pixel obstacle is worse than a 45 ms detector with high recall and bounded latency.

Recommended targets:

| Deployment class | Practical target | Notes |
| --- | ---: | --- |
| High-end edge GPU | 20 to 45 ms per high-res tile batch | Use TensorRT FP16/INT8 mixed precision. |
| Mid-range embedded GPU/NPU | 40 to 80 ms per frame with dynamic tiling | Use ROI refinement only when scene risk is high. |
| Low-power ARM/NPU | 80 to 150 ms per frame | Favor accuracy mode, smaller channels, and temporal accumulation. |

### 2.3 Generalization

Zero-shot and few-shot capability should not be bolted onto a closed-set classifier. The detector must learn class-agnostic objectness and region embeddings. Category identity becomes a matching problem against learned, text-derived, or exemplar-derived prototypes.

## 3. System Overview

Inference path:

```text
Native sensor frame
  -> calibration, denoise, motion metadata, optional stabilization
  -> overlapping high-resolution tiles
  -> YOLO backbone with P1/P2 detail preservation
  -> BiFPN/PAN neck with semantic injection into high-res maps
  -> unified heads:
       objectness + box distribution + mask coefficients
       closed-set classifier
       open-vocabulary embedding projector
       image/scene classification branch
  -> tile-local NMS or WBF
  -> cross-tile merge with uncertainty calibration
  -> temporal association and track-level confidence
  -> final detections, masks, labels, unknown-object events
```

Training path:

```text
Unlabeled domain video/images
  -> self-supervised visual pretraining and teacher distillation
Labeled detection/segmentation/classification data
  -> small-object curriculum and multi-task training
Open-vocabulary region-text/exemplar data
  -> embedding alignment and prototype training
Edge calibration data
  -> QAT, mixed precision export, hardware validation
```

## 4. Architecture Redesign

### 4.1 Input Strategy

Use native-resolution inference. Do not globally downscale below the point where a 2-pixel target remains at least 2 pixels after preprocessing.

Recommended tile sizes:

- Baseline: 1536 x 1536 with 128-pixel overlap.
- Accuracy mode: 2048 x 2048 with 192-pixel overlap.
- Low-power mode: 1280 x 1280 with 96-pixel overlap and temporal accumulation.

Preprocessing:

- Keep sensor demosaicing, color conversion, and normalization deterministic.
- Avoid compression artifacts in the training and deployment path.
- Add optional low-light denoise and deblur only if the same operation is available on edge hardware.
- Keep image sharpening conservative. Over-sharpening creates false positives at 2 to 5 pixels.

### 4.2 Backbone

The main change is to stop treating P3 as the first useful detection layer. Standard YOLO designs that begin detection at stride 8 have already destroyed most 2-pixel evidence. This design keeps P1 and P2 paths alive while using deeper stages for semantics.

Baseline backbone: `YOLO-Micro-B`

| Stage | Output stride | Example module | Channels | Repeats | Purpose |
| --- | ---: | --- | ---: | ---: | --- |
| Stem-A | 1 | 3x3 Conv, BN, SiLU | 32 | 1 | Initial local edges without downsampling. |
| Stem-B | 1 | 3x3 Conv, BN, SiLU | 48 | 1 | Stable high-res texture basis. |
| Detail sidecar | 1 | 1x1 Conv + EdgePreserveBlock | 24 | 2 | Thin, cheap P1 detail stream. |
| C2 | 2 | AntiAliasDown or SpaceToDepthConv + C2f-ED | 96 | 3 | Preserve 2x2 neighborhoods as channels. |
| C3 | 4 | BlurPool/Conv downsample + C2f | 160 | 4 | Small-object context. |
| C4 | 8 | Conv downsample + C2f + local attention | 256 | 6 | Mid-level semantics. |
| C5 | 16 | Conv downsample + C2f + SPPF | 384 | 6 | Large receptive field. |
| C6 | 32 | Conv downsample + C2f | 512 | 3 | Global context and large objects. |

Implementation notes:

- Use anti-aliased downsampling. A stride-2 convolution without low-pass behavior will erase or alias 2-pixel objects.
- Prefer Space-to-Depth downsampling at the first reduction. It moves each 2x2 neighborhood into channels before mixing, which is better than throwing away samples.
- Keep P1 narrow. A full-width stride-1 branch is too expensive and will dominate memory bandwidth.
- Avoid heavy global attention at P1/P2. Use local window attention, coordinate attention, or deformable convolution only where profiling shows value.
- Use SPPF or large-kernel depthwise modules at deeper stages for context, not at high resolution.
- Keep batch normalization fuseable for deployment. If domain variability is severe, use normalization adapters around the embedding branch rather than replacing every BN with GroupNorm.

Recommended model scales:

| Variant | Width | Depth | P1 detail channels | P2 channels | Primary use |
| --- | ---: | ---: | ---: | ---: | --- |
| Micro-S | 0.50 | 0.50 | 16 | 64 | Low-power edge, high frame count. |
| Micro-B | 0.75 | 0.75 | 24 | 96 | Default engineering baseline. |
| Micro-L | 1.00 | 1.00 | 32 | 128 | Accuracy-first edge GPU. |

### 4.3 Neck

Use a weighted bidirectional FPN/PAN neck, but add explicit high-resolution fusion.

Recommended pyramid:

- P1: stride 1, detail refinement only.
- P2: stride 2, primary micro-object detection.
- P3: stride 4, small-object detection and mask detail.
- P4: stride 8, normal object detection.
- P5: stride 16, large object detection and semantic context.
- Optional P6: stride 32 for wide-area scenes with large objects.

Fusion design:

```text
C6 -> P5 -> P4 -> P3 -> P2
                \       \-> P1 detail injection
                 \-> semantic context to P2/P1

P1/P2 -> light bottom-up refinement -> P3/P4
```

Layer recommendations:

- Use learned weighted fusion: `sum(w_i * x_i) / (epsilon + sum(w_i))`, with `w_i >= 0`.
- Use 1x1 lateral projections before fusion to keep channel dimensions controlled.
- Use nearest-neighbor or CARAFE-style upsampling for feature maps. Bilinear is acceptable but can smooth micro-object peaks.
- Add a feature-alignment layer before fusing high-res and low-res maps. Deformable convolution or learned offsets help when objects move across scale boundaries.
- Use a micro-object enhancement block on P1/P2:
  - 3x3 depthwise conv
  - 1x1 pointwise conv
  - residual edge gate from the detail sidecar
  - optional local contrast normalization

The neck should inject semantics into high-resolution maps. It should not ask the P1 branch to learn high-level classes from local pixels alone.

### 4.4 Unified Head

Use decoupled, anchor-free heads. The heads share a small task-adaptive stem per pyramid level, then split into detection, segmentation, classification, and embedding branches.

Per-level head:

```text
Pyramid feature Pk
  -> shared 3x3 Conv x 2
  -> objectness/quality branch
  -> box distribution branch
  -> closed-set class branch
  -> open-vocabulary embedding branch
  -> mask coefficient branch
```

Detection outputs:

- Objectness or quality score: class-agnostic.
- Box distances: left, top, right, bottom distribution with 16 to 32 bins.
- Subpixel center offset: continuous residual in image-pixel coordinates.
- Box uncertainty: optional log variance for downstream fusion.
- Class logits: closed-set categories.
- Region embedding: 256 or 512-dimensional L2-normalized vector for open-vocabulary matching.

Segmentation outputs:

- Prototype masks from P2/P3.
- Instance mask coefficients from each detection point.
- High-resolution boundary refinement from P1.
- For 2 to 5 pixel targets, permit point-mask or tiny ellipse representation. Requiring perfect instance polygons at 2 pixels creates annotation noise and bad gradients.

Classification outputs:

- Image-level classification branch from pooled P5/P6.
- Object-level classification from detection regions.
- Open-vocabulary class score from prototype similarity.

Known and novel class handling:

- Known classes: closed-set classifier plus embedding prototype.
- Novel classes with text prompt: text encoder generates a category prototype; region embedding is compared to that prototype.
- Novel classes with examples: support images generate visual prototypes; detections match against prototype memory.
- Unknown objects: high objectness with low maximum category similarity becomes an unknown-object event rather than forced misclassification.

### 4.5 Why Anchor-Free

For extreme scale variance, anchor-free heads are easier to train and maintain. Anchor boxes at 2x2, 3x3, 5x5, and 8x8 pixels become highly sensitive to annotation noise and resizing. Anchor-free point assignment with distributional regression handles fractional centers and variable object shapes more cleanly.

Use adaptive anchors only if the deployment domain has stable target geometry, such as a fixed industrial part or aerial object class. Even then, keep a class-agnostic anchor-free branch for novel categories.

## 5. Feature Pyramid Refinements for 2-Pixel Objects

### 5.1 Preserve Early Evidence

Rules:

- Never discard the first two spatial scales.
- Use P2 as the primary micro-object detection level.
- Use P1 for refinement and boundary evidence, not a full heavy detector unless the hardware budget allows it.
- Keep semantic injection separate from detail extraction.

For an object with diameter `d` pixels:

| Object diameter | Preferred levels | Notes |
| ---: | --- | --- |
| 2 to 5 px | P1/P2 | Detect as class-agnostic objectness plus coarse category. |
| 6 to 16 px | P2/P3 | Standard class prediction becomes more reliable. |
| 17 to 64 px | P3/P4 | Normal YOLO behavior. |
| >64 px | P4/P5/P6 | Large-object path. |

### 5.2 High-Resolution Fusion Block

Use a `MicroFusion` block at P1/P2:

```text
detail = Conv1x1(P1_detail)
sem = Upsample(Conv1x1(P3_or_P4_semantic))
aligned = DeformAlign(sem, detail)
gate = sigmoid(Conv3x3(concat(detail, aligned)))
out = Conv3x3(detail + gate * aligned)
```

This makes semantics available to 2-pixel candidates without smearing the detail map.

### 5.3 Tiled Multi-Scale Inference

For the most demanding scenes, run one native-scale pass and one selective refinement pass:

1. Full-frame tiling at native resolution.
2. Merge candidates with low thresholds to preserve recall.
3. Select uncertain or high-risk regions.
4. Run high-resolution refinement crops only on those regions.
5. Fuse results with weighted box fusion and temporal association.

This is the best field trade-off I know for tiny targets: a single giant frame pass is too expensive, and a low-resolution pass misses the target.

## 6. Loss Functions

### 6.1 Total Objective

Use a weighted sum, with dynamic balancing after the first stabilization period:

```text
L_total =
  lambda_det  * L_det
+ lambda_mask * L_mask
+ lambda_img  * L_image_cls
+ lambda_emb  * L_embedding
+ lambda_dist * L_distill
```

Initial weights:

| Loss group | Weight |
| --- | ---: |
| Detection | 1.00 |
| Segmentation | 0.75 |
| Image classification | 0.25 |
| Embedding alignment | 0.20 |
| Teacher distillation | 0.30 |

After 10 to 20 epochs, enable uncertainty weighting or GradNorm-style balancing with caps so segmentation or embedding losses cannot starve tiny-object localization.

### 6.2 Detection Loss

```text
L_det =
  7.5 * L_box
+ 1.5 * L_dfl
+ 1.0 * L_quality
+ 0.5 * L_closed_cls
+ 2.0 * L_nwd_micro
+ 0.5 * L_subpixel
```

Recommended components:

- `L_box`: CIoU/GIoU for normal objects; weighted less for boxes smaller than 5 pixels because IoU becomes unstable.
- `L_nwd_micro`: normalized Wasserstein distance for tiny boxes, active when area is below 16 to 25 pixels.
- `L_dfl`: distribution focal loss over box distances, 16 to 32 bins.
- `L_quality`: quality focal loss or varifocal loss using IoU/NWD quality targets.
- `L_closed_cls`: focal or BCE loss for known classes.
- `L_subpixel`: Smooth L1 in image-pixel coordinates for center and size residuals.

Small-object weighting:

```text
w_small(A) = min(6.0, sqrt(256 / max(A, 1)))
```

Apply `w_small` to positive localization and quality terms only, then normalize by the sum of positive weights in the batch. Do not apply this blindly to classification negatives; it will increase false alarms.

### 6.3 Assignment Strategy

Use size-aware dynamic assignment:

- For objects with max side <= 5 pixels, assign positives on P1 and P2.
- For 6 to 16 pixels, assign P2 and P3.
- For 17 to 64 pixels, assign P3 and P4.
- For larger objects, assign P4/P5/P6.

Positive center radius:

```text
radius = clamp(0.75 * min(width, height), min=1.0 px, max=4.0 px)
```

For 2-pixel boxes, ensure at least one positive cell on P1/P2 even when augmentation or rounding moves the center. This single detail prevents many "model cannot learn tiny objects" failures that are actually assignment bugs.

Quality target:

```text
quality = 0.5 * IoU + 0.5 * NWD_similarity
```

For micro objects, increase NWD contribution to 0.7 because IoU is too brittle.

### 6.4 Segmentation Loss

```text
L_mask =
  1.0 * BCE_or_focal_mask
+ 1.0 * Dice
+ 0.25 * boundary_loss
+ 0.25 * point_consistency_micro
```

Notes:

- Use mask prototypes at P2/P3 and boundary refinement from P1.
- For tiny objects, accept point masks or soft ellipses where polygon labels are not meaningful.
- Keep segmentation gradients from overwhelming P1 detail features by using gradient clipping or task-specific adapters.

### 6.5 Embedding and Open-Vocabulary Loss

Use supervised contrastive loss plus region-text alignment:

```text
L_embedding =
  L_supcon(region_embedding, class_or_instance_id)
+ L_region_text(region_embedding, text_embedding)
+ L_region_exemplar(region_embedding, support_prototype)
```

Use temperature-scaled cosine similarity. Start with temperature around 0.07 and learn a bounded scale parameter for deployment calibration.

## 7. Zero-Shot, Few-Shot, and Novel Category Capability

### 7.1 Design Principle

Separate these questions:

1. Is there an object-like entity here?
2. Does it match a known closed-set class?
3. Does it match a text or exemplar-defined novel class?
4. Is it unknown but operationally important?

A closed-set YOLO head answers only question 2. This roadmap builds all four.

### 7.2 Zero-Shot Novel Classes

Mechanism:

- Train objectness class-agnostically across many domains and object types.
- Project region features into a semantic embedding space.
- Align region embeddings with text embeddings and image exemplar embeddings.
- At deployment, register new class names with prompt templates and optional negative prompts.

Inference:

```text
score(category c | region r) =
  sigmoid(objectness_r)
* softmax_or_sigmoid(tau * cosine(embed_r, prototype_c))
* calibration_c
```

Use multiple prompts per category:

- "a photo of a {class}"
- "a small {class} in the distance"
- "{class} viewed by an industrial camera"
- Domain-specific templates approved during validation.

Calibrate prompt prototypes offline per deployment family. Poor prompt calibration can move false positives more than the detector itself.

### 7.3 Few-Shot Adaptation

Support-set flow:

1. Collect 1 to 50 labeled examples for a new class.
2. Encode support crops with the same region embedding branch.
3. Build a class prototype as a robust mean, optionally with covariance.
4. Add hard negatives from the deployment domain.
5. Optionally fine-tune only lightweight adapters.

Recommended adaptation levels:

| Label count | Update strategy | Expected behavior |
| ---: | --- | --- |
| 1 to 5 | Prototype memory only | Fast registration, moderate false positives. |
| 6 to 20 | Prototype memory + calibration scale | Better thresholds and ranking. |
| 20 to 50 | Fine-tune LoRA/adapters in embedding head | Stronger separation with low risk. |
| >50 | Fine-tune embedding head and final neck adapters | Treat as a regular incremental class update. |

Do not fine-tune the full backbone for a few-shot class unless the domain has permanently changed and regression testing is available. Full-backbone updates are where small-object recall regressions usually enter.

### 7.4 Meta-Learning

Use episodic training, but keep it practical:

- Sample episodes by domain and class family.
- Use support/query splits with 1, 5, and 10-shot support sets.
- Train the embedding head and adapters to form stable prototypes.
- Include background-only episodes to reduce false positives.
- Include micro-object episodes where support examples are tiny, blurred, occluded, or low contrast.

Recommended approach:

- Prototypical learning for the embedding head.
- Reptile/MAML-style updates only for small adapter modules, not the whole detector.
- Hard-negative mining from visually similar unknown objects.
- Teacher distillation from a larger open-vocabulary detector or segmenter during training, with deployment using only the compact model.

### 7.5 Multi-Domain Generalization Without Retraining

Training domains should include real and synthetic variation:

- Day/night, glare, fog, rain, dust, motion blur, sensor noise.
- Industrial lighting, specular surfaces, conveyor vibration, lens distortion.
- Aerial, automotive, robotics, fixed-camera surveillance, and close-range inspection when relevant.
- Compression artifacts and ISP differences matching target hardware.

Model mechanisms:

- Heavy domain randomization during training.
- Self-supervised pretraining on unlabeled target-domain video.
- Domain-balanced sampling so easy domains do not dominate gradients.
- Feature normalization adapters in the embedding head.
- Confidence calibration per sensor family.
- Test-time adaptation limited to normalization statistics or prototype thresholds, not full model retraining.

## 8. Multi-Task Integration

### 8.1 Single-Pass Output Contract

Each inference call should return:

- Boxes with objectness, localization uncertainty, and source pyramid level.
- Known-class scores.
- Open-vocabulary scores for registered classes.
- Unknown-object score.
- Instance masks where meaningful.
- Image-level classification.
- Optional track association features.

### 8.2 Task Interference Controls

Multi-task learning often fails because segmentation and classification gradients damage localization. Use controls from the beginning:

- Separate final heads with only a small shared stem.
- Use task-specific adapters after the neck.
- Warm up detection for several epochs before enabling full mask and embedding loss.
- Clip per-task gradient norms.
- Monitor tiny-object recall separately when introducing each task.

Recommended schedule:

| Training stage | Epochs | Active losses |
| --- | ---: | --- |
| Stabilize detection | 0 to 20 | detection, small-object losses, distillation |
| Add segmentation | 20 to 40 | detection, segmentation, boundary |
| Add embedding | 40 to 70 | detection, segmentation, embedding |
| Joint refine | 70+ | all losses with dynamic balancing |

### 8.3 Segmentation at Tiny Scale

For 2 to 5 pixel objects, segmentation should be treated as localization support and downstream geometry, not semantic shape recovery. Use masks when they are physically meaningful. Otherwise, output:

- Center point.
- Bounding box.
- Soft occupancy map.
- Track-level object extent estimate.

## 9. Training Data and Annotation Strategy

### 9.1 Label Quality

Tiny-object performance is usually limited by labels before architecture. A 1-pixel annotation error on a 2-pixel object is a major relative error.

Requirements:

- Store labels in floating-point image coordinates.
- Preserve original image dimensions and camera metadata.
- Flag labels smaller than 5 pixels for specialized QA.
- Prefer center-point labels plus extent uncertainty when boxes are ambiguous.
- Maintain ignored regions for ambiguous dust, glare, compression noise, and reflections.

### 9.2 Data Mix

Use a balanced mixture:

- Real labeled micro-object data.
- Real unlabeled video for self-supervised pretraining and temporal consistency.
- Synthetic micro-objects inserted with physically plausible blur, noise, lighting, and occlusion.
- Hard negative scenes with no target objects.
- Open-vocabulary region-text or image-caption data.
- Segmentation data from related domains.

### 9.3 Augmentation

Use:

- Copy-paste tiny instances with sensor-realistic blending.
- Motion blur and defocus blur matching camera exposure.
- Low-light noise, rolling shutter, compression, glare, rain/fog/dust.
- Scale jitter that does not erase tiny objects.
- Mosaic only when it preserves the final object size. Standard mosaic can accidentally make a 4-pixel object subpixel.

Avoid:

- Aggressive random resize without object-size constraints.
- Strong sharpening that creates artificial dots.
- Cutout that disproportionately removes tiny positives.
- Label-preserving augmentations that move objects below 1 pixel.

### 9.4 Curriculum

Train in this order:

1. Normal and small objects at moderate resolution.
2. Add micro-object oversampling and high-resolution tiles.
3. Add hard negatives and clutter.
4. Add segmentation.
5. Add open-vocabulary embedding.
6. Add quantization-aware training and edge calibration.

## 10. Inference Pipeline for Edge Devices

### 10.1 Accuracy-First Pipeline

```text
Frame N arrives
  -> preprocess on ISP/GPU
  -> build overlapping tiles
  -> run YOLO-Micro in FP16/INT8 mixed precision
  -> decode dense heads on device
  -> low-threshold candidate preservation
  -> tile merge with WBF/NMS
  -> temporal tracking and confidence smoothing
  -> optional refinement for uncertain high-risk candidates
  -> publish final perception message
```

Use low thresholds before merging. Small-object confidence is often low per frame but stable over tracks.

### 10.2 Quantization

Recommended:

- Quantization-aware training, not post-training quantization only.
- Per-channel weight quantization.
- Per-tensor or per-channel activation quantization depending on hardware support.
- Keep first stem, P1/P2 fusion, box distribution head, and mask boundary head in FP16 or higher precision if INT8 hurts recall.
- Use INT8 for deeper backbone, neck middle layers, and closed-set classifier when calibrated.

Do not quantize subpixel regression blindly. For 2-pixel targets, one quantization step can be a large fraction of the object.

### 10.3 Hardware-Specific Optimization

NVIDIA Jetson or discrete edge GPU:

- Export ONNX with static tile shapes.
- Use TensorRT FP16 baseline, then INT8 with QAT.
- Fuse Conv-BN-Activation.
- Use TensorRT plugins or custom kernels for DFL decode, NMS/WBF, and mask assembly.
- Run preprocessing, inference, and postprocessing in separate CUDA streams.

ARM CPU plus NPU:

- Replace unsupported activations with supported equivalents only after accuracy testing.
- Benchmark depthwise convolution. Some NPUs handle it well; some do not.
- Prefer static shapes and fixed tile counts.
- Keep fallback CPU path for postprocessing but avoid CPU-GPU copies.

Qualcomm/Android-class edge:

- Use fixed operator sets supported by the delegate.
- Avoid exotic deformable layers in the production path unless a plugin exists.
- Consider a Micro-S variant with P1 refinement disabled and stronger temporal accumulation.

### 10.4 Batching

Batch tiles, not unrelated frames, when latency matters. For a single camera:

- Batch adjacent tiles from the same frame.
- Keep temporal order intact.
- Use micro-batches sized to saturate the accelerator without exceeding latency budget.

For multi-camera systems:

- Batch by shared resolution and exposure mode.
- Keep per-camera calibration and thresholds separate.
- Do not let high-frame-rate easy cameras starve low-frame-rate safety-critical cameras.

## 11. Phased Roadmap

### Phase 1: Core Architecture and Small-Object Detection

Objective: prove that the architecture can detect 2 to 5 pixel objects at native sensor resolution with acceptable false positives.

Build:

- YOLO-Micro-B backbone with P1/P2 preservation.
- Weighted BiFPN/PAN neck with MicroFusion.
- Anchor-free detection head at P1/P2/P3/P4/P5.
- Size-aware dynamic assignment.
- NWD, DFL, subpixel, and small-object-weighted losses.
- Native-resolution tiled dataloader and inference harness.
- Micro-object evaluation suite.

Milestones:

| Milestone | Exit criteria |
| --- | --- |
| P1.1 Data audit | Labels are float-coordinate, tiny-object bins are measured, ignored regions are defined. |
| P1.2 Baseline detector | P2/P3 detector runs on native tiles and beats standard YOLO baseline on 2 to 16 px recall. |
| P1.3 P1/P2 fusion | MicroFusion improves 2 to 5 px recall without unacceptable false positives. |
| P1.4 Edge prototype | FP16 model runs on target edge hardware with measured memory and latency. |

Acceptance metrics:

- Recall for 2 to 5 pixel objects at fixed false positives per megapixel.
- AP by object diameter: 2 to 5, 6 to 16, 17 to 64, >64 pixels.
- Localization error in pixels, not only IoU.
- Track-level recall over short video clips.
- Latency per frame and per tile on target hardware.

Trade-off recommendation:

- Spend compute on input resolution and P2 quality before adding exotic attention. In field systems, preserving pixels beats clever late-stage modeling.

### Phase 2: Multi-Task Integration

Objective: add segmentation and classification without regressing tiny-object detection.

Build:

- Mask prototype branch at P2/P3.
- P1 boundary refinement.
- Image-level classification branch.
- Object-level closed-set classifier.
- Task-specific adapters and gradient balancing.
- Unified output schema.

Milestones:

| Milestone | Exit criteria |
| --- | --- |
| P2.1 Mask prototype | Normal and small-object masks work with less than 3 percent detection recall regression. |
| P2.2 Boundary refinement | P1 boundary branch improves mask edges without increasing false dots. |
| P2.3 Classification | Closed-set classifier meets baseline mAP/top-k targets while preserving objectness recall. |
| P2.4 Joint training | Dynamic loss balancing is stable across domains and object sizes. |

Acceptance metrics:

- Detection recall regression from Phase 1 less than 3 percent relative.
- Mask AP for objects larger than 8 pixels.
- Pixel/point localization quality for 2 to 5 pixel objects.
- Closed-set classification mAP and calibration error.
- Runtime increase less than the agreed budget, typically 15 to 25 percent.

Trade-off recommendation:

- Treat segmentation as optional detail for micro objects. Do not sacrifice detection recall to produce visually pleasing masks where the sensor does not contain shape information.

### Phase 3: Zero/Few-Shot and Novel Category Optimization

Objective: support novel category detection and rapid adaptation while preserving closed-set and micro-object performance.

Build:

- Region embedding projector.
- Text and exemplar prototype registry.
- Open-vocabulary scoring path.
- Unknown-object scoring.
- Episodic few-shot training.
- Adapter-only fine-tuning workflow.
- Domain calibration toolkit.

Milestones:

| Milestone | Exit criteria |
| --- | --- |
| P3.1 Embedding alignment | Region embeddings separate known classes and match text/exemplar prototypes. |
| P3.2 Zero-shot path | Registered text categories produce usable detections with calibrated thresholds. |
| P3.3 Few-shot path | 1 to 20 examples improve novel-class precision without full retraining. |
| P3.4 Domain validation | Model holds performance across target domains with calibration only. |

Acceptance metrics:

- Zero-shot recall and precision at operational false-positive rate.
- Few-shot improvement curve for 1, 5, 10, 20, and 50 examples.
- Unknown-object detection rate.
- Closed-set regression after embedding integration.
- Domain-shift robustness across camera, lighting, weather, and industrial condition splits.

Trade-off recommendation:

- Keep the open-vocabulary path as a calibrated companion to objectness, not a replacement for validated closed-set classes. Closed-set heads are still more reliable for safety-critical known categories.

## 12. Evaluation Protocol

Report metrics by object size and domain. A single mAP number will hide the failures that matter.

Required slices:

- Object diameter: 2 to 5, 6 to 16, 17 to 64, >64 pixels.
- Domain: sensor family, lighting, weather, operating site, camera height/angle.
- Motion: static, slow, fast, blurred.
- Occlusion: none, partial, heavy.
- Category state: known, zero-shot novel, few-shot novel, unknown.

Required metrics:

- Recall at fixed false positives per megapixel.
- AP and AR by size bin.
- Pixel localization error.
- Track-level recall and time-to-detect.
- Calibration error for closed-set and open-vocabulary classes.
- Latency p50/p90/p99 on hardware.
- Energy per frame when battery or thermal limits matter.

Use operational false positive units. In industrial automation, false positives per hour or per megapixel are often more meaningful than COCO-style AP.

## 13. Practical Trade-Off Recommendations

1. Prioritize native resolution over model novelty.
   If the object is gone before the backbone sees it, no head design will recover it.

2. Use P2 as the main micro-object level and P1 as a refinement path.
   A full P1 detector is expensive. Use it only when hardware and false-positive testing justify it.

3. Prefer anchor-free assignment.
   It is more robust to fractional centers, label noise, and novel shapes at extreme scales.

4. Keep objectness class-agnostic.
   This is the foundation for novel-category detection and unknown-object handling.

5. Add open-vocabulary embeddings after tiny-object detection is stable.
   Embedding losses can degrade localization if introduced too early.

6. Use temporal evidence whenever available.
   In autonomous systems, a 2-pixel object is often weak in one frame but meaningful over several frames.

7. Protect early layers during quantization.
   INT8 everywhere is attractive on paper and risky for subpixel localization.

8. Validate on hardware early.
   Operator support, memory bandwidth, and postprocessing overhead decide whether the architecture is deployable.

9. Invest in labels and hard negatives.
   At 2 to 5 pixels, bad labels and unmodeled background clutter dominate error.

10. Keep a safety margin in latency.
    Thermal throttling, multi-camera load, and postprocessing spikes are normal in field deployments.

## 14. Major Risks and Mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Label noise at 2 to 5 px | Unstable training, poor localization | Float labels, QA tooling, point labels with uncertainty. |
| False positives from texture/noise | Operational overload | Hard negatives, temporal confirmation, calibrated thresholds. |
| P1/P2 memory cost | Edge deployment failure | Narrow P1 branch, dynamic tiling, Micro-S variant. |
| Multi-task gradient conflict | Detection recall regression | Task adapters, staged training, gradient norm caps. |
| Open-vocabulary overconfidence | Novel false alarms | Prototype calibration, negative prompts, unknown thresholding. |
| Quantization regression | Missed micro objects | QAT, mixed precision, protect early/regression layers. |
| Domain shift | Field failure | Domain randomization, unlabeled pretraining, calibration per sensor family. |

## 15. Implementation Skeleton

Minimum module structure:

```text
models/
  backbone/yolo_micro.py
  neck/micro_bifpn.py
  heads/unified_anchor_free.py
  heads/mask_prototype.py
  heads/open_vocab_embedding.py
losses/
  tiny_box_loss.py
  nwd_loss.py
  multitask_balancer.py
data/
  native_tile_dataset.py
  tiny_object_augment.py
  fewshot_episode_sampler.py
deploy/
  export_onnx.py
  tensorrt_build.py
  quant_qat.py
eval/
  micro_object_metrics.py
  domain_slice_report.py
```

Initial engineering sequence:

1. Implement native tile dataloader and micro-object metrics before changing the model.
2. Build YOLO-Micro-B P2/P3 detector and compare to a standard YOLO baseline.
3. Add P1 sidecar and MicroFusion.
4. Add NWD/subpixel losses and size-aware assignment.
5. Validate edge FP16 latency.
6. Add segmentation and classification.
7. Add embedding branch and prototype registry.
8. Run QAT and mixed-precision hardware validation.

## 16. Final Recommendation

Build this as a staged perception system, not a single heroic model change. The critical path is:

```text
native pixels -> P1/P2 preservation -> stable tiny-object loss/assignment
-> multi-task heads -> open-vocabulary embeddings -> edge QAT
```

Do not start with zero-shot learning or segmentation polish. Start by proving that the detector can see 2 to 5 pixel objects at an operational false-positive rate on real native-resolution data. Once that is stable, the multi-task and novel-category systems have a solid objectness foundation to build on.
