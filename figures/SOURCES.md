# Figure provenance

One row per figure used in the notes. This file is the authoritative record; the
short credit in each figure caption is a summary of the row here.

**Status column:**

- **verified** — the figure was compared directly against the primary source
  (the paper PDF or the CS231n slide deck) and matches. The comparison is
  reproducible: fetch the source listed and look at the page given.
- **attributed** — the figure is a recognisable, canonical illustration of the
  cited work, but the specific bitmap in this repository has not been matched
  against the publication, so it may be a redraw or a copy passed through an
  intermediary.
- **unidentified** — the origin is not known. These are _not_ claimed as this
  project's own work; see the note at the bottom.

Diagrams in `diagram_*.svg` are drawn for this project (TikZ source beside each
one) and are the only figures covered by the MIT licence.

## Lecture 1 — Introduction

Figures used by `lectures/01-introduction.qmd`. The rows below supersede the
Lecture 1.1 / 1.2 rows further down, which now describe assets that are staged
for later chapters rather than in use.

| File | Status | Source |
| :--- | :----- | :----- |
| `diagram_scope.svg` | own work | Drawn for this project. Redraws the AI/ML/CV/DL Venn diagram from CS231n 2025 Lecture 1 Part 1, slide 8. Previously `01-history/diagram_01.svg`. |
| `diagram_projection.svg` | own work | Drawn for this project. Shows that all world points on one ray through the optical centre share an image coordinate, illustrating @eq-projection. Hand-authored SVG; no TikZ source. |
| `diagram_ilsvrc_error.svg` | own work | Drawn for this project. Winning top-5 error on ILSVRC, 2010–2015, plotted from the official challenge results at <https://www.image-net.org/challenges/LSVRC/>: 2010 NEC-UIUC 0.28191; 2011 XRCE 0.25770; 2012 SuperVision 0.15315 with the best non-network entry (ISI) at 0.26172; 2013 Clarifai 0.11197; 2014 GoogLeNet 6.66%. The 2015 figure (3.57%) is from the ResNet paper, and the 5.1% human annotator figure from Russakovsky et al., IJCV 2015. Plotted from published numbers, not traced from any existing chart. |
| `diagram_compute_cost.svg` | own work | Drawn for this project. FP32 throughput divided by launch price for six consumer NVIDIA cards (8800 GTX 2006, GTX 580 2010, GTX 780 Ti 2013, GTX 1080 Ti 2017, RTX 2080 Ti 2018, RTX 3090 2020). Throughput is the vendor single-precision figure; price is launch MSRP in nominal US dollars, **not** inflation-adjusted — the caption says so. The CS231n decks carry a similar chart; this one is computed from specifications rather than traced from theirs. |
| `diagram_four_tasks.svg` | own work | Drawn for this project. Classification, semantic segmentation, object detection and instance segmentation on one schematic scene. Deliberately not sourced from a slide figure: the widely circulated cat/dog/duck version of this comparison is a CS231n deck figure, and copies of it on third-party sites are re-hosts with no traceable licence. |
| `diagram_linear_classifier.svg` | own work | Drawn for this project (linear classifier and margins). Previously `01b-course-overview/diagram_01.svg`. |
| `camera_obscura_1545.jpg` | verified | Gemma Frisius, _De Radio Astronomico et Geometrico Liber_ (1545). Public domain; via Wikimedia Commons. The same engraving appears on CS231n 2025 Lecture 1 Part 1 slide 16, also marked public domain. |
| `roberts_1963_blocks.avif` | attributed | Scan of Pictures 3A–3D from L. G. Roberts, _Machine Perception of Three-Dimensional Solids_, PhD thesis, MIT, 1963. <https://dspace.mit.edu/handle/1721.1/6125> |
| `neocognitron_lenet.avif` | attributed | Composite. (a) after K. Fukushima, "Neocognitron", _Biological Cybernetics_ 36:193–202, 1980. (b) a colourised redraw of Figure 2 of Y. LeCun et al., _Proc. IEEE_ 86(11), 1998. |


Figures stored as `.avif` keep a PNG or JPEG sibling beside them purely so the PDF
build has something LaTeX can embed; `filters/pdf-images.lua` substitutes it for the
latex writer. The sibling is not published to the site and no browser fetches it.

### Deliberately not carried over

`alexnet_2012.png` reproduces Figure 2 of the AlexNet paper. The CS231n decks use
it under a permission granted to that course, which does not extend to this
repository, so it is not used here — the same ground is covered by
`neocognitron_lenet.avif` and by prose. `backprop_1986.png` is a figure from a
_Nature_ paper and carries the same problem. The `unidentified` figures from the
old Lecture 1.1 table (`hubel_wiesel_1959.png`, `pictorial_structures.png`,
`canny_edge_detection.png`, `sift_features.jpg`, `viola_jones.png`) are untraced
and were not carried over; where the chapter needs them, replacements must be
sourced or drawn.

## Lecture 7 — Recurrent neural networks

Figures used by `lectures/07-recurrent-neural-networks.qmd`. All six are generated
by `figures/07-recurrent-neural-networks/make_diagrams.py`, which is committed
beside them and is the authoritative source: running it from the repository root
rewrites all six and prints the gradient figures the chapter quotes. Nothing here
is traced from a slide.

| File | Status | Source |
| :--- | :----- | :----- |
| `diagram_seq_shapes.svg` | own work | Drawn for this project. Five panels showing the input/output patterns a sequence problem can take, generated from the `SS_PANELS` table (input steps, output steps, band span) rather than placed by hand. The CS231n decks carry a comparable five-panel figure in a different colour scheme; this one is laid out, coloured and captioned independently, and the aligned many-to-many panel is drawn as a distinct case rather than folded into the fourth. |
| `diagram_unrolled.svg` | own work | Drawn for this project. The recurrence in loop notation beside the same computation unrolled over four steps, with the shared weight drawn as one node on a rail feeding every step — the detail that makes @eq-bptt-sum visible, and the reason the figure exists. |
| `diagram_truncated_bptt.svg` | own work | Drawn for this project. Nine steps in three chunks, with the hidden state crossing every chunk boundary and the gradient stopping at each one. Chunk width, step count and the arrow pattern are generated from `TB_N` and `TB_CHUNK`. |
| `diagram_gradient_norm.svg` | own work | **Computed** for this project. The geometric factor of @eq-jacobian-product evaluated for 150 steps at largest singular values of 1.05, 1.00 and 0.95, plus a fourth curve at 1.05 with a mean $\tanh'$ of 0.85 folded in. The script produces ×1508, ÷2195 and ÷26 million after 150 steps, and those are the numbers the figure and the chapter state. These are exact evaluations of a stated model, not measurements of a trained network, and neither the caption nor the prose claims otherwise. |
| `diagram_lstm_cell.svg` | own work | Drawn for this project. One LSTM step: the four gates from a single weight matrix, the two elementwise multiplies and the addition, and the cell state drawn as an uninterrupted path. Structurally equivalent to figures in the CS231n decks and in much of the literature because all of them draw @eq-lstm-gates and @eq-lstm-state; the layout is ordered strictly left to right so that no signal path doubles back, and every coordinate is this project's own. |
| `diagram_captioning.svg` | own work | Drawn for this project. A tapering convolutional stack whose penultimate vector conditions every step of a recurrence, with the previously emitted word as the input at each step. Deliberately not sourced from the deck's version, which reproduces a figure from Karpathy and Fei-Fei, CVPR 2015, copyright IEEE, under a permission granted to that course and not to this repository. |

## Staged assets — Lecture 1.1 (superseded)

| File                             | Status       | Source                                                                                                                                                                                                                                                                                        |
| :------------------------------- | :----------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `diagram_01.svg`                 | own work     | Drawn for this project. Redraws the AI/ML/CV/DL Venn diagram from CS231n 2025 Lecture 1 Part 1, slide 8.                                                                                                                                                                                      |
| `camera_obscura_1545.jpg`        | verified     | Gemma Frisius, _De Radio Astronomico et Geometrico Liber_ (1545). Public domain; via Wikimedia Commons. The same engraving appears on CS231n 2025 Lecture 1 Part 1 slide 16, also marked public domain.                                                                                       |
| `roberts_1963_blocks.avif`        | attributed   | Scan of Pictures 3A–3D from L. G. Roberts, _Machine Perception of Three-Dimensional Solids_, PhD thesis, MIT, 1963. <https://dspace.mit.edu/handle/1721.1/6125> (the same thesis is cited on slide 21 of the 2025 deck).                                                                      |
| `rosenblatt_perceptron_1958.png` | attributed   | Composite. Left: archival photograph of Frank Rosenblatt with the Mark I Perceptron (Cornell University). Right: "Figure 1: Organization of the Mark I Perceptron", from the Mark I Perceptron report, Cornell Aeronautical Laboratory.                                                       |
| `minsky_papert_1969.jpg`         | attributed   | Composite of figures from, and the cover of, M. Minsky & S. Papert, _Perceptrons: An Introduction to Computational Geometry_, MIT Press, 1969.                                                                                                                                                |
| `backprop_1986.png`              | attributed   | The XOR network figure from D. Rumelhart, G. Hinton & R. Williams, "Learning representations by back-propagating errors", _Nature_ 323:533–536, 1986. <https://doi.org/10.1038/323533a0>                                                                                                      |
| `lenet_1998.png`                 | attributed   | Colourised redraw of Figure 2 of Y. LeCun, L. Bottou, Y. Bengio & P. Haffner, "Gradient-Based Learning Applied to Document Recognition", _Proc. IEEE_ 86(11), 1998. The 2025 deck labels this same lineage "Illustration of LeCun et al. 1998 from CS231n 2017 Lecture 1" (slide 16, Part 2). |
| `neocognitron_lenet.avif`         | attributed   | Composite. (a) after K. Fukushima, "Neocognitron", _Biological Cybernetics_ 36:193–202, 1980. (b) the same colourised LeCun et al. (1998) redraw as `lenet_1998.png`.                                                                                                                         |
| `imagenet_mosaic.jpg`            | attributed   | ImageNet promotional mosaic, <https://image-net.org>. Dataset: J. Deng et al., "ImageNet: A Large-Scale Hierarchical Image Database", CVPR 2009.                                                                                                                                              |
| `alexnet_2012.png`               | attributed   | Figure 2 of A. Krizhevsky, I. Sutskever & G. Hinton, "ImageNet Classification with Deep Convolutional Neural Networks", NeurIPS 2012. Note: the CS231n decks reproduce this figure "with permission" granted to that course; no such permission extends to this repository.                   |
| `hubel_wiesel_1959.png`          | unidentified | Line drawing of the Hubel & Wiesel recording setup. Widely reproduced; the specific drawing is _not_ the one used in the 2025 deck (slide 19), which is a different redraw. Underlying work: D. Hubel & T. Wiesel, _J. Physiol._ 148:574–591, 1959.                                           |
| `pictorial_structures.png`       | unidentified | Pose-estimation results on a grayscale image sequence. Probably from P. Felzenszwalb & D. Huttenlocher, "Pictorial Structures for Object Recognition", _IJCV_ 61(1), 2005, but not confirmed. Not the figure used on slide 23 of the 2025 deck.                                               |
| `canny_edge_detection.png`       | unidentified | Canny edges on the standard "cameraman" test image. The matplotlib axis ticks show it is a generated plot rather than a published figure; who generated it is unknown. Underlying method: J. Canny, _IEEE TPAMI_ 8(6), 1986.                                                                  |
| `sift_features.jpg`              | unidentified | SIFT keypoint matching on a toy truck. Underlying work: D. Lowe, "Object recognition from local scale-invariant features", ICCV 1999.                                                                                                                                                         |
| `viola_jones.png`                | unidentified | Five-panel composite (integral image, Haar features, cascade, faces, LBP). The inclusion of Local Binary Patterns means it is not from the original paper; it appears to come from a later survey. Underlying work: P. Viola & M. Jones, CVPR 2001.                                           |

## Staged assets — Lecture 1.2 (superseded)

| File                              | Status       | Source                                                                                                                                                                                                                                                                                                                     |
| :-------------------------------- | :----------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `diagram_01.svg`                  | own work     | Drawn for this project (linear classifier / margins).                                                                                                                                                                                                                                                                      |
| `diagram_02.svg`                  | own work     | Drawn for this project (regularisation and overfitting).                                                                                                                                                                                                                                                                   |
| `semantic_segmentation.jpg`       | verified     | The CS231n cat/cow segmentation figure; the same cat photograph and mask appear on CS231n 2025 Lecture 1 Part 2, slide 14. That deck credits the photograph to Nikita under CC BY 2.0 (slide 5).                                                                                                                           |
| `style_transfer.jpg`              | verified     | Same content as Figure 2C of L. Gatys, A. Ecker & M. Bethge, "A Neural Algorithm of Artistic Style", arXiv:1508.06576, 2015 — compared against page 5 of that paper. Content photograph: the Neckarfront in Tübingen, by Andreas Praefcke, as credited in the paper. _The Starry Night_ (Van Gogh, 1889) is public domain. |
| `diffusion_process.png`           | verified     | Reproduces the structure and notation of Figure 2 of J. Ho, A. Jain & P. Abbeel, "Denoising Diffusion Probabilistic Models", arXiv:2006.11239, 2020 — compared against page 2. The sample thumbnails differ, so this is a redraw rather than the paper's own bitmap.                                                       |
| `clip_architecture.jpg`           | verified     | Figure 1 of A. Radford et al., "Learning Transferable Visual Models From Natural Language Supervision", arXiv:2103.00020, 2021 — exact match against page 2.                                                                                                                                                               |
| `mae.jpg`                         | verified     | Figure 1 of K. He et al., "Masked Autoencoders Are Scalable Vision Learners", arXiv:2111.06377, 2021 — exact match against page 1.                                                                                                                                                                                         |
| `stable_diffusion_examples.jpg`   | verified     | Figure 1 of B. Kawar et al., "Imagic: Text-Based Real Image Editing with Diffusion Models", arXiv:2210.09276, 2022 — exact match against page 1. Imagic builds on Imagen, not Stable Diffusion.                                                                                                                            |
| `simclr.png`                      | attributed   | Diagram of the SimCLR framework of T. Chen et al., "A Simple Framework for Contrastive Learning of Visual Representations", arXiv:2002.05709, 2020. The rendering is _not_ the paper's Figure 2; it is a third-party redraw whose author is unknown. The 2025 deck credits a related figure on slide 22 to Rohit Kundu.    |
| `parallelism.png`                 | attributed   | The same data-parallel / model-parallel illustration used on CS231n 2025 Lecture 1 Part 2, slide 18, in different colours — so both are probably recolourings of a common upstream diagram that has not been traced.                                                                                                       |
| `neural_network.png`              | unidentified | Generic multi-layer perceptron diagram.                                                                                                                                                                                                                                                                                    |
| `object_detection.jpg`            | unidentified | YOLO detections on a desk scene. Underlying method: J. Redmon et al., "You Only Look Once", CVPR 2016.                                                                                                                                                                                                                     |
| `instance_segmentation.jpg`       | unidentified | Instance masks on two crowd photographs.                                                                                                                                                                                                                                                                                   |
| `video_classification_frames.jpg` | unidentified | Constructed diagram: seven frames of a runner, each through a CNN. Not the figure used on slide 15 of the 2025 deck.                                                                                                                                                                                                       |
| `multimodal_video.png`            | unidentified | Uni-modal / multi-modal video understanding pipeline with RGB, audio and ASR streams.                                                                                                                                                                                                                                      |
| `cnn_detailed.jpg`                | unidentified | Tutorial-style CNN diagram with a zebra input and Horse/Zebra/Dog output. Not the figure used on slide 16 of the 2025 deck.                                                                                                                                                                                                |
| `cnn_layer_activations.jpg`       | unidentified | Conv1–Conv5 activation heatmaps over video frames. Also used as the banner image in `README.md`.                                                                                                                                                                                                                           |
| `rnn_diagram.png`                 | unidentified | Generic recurrent-network diagram. Not the figure used on slide 17 of the 2025 deck.                                                                                                                                                                                                                                       |
| `vit_attention.jpg`               | unidentified | Grid of images with per-head ViT attention maps. Checked against DINO (M. Caron et al., arXiv:2104.14294) page 1 and it is not that figure.                                                                                                                                                                                |
| `gan_architecture.png`            | unidentified | Generator/discriminator diagram with MNIST digits. Underlying work: I. Goodfellow et al., "Generative Adversarial Networks", NeurIPS 2014.                                                                                                                                                                                 |
| `vqa_example.jpg`                 | unidentified | VQA-with-attention pipeline, "What color shirt is the referee wearing?".                                                                                                                                                                                                                                                   |
| `voxel_reconstruction.jpg`        | unidentified | Single-image 3D reconstruction of indoor scenes with an input / methods / ground-truth row layout. Checked against 3D-R2N2 (C. Choy et al., arXiv:1604.00449), which the 2025 deck cites on slide 27, and it is not that figure.                                                                                           |
| `shape_completion.jpg`            | unidentified | Blue meshes (fox, bear, chair, dragon) shown partial and completed. Checked against Point-Voxel Diffusion (L. Zhou et al., arXiv:2104.03670) page 7 and it is not that figure.                                                                                                                                             |
| `3d_object_detection.png`         | unidentified | LiDAR point-cloud visualisation for autonomous driving.                                                                                                                                                                                                                                                                    |
| `embodied_ai_loop.png`            | unidentified | Sensors → representation → AI/ML → action loop diagram.                                                                                                                                                                                                                                                                    |
| `robot_tidying.jpg`               | unidentified | Photograph of a humanoid robot being teleoperated by a person in a VR headset. Appears to be a press photograph.                                                                                                                                                                                                           |
| `ai_medical_xray.jpg`             | unidentified | Four chest X-rays comparing heatmap visualisation styles (A–D).                                                                                                                                                                                                                                                            |

## Site assets

| File              | Status               | Source                                                                                                                                                                                                                                                                                                          |
| :---------------- | :------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `social-card.png` | own work (composite) | The 1280×640 Open Graph / Twitter card set in `_quarto.yml`. Layout, type and colours drawn for this project. Its background, however, is a blurred and dimmed tiling of `cnn_layer_activations.jpg`, so the `unidentified` status of that figure applies to the card as well until the background is replaced. |

Because the card is what X, Slack, Discord and GitHub display for every shared
link, its background should be swapped for one of the `diagram_*.svg` drawings or
a flat pattern; that is the one place where an untraced bitmap is served as this
project's own public face.

## Figures no longer in the repository

`cat_example.jpg`, `cnn_architecture.png`, `rnn_transformer.png` and
`temporal_tasks.jpg` were present but referenced by no lecture, and have been
deleted.

Three of them — `cnn_architecture.png`, `rnn_transformer.png` and
`temporal_tasks.jpg` — were verbatim screen captures of CS231n 2025 Lecture 1
Part 2 slides 16, 17 and 15, footer (_"Stanford CS231n 10th Anniversary · April
1, 2025"_) included. Redistributing whole slides is exactly what keeping
`slides/` out of version control is meant to avoid, so their removal is a
licensing fix and not only housekeeping. `cat_example.jpg` was an untraced stock
photograph of a cat.

Every remaining figure was checked for that footer, by eye and not only by
script: none of the figures still in the repository is a slide capture.
`semantic_segmentation.jpg` is the closest case — it is the CS231n cat/cow
figure, but cropped to the figure itself, and the deck credits the underlying
photograph to Nikita under CC BY 2.0.

## On the unidentified figures

These were collected while drafting the notes without a record of where each one
came from, and the record cannot be reconstructed after the fact: the files carry
no metadata, they were all added in a single commit, and — with the exception of
`camera_obscura_1545.jpg` and `semantic_segmentation.jpg` — none of them is the
bitmap used in the CS231n slides, so the decks' own credit lines do not apply to
them. Guessing a plausible-looking citation would be worse than saying nothing,
because a wrong credit reads as a verified one.

The intended fix is to replace them rather than to keep hunting for sources:
schematic diagrams (`gan_architecture`, `diffusion_process`, `rnn_diagram`,
`neural_network`, `embodied_ai_loop`, `parallelism`, `cnn_detailed`) can be
redrawn in TikZ or Mermaid like the existing `diagram_*.svg`, which also fixes
their weight and their poor legibility in dark mode. Figures that show results or
data (`vit_attention`, `imagenet_mosaic`, `ai_medical_xray`) cannot be redrawn and
need either a traced source or removal.

If you recognise one of these figures, please
[open an issue](https://github.com/raimbekovm/cs231n-2025-notes/issues).

## Lecture 2 — Image classification

| File | Status | Source |
| :--- | :----- | :--- |
| `02-image-classification/diagram_model_loss.svg` | own work | Drawn for this project. Shows the separation between input representation, the linear score function, class scores, and the loss used for training. |

## Lecture 3 — Regularization and optimization

All four are own work with no external source. They are generated rather than hand
authored: `03-regularization-optimization/make_diagrams.py` sits beside them, is the
authoritative source, and rewrites all four when run from the repository root. Every
curve in these figures is evaluated numerically, so what is drawn is what the stated
formula or simulation produces, not an impression of it.

| File | Status | Source |
| :--- | :----- | :--- |
| `03-regularization-optimization/diagram_overfitting.svg` | own work | Drawn for this project. Nine points on the line $y = 1.2111 + 0.5667x$ with hand-chosen residuals, the degree-eight Lagrange interpolant through all nine, and the least-squares line fitted to them. Both curves are computed from the plotted points. |
| `03-regularization-optimization/diagram_sgd_problems.svg` | own work | Drawn for this project. Left: exact level sets of $f = \tfrac12(w_1^2 + 20w_2^2)$, with gradient descent from $(4.6, 1.25)$ at step size $0.09$ run for 26 iterations. Right: the two principal cross-sections $\pm 0.34u^2$ through a saddle. |
| `03-regularization-optimization/diagram_lr_curves.svg` | own work | Drawn for this project. SGD on $f = \tfrac12(0.25w_1^2 + 8w_2^2)$ from $(5.0, 0.9)$ with uniform additive gradient noise, ten updates per epoch, plotting the mean loss per epoch at step sizes $0.252$, $0.005$, $0.16$ and $0.06$. Randomness comes from a linear congruential generator seeded in the script, so the figure is reproducible. |
| `03-regularization-optimization/diagram_lr_schedules.svg` | own work | Drawn for this project. Step, cosine, linear and inverse-square-root decay evaluated over 100 epochs, with a five-epoch linear warmup on the cosine curve. |

## Lecture 4 — Neural networks and backpropagation

Figures used by `lectures/04-neural-networks-backpropagation.qmd`. All five are
generated by `figures/04-neural-networks-backpropagation/make_diagrams.py`,
which is committed beside them and is the authoritative source: running it from
the repository root rewrites all five. Nothing here is traced from a slide, and
every annotated gradient is the number the chain rule produces for the stated
inputs, checked by hand.

| File | Status | Source |
| :--- | :----- | :----- |
| `diagram_network.svg` | own work | Drawn for this project. Two fully connected networks — 4→5→3 and 4→5→5→3 — drawn as complete bipartite graphs between consecutive layers, to show the layer-counting convention of @sec-hidden-layer. Unit counts are illustrative, not the 3072→100→10 of the text. |
| `diagram_relu_bending.svg` | own work | Drawn for this project. Three weighted rectified units, `+1.0·max(0, x+2.5)`, `−2.2·max(0, x−0.3)` and `+2.6·max(0, x−2.2)`, plotted over x ∈ [−4, 4] together with their sum. Each unit's corner sits at its own offset, so the summed curve has exactly three joints. Evaluated at 401 sample points; no curve is sketched. |
| `diagram_activations.svg` | own work | Drawn for this project. ReLU, leaky ReLU, GELU, tanh and sigmoid over z ∈ [−3, 3], evaluated at 601 points. GELU is `z·Φ(z)` with Φ from `math.erf`. **Leaky ReLU is drawn with α = 0.1, not the α = 0.01 of the text**, because at 0.01 the negative slope is under half a pixel across the plot; the caption and the SVG description both say so. |
| `diagram_backprop_graph.svg` | own work | Drawn for this project. The graph of `f = (x + y)z` at x = −2, y = 5, z = −4, with forward values above each edge and ∂f/∂edge below it: −4 on x, y and q, 3 on z, 1 on f. Verified by hand against a finite difference in the text. |
| `diagram_gradient_patterns.svg` | own work | Drawn for this project. The add, multiply, max and copy patterns with the worked numbers of @sec-gradient-patterns: add 3+4=7 with upstream 2; multiply 2×3=6 with upstream 5 giving 15 and 10; max(4,5)=5 with upstream 9 routed entirely to 5; copy of 7 receiving 4 and 2 back, summing to 6. |

## Lecture 5 — Convolutional networks

Figures used by `lectures/05-convolutional-networks.qmd`. All seven are generated
by `figures/05-convolutional-networks/make_diagrams.py`, which is committed
beside them and is the authoritative source: running it from the repository root
rewrites all seven. Nothing here is traced from a slide.

| File | Status | Source |
| :--- | :----- | :----- |
| `diagram_flatten.svg` | own work | Drawn for this project. A 4×4 image with one pixel and its four four-connected neighbours marked, beside the same sixteen pixels in row-major order. The annotated distances (1, 1, 4, 4) are the actual index differences: pixel (1,1) is index 5, its neighbours are 1, 4, 6 and 9. |
| `diagram_conv_layer.svg` | own work | Drawn for this project. A 3×32×32 input in oblique projection with one 3×5×5 filter placement, the 28×28 activation map it fills, and six such maps stacked into a 6×28×28 volume. Spatial proportions are to scale (the filter is 5/32 of the input's width); the input grid is drawn at 8×8 rather than 32×32 for legibility, which the caption does not claim otherwise. |
| `diagram_gabor.svg` | own work | **Computed** for this project, not sampled from any trained model. Rows 1–6 are Gabor functions `exp(−(x'² + γ²y'²)/2σ²)·cos(2πx'/λ + φ)` with σ = 1.15, γ = 0.72, at eight orientations kπ/8, three wavelengths (1.3, 1.9, 2.7) and two phases (0, π/2); rows 7–8 are differences of Gaussians at four scales, both polarities, and two centre offsets. Each patch is evaluated on an 11×11 lattice over [−2.5, 2.5]² and normalised to its own peak, then mapped through the viridis ramp. Deliberately **not** the real AlexNet first layer: that grid is Figure 3 of Krizhevsky et al. (2012) and falls under the same restriction as Figure 2, recorded above. The caption states that these are analytic rather than learned. |
| `diagram_padding_stride.svg` | own work | Drawn for this project. A 7×7 input with a 3×3 filter in three configurations, with every valid placement outlined. The placement counts (5, 7, 3) are enumerated by the script from `range(0, W + 2P − K + 1, S)`, not asserted. |
| `diagram_receptive_field.svg` | own work | Drawn for this project. Three stacked 3-wide convolutions in one spatial dimension, 7→5→3→1, with every dependency edge drawn from the actual index arithmetic. Matches `1 + L(K − 1) = 7` for L = 3, K = 3. |
| `diagram_pooling.svg` | own work | Drawn for this project. The 4×4 plane of @eq-maxpool, with the maximum of each 2×2 tile highlighted by comparison in the script rather than marked by hand, and the 2×2 output beside it. |
| `diagram_equivariance.svg` | own work | Drawn for this project. The commutative square for `f(T(x)) = T(f(x))`, with a schematic scene shifted by the same fraction of the frame along both routes. |

## Lecture 6 — CNN architectures

Figures used by `lectures/06-cnn-architectures.qmd`. All five are generated by
`figures/06-cnn-architectures/make_diagrams.py`, which is committed beside them
and is the authoritative source: running it from the repository root rewrites all
five and prints the VGG-16 figures the chapter quotes. Nothing here is traced
from a slide.

| File | Status | Source |
| :--- | :----- | :----- |
| `diagram_norm_axes.svg` | own work | Drawn for this project. Four copies of an N×C×(H·W) activation block in oblique projection, each shading the subset one normalization layer pools. Group Normalization (Wu and He, ECCV 2018) contains a figure making the same comparison; that one is a publisher figure and is not used or traced here. The geometry below is generated from the shading rule `(row0, nrow, slice0, nslice)` per layer, and the layout, projection and labelling are this project's own. |
| `diagram_depth_error.svg` | own work | **Computed** for this project. Winning ILSVRC top-5 error, plotted from the official challenge results at <https://www.image-net.org/challenges/LSVRC/> — the same numbers as `01-introduction/diagram_ilsvrc_error.svg` — extended with 2017 (2.251%, from the SENet abstract, [arXiv:1709.01507](https://arxiv.org/abs/1709.01507)). 2016 is deliberately omitted: its winning entry is an ensemble with no single depth and no paper of its own, and the chart says so. Depths are those the winning papers report for their own entry (AlexNet 8, ZFNet 8, GoogLeNet 22, ResNet 152); the 2010 and 2011 winners are feature pipelines and are drawn at zero. The 5.1% human annotator baseline is from Russakovsky et al., IJCV 2015. |
| `diagram_vgg_budget.svg` | own work | **Computed** for this project. Activation count and parameter count per layer, evaluated by `vgg16_budget()` from configuration D of Simonyan and Zisserman ([arXiv:1409.1556](https://arxiv.org/abs/1409.1556)) on a 224×224×3 input. Nothing is quoted from a published table: the script produces 15.09M activation values (71.8% of them in the first two blocks) and 138.4M parameters (102.8M, or 74.3%, in `fc6`), and those are the numbers the chapter states. |
| `diagram_residual_block.svg` | own work | Drawn for this project. A plain two-convolution block beside the residual block of @eq-residual, with the identity path and the addition junction labelled. Structurally equivalent to Figure 2 of He et al. (2015) because both draw the same equation; the layout, the side-by-side comparison with the plain block, and every coordinate are this project's own. |
| `diagram_learning_curves.svg` | own work | **Computed** for this project. Four schematic training runs, each a saturating exponential evaluated by the script rather than sketched. These are illustrative shapes, not measured runs, and the caption does not claim otherwise. |

## Lecture 8 — Attention and transformers

Figures used by `lectures/08-attention-transformers.qmd`. All eight are generated
by `figures/08-attention-transformers/make_diagrams.py`, which is committed beside
them and is the authoritative source: running it from the repository root rewrites
all eight and prints the two figures the chapter quotes. Nothing here is traced
from a slide.

| File | Status | Source |
| :--- | :----- | :----- |
| `diagram_bottleneck.svg` | own work | Drawn for this project. The encoder–decoder model of @eq-seq2seq-decoder, with the single context vector drawn as a purple rail feeding every decoder step so that the one channel between source and output is the visible fact. Box positions come from the `BN_ENC_X`/`BN_DEC_X` tables. |
| `diagram_attention_step.svg` | own work | Drawn for this project. One decoder step of @eq-alignment-score through @eq-context-vector: the four encoder states, the score column, the softmax, the weighted sum, and the resulting context vector entering the recurrent unit. The decoder state is drawn feeding both the scores and the recurrent unit, which is the pair of roles the equations give it. |
| `diagram_alignment.svg` | own work | **Computed** for this project, and a schematic rather than a measurement. Cell `(j, i)` is the row-normalized Gaussian `exp(−(i − m_j)² / 2τ²)` with τ = 0.62 over the stated alignment `m = [0, 2, 1, 3, 4]` for "the black cat sleeps here" → "le chat noir dort ici", mapped through the viridis ramp. It is not the attention matrix of any trained model, and the caption says so. Bahdanau et al. (2015) contains real alignment maps; that is a publisher figure and is not used or traced here. |
| `diagram_attention_shapes.svg` | own work | Drawn for this project. The layer of @eq-attention with `NQ × D`, `NX × D`, `NQ × NX` and `NQ × DV` marked at each stage. The shading inside the score and weight grids is decorative rather than data; the weights grid uses rows that sum to one so the normalization direction reads correctly. |
| `diagram_masking.svg` | own work | **Computed** for this project. The left grid is the causal mask of @eq-mask applied to a score pattern `0.9 − 0.35·|i − j|`; the right grid is that pattern's row-wise softmax, evaluated by `masked_weights()` and shaded through the viridis ramp, so the zeros above the diagonal and the row sums are produced by the script rather than asserted. |
| `diagram_three_ways.svg` | own work | Drawn for this project. Three connectivity graphs over the same five positions: a recurrent chain, a width-3 convolution, and all-to-all attention. Every edge is enumerated from the connectivity rule (`|i − j| ≤ 1` for the convolution, all pairs for attention), so the twenty-five edges in the third panel and the four hops in the first are counted rather than sketched. The CS231n decks compare the same three primitives in a table; the drawing, the choice to show connectivity rather than prose, and the layout are this project's own. |
| `diagram_transformer_block.svg` | own work | Drawn for this project. The block of @eq-transformer-block, with the self-attention band drawn full width to show that it is the only place the vectors interact and the MLP drawn per column to show that it is not. Structurally equivalent to any drawing of the same equation; the layout, the residual routing and every coordinate are this project's own. |
| `diagram_vit.svg` | own work | Drawn for this project. Patchify, project, add positional encodings, attend, pool, classify. The quoted arithmetic — 224/16 = 14 patches per side, 196 tokens, 16·16·3 = 768 numbers per patch — is printed by the script and matches the chapter. Dosovitskiy et al. (2021) contains a figure making the same journey; that one is a publisher figure and is not used or traced here. |

## Lecture 9 — Object detection and segmentation

Figures used by `lectures/09-detection-segmentation.qmd`. All nine are generated by
`figures/09-detection-segmentation/make_diagrams.py`, which is committed beside them
and is the authoritative source: running it from the repository root rewrites all
nine and prints the transposed-convolution terms the chapter quotes. Nothing here is
traced from a slide, and no figure from any of the cited papers is reproduced —
several of them (the R-CNN and Fast R-CNN pipeline drawings, the U-Net figure, the
Mask R-CNN results) appear in the CS231n decks under permissions granted to that
course and not to this repository.

| File | Status | Source |
| :--- | :----- | :----- |
| `diagram_seg_shapes.svg` | own work | Drawn for this project. A constant-resolution fully convolutional stack above an encoder–decoder one, with each feature map's height standing for spatial size and its width for channel count so that the cost argument in @sec-semantic-segmentation is visible as geometry. Long, Shelhamer and Darrell (2015) draw the same contrast; that is a publisher figure and is not used or traced here. |
| `diagram_unpooling.svg` | own work | **Computed** for this project. Nearest neighbour and bed of nails are written out directly; the max-unpooling panel places the values 1–4 at the indices in `MAX_IDX`, each of which is constrained to lie inside the 2×2 pooling quadrant its value was pooled from, so the panel is consistent with an actual max-pool rather than decorative. |
| `diagram_transposed_conv.svg` | own work | **Computed** for this project. The output terms are produced by `transposed_1d()` from the input symbols, the filter symbols and the stride, not written out by hand: the script prints `(ax, ay, az + bx, by, bz)` and identifies position 2 as the only one reached by both filter copies, which is what @eq-transposed-1d and the caption state. |
| `diagram_unet.svg` | own work | Drawn for this project. The descending and ascending branches with a skip connection at each level, laid out so that x increases monotonically along the direction of flow. Ronneberger et al. (2015) contains the original U figure; that is a publisher figure and is not used or traced here. |
| `diagram_detector_family.svg` | own work | Drawn for this project. R-CNN, Fast R-CNN and Faster R-CNN as three aligned pipelines sharing a column grid, so that what moves inside the network at each generation is read off vertically. Dashed outlines mark the stages that are not learned. The Girshick pipeline figures reproduced in the CS231n decks are copyright the author and are not used or traced here. |
| `diagram_roi_align.svg` | own work | **Computed** for this project. Both panels place the same proposal at the real-valued grid coordinates in `BOX`; the left panel derives the snapped rectangle by flooring and ceiling those coordinates, and the right panel derives its sample points by subdividing the unsnapped box, so the displacement between the two is generated rather than drawn in by eye. |
| `diagram_yolo_grid.svg` | own work | Drawn for this project. A 7×7 grid with two base boxes on one cell, the whole prediction as an `S × S × (5B + C)` cuboid, and that cell's slice expanded into its groups. The group widths follow `S`, `B` and the number of class cells shown, and the script prints the tensor shape the chapter quotes. |
| `diagram_detr.svg` | own work | Drawn for this project. The backbone–encoder–decoder rail with object queries entering the decoder from the left, and four predictions matched one-to-one against two ground-truth objects with the remainder assigned ∅. Carion et al. (2020) contains a pipeline figure covering the same ground; that is a publisher figure and is not used or traced here. |
| `diagram_grad_cam.svg` | own work | **Computed** for this project. Two structurally identical panels differing only in where the channel weights come from, which is the chapter's claim about the relationship between @eq-cam and @eq-gradcam. The heatmap cells are shaded by interpolating the five viridis samples in `_viridis()` over the illustrative array `HEAT`; it is a schematic, not the output of any trained model, and the caption does not claim otherwise. |
