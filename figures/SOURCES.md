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

## Lecture 1.1 — History of Computer Vision

| File                             | Status       | Source                                                                                                                                                                                                                                                                                        |
| :------------------------------- | :----------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `diagram_01.svg`                 | own work     | Drawn for this project. Redraws the AI/ML/CV/DL Venn diagram from CS231n 2025 Lecture 1 Part 1, slide 8.                                                                                                                                                                                      |
| `camera_obscura_1545.jpg`        | verified     | Gemma Frisius, _De Radio Astronomico et Geometrico Liber_ (1545). Public domain; via Wikimedia Commons. The same engraving appears on CS231n 2025 Lecture 1 Part 1 slide 16, also marked public domain.                                                                                       |
| `roberts_1963_blocks.png`        | attributed   | Scan of Pictures 3A–3D from L. G. Roberts, _Machine Perception of Three-Dimensional Solids_, PhD thesis, MIT, 1963. <https://dspace.mit.edu/handle/1721.1/6125> (the same thesis is cited on slide 21 of the 2025 deck).                                                                      |
| `rosenblatt_perceptron_1958.png` | attributed   | Composite. Left: archival photograph of Frank Rosenblatt with the Mark I Perceptron (Cornell University). Right: "Figure 1: Organization of the Mark I Perceptron", from the Mark I Perceptron report, Cornell Aeronautical Laboratory.                                                       |
| `minsky_papert_1969.jpg`         | attributed   | Composite of figures from, and the cover of, M. Minsky & S. Papert, _Perceptrons: An Introduction to Computational Geometry_, MIT Press, 1969.                                                                                                                                                |
| `backprop_1986.png`              | attributed   | The XOR network figure from D. Rumelhart, G. Hinton & R. Williams, "Learning representations by back-propagating errors", _Nature_ 323:533–536, 1986. <https://doi.org/10.1038/323533a0>                                                                                                      |
| `lenet_1998.png`                 | attributed   | Colourised redraw of Figure 2 of Y. LeCun, L. Bottou, Y. Bengio & P. Haffner, "Gradient-Based Learning Applied to Document Recognition", _Proc. IEEE_ 86(11), 1998. The 2025 deck labels this same lineage "Illustration of LeCun et al. 1998 from CS231n 2017 Lecture 1" (slide 16, Part 2). |
| `neocognitron_lenet.png`         | attributed   | Composite. (a) after K. Fukushima, "Neocognitron", _Biological Cybernetics_ 36:193–202, 1980. (b) the same colourised LeCun et al. (1998) redraw as `lenet_1998.png`.                                                                                                                         |
| `imagenet_mosaic.jpg`            | attributed   | ImageNet promotional mosaic, <https://image-net.org>. Dataset: J. Deng et al., "ImageNet: A Large-Scale Hierarchical Image Database", CVPR 2009.                                                                                                                                              |
| `alexnet_2012.png`               | attributed   | Figure 2 of A. Krizhevsky, I. Sutskever & G. Hinton, "ImageNet Classification with Deep Convolutional Neural Networks", NeurIPS 2012. Note: the CS231n decks reproduce this figure "with permission" granted to that course; no such permission extends to this repository.                   |
| `hubel_wiesel_1959.png`          | unidentified | Line drawing of the Hubel & Wiesel recording setup. Widely reproduced; the specific drawing is _not_ the one used in the 2025 deck (slide 19), which is a different redraw. Underlying work: D. Hubel & T. Wiesel, _J. Physiol._ 148:574–591, 1959.                                           |
| `pictorial_structures.png`       | unidentified | Pose-estimation results on a grayscale image sequence. Probably from P. Felzenszwalb & D. Huttenlocher, "Pictorial Structures for Object Recognition", _IJCV_ 61(1), 2005, but not confirmed. Not the figure used on slide 23 of the 2025 deck.                                               |
| `canny_edge_detection.png`       | unidentified | Canny edges on the standard "cameraman" test image. The matplotlib axis ticks show it is a generated plot rather than a published figure; who generated it is unknown. Underlying method: J. Canny, _IEEE TPAMI_ 8(6), 1986.                                                                  |
| `sift_features.jpg`              | unidentified | SIFT keypoint matching on a toy truck. Underlying work: D. Lowe, "Object recognition from local scale-invariant features", ICCV 1999.                                                                                                                                                         |
| `viola_jones.png`                | unidentified | Five-panel composite (integral image, Haar features, cascade, faces, LBP). The inclusion of Local Binary Patterns means it is not from the original paper; it appears to come from a later survey. Underlying work: P. Viola & M. Jones, CVPR 2001.                                           |

## Lecture 1.2 — Course Overview

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

## Figures not currently used

`cat_example.jpg`, `cnn_architecture.png`, `rnn_transformer.png` and
`temporal_tasks.jpg` are in the repository but referenced by no lecture. They
carry the same provenance uncertainty and should be either sourced or deleted.

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
