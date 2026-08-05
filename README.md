<p align="center">
  <img src="figures/readme-hero.svg" width="820" alt="CS231n 2025 Lecture Notes — Deep Learning for Computer Vision, Stanford Spring 2025">
</p>

<h1 align="center">CS231n 2025 Lecture Notes</h1>

<p align="center">
  <b>Free, in-depth lecture notes for Stanford CS231n: Deep Learning for Computer Vision (Spring 2025).</b><br>
  CNNs, RNNs, Transformers, object detection and segmentation — written to be read, not skimmed.
</p>

<h3 align="center">
  <a href="https://raimbekovm.github.io/cs231n-2025-notes/">Read the notes online</a> ·
  <a href="https://raimbekovm.github.io/cs231n-2025-notes/CS231n-2025-Lecture-Notes.pdf">Download the PDF</a> ·
  <a href="https://raimbekovm.github.io/cs231n-2025-notes/lectures/01-introduction.html">Start with Lecture 1</a>
</h3>

<p align="center">
  <a href="https://github.com/raimbekovm/cs231n-2025-notes/actions/workflows/publish.yml"><img src="https://github.com/raimbekovm/cs231n-2025-notes/actions/workflows/publish.yml/badge.svg" alt="Publish site"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/licence-MIT-0b5cad" alt="MIT licence"></a>
  <a href="CONTRIBUTING.md"><img src="https://img.shields.io/badge/contributions-welcome-orange" alt="Contributions welcome"></a>
</p>

---

## Lectures

| #   | Lecture                                                                                                                                | What it covers                                                                                            |
| :-- | :------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------- |
| 1   | [Introduction](https://raimbekovm.github.io/cs231n-2025-notes/lectures/01-introduction.html)                                           | Why vision is hard, Hubel and Wiesel, features and benchmarks, ImageNet and AlexNet in 2012               |
| 2   | [Image Classification with Linear Classifiers](https://raimbekovm.github.io/cs231n-2025-notes/lectures/02-image-classification.html)   | The semantic gap, k-nearest neighbours, validation, linear classifiers, softmax and multiclass SVM loss   |
| 3   | [Regularization and Optimization](https://raimbekovm.github.io/cs231n-2025-notes/lectures/03-regularization-optimization.html)         | L1 and L2 regularization, gradient descent, SGD, momentum, Adam and AdamW, learning-rate schedules        |
| 4   | [Neural Networks and Backpropagation](https://raimbekovm.github.io/cs231n-2025-notes/lectures/04-neural-networks-backpropagation.html) | What a hidden layer buys you, activation functions, the chain rule on a computational graph, Jacobians    |
| 5   | [Convolutional Networks](https://raimbekovm.github.io/cs231n-2025-notes/lectures/05-convolutional-networks.html)                       | The convolution layer, padding and stride arithmetic, receptive fields, pooling, equivariance             |
| 6   | [CNN Architectures](https://raimbekovm.github.io/cs231n-2025-notes/lectures/06-cnn-architectures.html)                                 | Batch normalization, dropout, VGG, ResNet and residual connections, initialization, transfer learning     |
| 7   | [Recurrent Neural Networks](https://raimbekovm.github.io/cs231n-2025-notes/lectures/07-recurrent-neural-networks.html)                 | Recurrence, backpropagation through time, character-level language models, vanishing gradients, LSTM      |
| 8   | [Attention and Transformers](https://raimbekovm.github.io/cs231n-2025-notes/lectures/08-attention-transformers.html)                   | Seq2seq attention, self-attention, multi-head attention, the transformer block, LLMs, Vision Transformers |
| 9   | [Object Detection and Segmentation](https://raimbekovm.github.io/cs231n-2025-notes/lectures/09-detection-segmentation.html)            | Semantic segmentation, U-Net, R-CNN to Faster R-CNN to YOLO to DETR, Mask R-CNN, saliency maps and CAM    |
| 10  | [Video Understanding](https://raimbekovm.github.io/cs231n-2025-notes/lectures/10-video-understanding.html)                             | Clip training, late and early fusion, 3D convolution, optical flow and two-stream nets, non-local blocks, I3D |

**Coming next**, in course order: large-scale distributed training,
self-supervised learning, generative models (VAEs, GANs, diffusion), 3D vision,
vision–language models, and robot learning.

Every lecture is also in the
[single-file PDF](https://raimbekovm.github.io/cs231n-2025-notes/CS231n-2025-Lecture-Notes.pdf),
and the site has full-text search, a dark theme, and numbered sections you can
link to directly.

---

## About these notes

Stanford's [CS231n](https://cs231n.stanford.edu/) is the course a lot of people
learned computer vision from, and its [official notes](https://cs231n.github.io/)
stopped being updated in **2017** — before Vision Transformers, contrastive
pretraining, latent diffusion and vision–language models.

These notes are written from the **Spring 2025 lectures**, not summarised from
the old ones. Each chapter is self-contained prose with the derivations written
out, so you do not need the video open in another tab, and every result links to
the paper it came from.

This is an **unofficial** resource, not affiliated with Stanford University. For
official course materials see [cs231n.stanford.edu](https://cs231n.stanford.edu/)
and the [2025 lecture videos](https://www.youtube.com/playlist?list=PLoROMvodv4rOmsNzYBMe0gJY2XS8AQg16).

## Found a mistake?

Corrections are genuinely welcome. Every page has an **Edit this page** link at
the bottom that opens a pull request from your browser; for anything larger, open
an [issue](https://github.com/raimbekovm/cs231n-2025-notes/issues).
[CONTRIBUTING.md](CONTRIBUTING.md) has the style guide.

## Licence

The writing and the diagrams made for this project are [MIT](LICENSE)-licensed.
Figures from research papers and lecture slides remain their authors' —
per-figure provenance is in [figures/SOURCES.md](figures/SOURCES.md), and
[NOTICE.md](NOTICE.md) has the full picture.

<details>
<summary><b>Build the site locally</b></summary>

The notes are a [Quarto](https://quarto.org/) book: one `.qmd` per lecture in
`lectures/`, figures generated by scripts in `figures/`, and CI renders the site
and the PDF on every push to `main`.

```bash
quarto preview           # live-reloading site at localhost:4321
quarto render            # full build into _site/ (site + PDF; PDF needs LaTeX)
quarto render --to html  # site only — no LaTeX needed
```

To redraw a chapter's figures:

```bash
python3 figures/09-detection-segmentation/make_diagrams.py
```

</details>
