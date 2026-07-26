# Contributing

Thank you for your interest in improving these lecture notes! This document explains how to contribute effectively.

## Table of Contents

- [Getting Started](#getting-started)
- [Types of Contributions](#types-of-contributions)
- [Reporting Issues](#reporting-issues)
- [Submitting Changes](#submitting-changes)
- [Style Guide](#style-guide)
- [Project Structure](#project-structure)
- [Review Process](#review-process)
- [Code of Conduct](#code-of-conduct)

## Getting Started

1. **Fork** the repository
2. **Clone** your fork locally
3. Set up the **upstream** remote:
   ```bash
   git remote add upstream https://github.com/raimbekovm/cs231n-2025-notes.git
   ```
4. Keep your fork in sync:
   ```bash
   git fetch upstream
   git merge upstream/main
   ```

## Types of Contributions

We welcome the following contributions:

| Type          | Description                           | Label         |
| ------------- | ------------------------------------- | ------------- |
| Typo          | Spelling or grammar error             | `typo`        |
| Bug fix       | Formatting issues, broken links       | `bug`         |
| Correction    | Academic inaccuracies, wrong formulas | `correction`  |
| Clarification | Improve unclear explanations          | `enhancement` |
| Suggestion    | Ideas for improvement                 | `suggestion`  |

### Available Labels

**Issue type:**

- `typo` — Spelling or grammar error
- `bug` — Something isn't working (formatting, links)
- `correction` — Academic inaccuracies, wrong formulas
- `enhancement` — Improve clarity or add content
- `suggestion` — Ideas for improvement
- `question` — Questions about the content

**Lecture-specific:**

- `lecture-1` through `lecture-N` — Tag by lecture number

**Status:**

- `good first issue` — Good for newcomers
- `help wanted` — Extra attention needed
- `duplicate` — Already exists
- `wontfix` — Won't be addressed

## Reporting Issues

### Before Opening an Issue

- Search existing issues to avoid duplicates
- Check if the issue exists in the latest version

### Issue Template

When reporting an error, please include:

```markdown
**Lecture**: [e.g., Lecture 1, Part 1]
**Section**: [e.g., Section 3.2 - Backpropagation]
**Page/Line**: [if applicable]

**Current content**:
[Quote the problematic text]

**Problem**:
[Describe what's wrong]

**Suggested fix** (optional):
[Your proposed correction]

**Source** (for academic corrections):
[Link to paper/textbook supporting the correction]
```

### Issue Types

**Typo/Grammar**

> "In section 2.1, 'recieve' should be 'receive'"

**Academic Error**

> "The gradient formula in equation (3) is missing a negative sign. See [Goodfellow et al., Deep Learning, p. 205]"

**Unclear Explanation**

> "The explanation of receptive fields in section 4 assumes prior knowledge of convolutions, but convolutions aren't introduced until section 5"

**Broken Link**

> "The link to the AlexNet paper in section 7 returns 404"

## Submitting Changes

### Branch Naming

Use descriptive branch names:

```
fix/lecture1-typo-backprop
fix/lecture3-gradient-formula
docs/improve-cnn-explanation
```

### Workflow

1. Create a branch from `main`:

   ```bash
   git checkout -b fix/description
   ```

2. Make your changes

3. Test your changes with `quarto preview` and check that:
   - the page renders without warnings in the terminal
   - equations, figures, and cross-references all resolve
   - any links you added actually work

4. Commit with a clear message:

   ```bash
   git commit -m "Fix typo in lecture 1 backprop section"
   ```

5. Push to your fork:

   ```bash
   git push origin fix/description
   ```

6. Open a Pull Request against `main`

### Pull Request Guidelines

- **Title**: Brief description of the change
- **Description**: Explain what and why (not how)
- **Reference**: Link to related issue if applicable
- **Scope**: Keep changes focused — one fix per PR when possible

#### Good PR Example

```markdown
**Title**: Fix gradient descent formula in Lecture 3

**Description**:
The learning rate was on the wrong side of the equation.

Fixes #12

**Changes**:

- Corrected equation (7) in section 3.2
```

## Style Guide

Notes are written in [Quarto](https://quarto.org/) markdown (`.qmd`). One file
per lecture, in `lectures/`.

### Document structure

Each file opens with YAML front matter, then uses markdown headings. Section
numbering is automatic — don't number headings by hand.

```markdown
---
title: "Image Classification"
subtitle: "Lecture 2"
description: "One sentence, used for search results and link previews."
lecturer: "Justin Johnson"
---

## Section title

### Subsection title
```

### Text formatting

| Element              | Markup                | Renders as          |
| -------------------- | --------------------- | ------------------- |
| Key term (first use) | `**backpropagation**` | **backpropagation** |
| Emphasis             | `*critical*`          | _critical_          |
| Code and variables   | `` `learning_rate` `` | `learning_rate`     |

### Mathematics

Inline math uses single dollars, display math double:

```markdown
The loss function $L(\theta)$ measures error.

$$
\nabla_\theta L = \frac{1}{n} \sum_{i=1}^{n} \nabla_\theta \ell(f(x_i; \theta), y_i)
$$ {#eq-gradient}
$$
```

Reference it as `@eq-gradient`. Quarto inserts the word "Equation" itself, so
write `see @eq-gradient`, not `see Equation @eq-gradient`.

### Figures

```markdown
![Description of the figure. Source: Krizhevsky et al., NeurIPS 2012.](../figures/02-classification/example.png){#fig-example}
```

Reference as `@fig-example`. Identifiers must start with `fig-` for figures,
`eq-` for equations, `tbl-` for tables, and `sec-` for sections — Quarto uses
the prefix to decide how to number and label the cross-reference.

Figures live in `figures/<lecture-slug>/`, named in lowercase with underscores.
Always credit the original paper in the caption, not the lecture slide it was
screenshotted from.

### Callout boxes

Three kinds carry the pedagogical structure. Every major section ends with Key
Takeaways.

```markdown
::: {.callout-tip title="Key Takeaways"}
The main points of the section.
:::

::: {.callout-note title="Deep Dive: Why This Works"}
Material that goes beyond what the lecture covered.
:::

::: {.callout-warning title="Note"}
An important caveat or common misconception.
:::
```

### Links and footnotes

Link to the original paper whenever research is mentioned:

```markdown
[AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks) achieved...
```

Footnotes hold supplementary context:

```markdown
This phenomenon[^hw] suggests...

[^hw]: First observed by Hubel & Wiesel in 1959.
```

Use a descriptive label rather than a number, so footnotes stay stable when you
insert one in the middle.

### Tables

Use pipe tables — they are the only kind that survives hand-editing:

```markdown
| Task           | Output           |
| :------------- | :--------------- |
| Classification | one label        |
| Detection      | boxes and labels |
```

### Commit Messages

Format: `<type>: <description>`

Types:

- `fix`: Bug fixes, typos, corrections
- `docs`: Documentation improvements
- `style`: Formatting changes (no content change)

Examples:

```
fix: correct ReLU derivative formula in lecture 4
fix: typo in lecture 1 section 2.3
docs: clarify batch normalization explanation
style: fix equation alignment in lecture 5
```

Keep messages:

- Under 72 characters
- In imperative mood ("fix" not "fixed")
- Lowercase (except proper nouns)

## Project Structure

```
cs231n-2025-notes/
├── index.qmd                 # site landing page
├── lectures/                 # one .qmd per lecture — the source of truth
│   ├── 01-history.qmd
│   └── 01b-course-overview.qmd
├── figures/                  # images, one directory per lecture
│   └── 01-history/
├── _quarto.yml               # site and PDF configuration
├── theme.scss                # light theme
├── theme-dark.scss           # dark theme
├── notebooks/                # Jupyter notebooks (future)
├── CONTRIBUTING.md
├── LICENSE
└── README.md
```

`_site/` holds the rendered output. It is generated, gitignored, and rebuilt by
CI on every push — never edit or commit it.

### File Naming

- Lectures: `NN-topic-slug.qmd`, matching the order they appear in `_quarto.yml`
- Figures: `figures/<lecture-slug>/descriptive_name.png` (lowercase, underscores)

## Review Process

1. **Automated checks**: the PR must render cleanly with `quarto render`
2. **Maintainer review**: Changes reviewed for accuracy and style
3. **Feedback**: You may be asked to make adjustments
4. **Merge**: Once approved, changes are merged to `main`

Typical review time: 1-7 days depending on complexity.

## Code of Conduct

### Our Standards

- Be respectful and constructive in all interactions
- Focus on the content, not the contributor
- Assume good intentions
- Accept feedback gracefully
- Help others learn

### Unacceptable Behavior

- Personal attacks or insults
- Dismissive or condescending responses
- Spam or off-topic content

### Enforcement

Violations may result in:

1. Warning
2. Temporary ban
3. Permanent ban

## Questions?

- **General questions**: Open an issue with the `question` label
- **Discussions**: Use GitHub Discussions (if enabled)

Thank you for contributing!
