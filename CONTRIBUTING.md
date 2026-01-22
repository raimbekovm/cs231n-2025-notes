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

| Type | Description | Label |
|------|-------------|-------|
| Typo | Spelling or grammar error | `typo` |
| Bug fix | Formatting issues, broken links | `bug` |
| Correction | Academic inaccuracies, wrong formulas | `correction` |
| Clarification | Improve unclear explanations | `enhancement` |
| Suggestion | Ideas for improvement | `suggestion` |

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

3. Test your changes:
   - For LaTeX: compile the PDF and verify formatting
   - Check that links work
   - Verify equations render correctly

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

### LaTeX Conventions

#### Document Structure

```latex
\section{Section Title}
\subsection{Subsection Title}
\subsubsection{Subsubsection Title}  % use sparingly
```

#### Text Formatting

| Element | Markup | Example |
|---------|--------|---------|
| Key terms (first use) | `\textbf{}` | \textbf{backpropagation} |
| Emphasis | `\emph{}` | \emph{critical} |
| Code/variables | `\texttt{}` | \texttt{learning\_rate} |
| Quotations | `\enquote{}` | \enquote{quote} |

#### Mathematics

Inline math for simple expressions:
```latex
The loss function $L(\theta)$ measures error.
```

Display math for important equations:
```latex
\begin{equation}
    \nabla_\theta L = \frac{1}{n} \sum_{i=1}^{n} \nabla_\theta \ell(f(x_i; \theta), y_i)
    \label{eq:gradient}
\end{equation}
```

Reference equations as `Equation~\ref{eq:gradient}` (with non-breaking space).

#### Figures

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.8\textwidth]{figures/example.png}
    \caption{Description of the figure. Source: \cite{author2024}.}
    \label{fig:example}
\end{figure}
```

Reference as `Figure~\ref{fig:example}`.

#### Citations and Links

Link to original papers when mentioning research:
```latex
\href{https://papers.nips.cc/paper/...}{AlexNet} achieved...
```

Use footnotes for supplementary information:
```latex
This phenomenon\footnote{First observed by Hubel \& Wiesel in 1959.} suggests...
```

#### Custom Environments

The project uses custom `tcolorbox` environments:

```latex
\begin{keytakeaways}
    Main points to remember...
\end{keytakeaways}

\begin{deepdive}
    Advanced material for interested readers...
\end{deepdive}

\begin{notebox}
    Important notes or warnings...
\end{notebox}
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
├── notes/                    # LaTeX source and PDFs
│   ├── lecture_01_part1.pdf
│   ├── lecture_01_introduction.tex
│   └── figures/              # Images for notes
├── notebooks/                # Jupyter notebooks (future)
├── CONTRIBUTING.md
├── LICENSE
└── README.md
```

### File Naming

- Notes: `lecture_XX_partY.pdf` or `lecture_XX_topic.tex`
- Figures: `descriptive_name.png` (lowercase, underscores)

## Review Process

1. **Automated checks**: PR must not break LaTeX compilation
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
