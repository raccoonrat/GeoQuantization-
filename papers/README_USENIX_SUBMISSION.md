# USENIX Security 2026 Submission

## Geometric Theory of Functional-Sensitive Outlier Separability and Its Application in Privacy-Aware Model Quantization

This directory contains the LaTeX source for our USENIX Security 2026 submission.

---

## Files

### Main Paper
- `functional_outlier_separability.tex` - Main paper source (LaTeX)
- `functional_outlier_separability.bib` - Bibliography file (BibTeX)
- `usenix.sty` - USENIX conference style file

### Figures (to be added)
- `figs/` - Directory for all figures
  - `framework.png` - System architecture diagram
  - `tsne_visualization.pdf` - t-SNE visualization of parameter separability
  - `quantization_error_dist.pdf` - Quantization error distribution
  - `privacy_accuracy_pareto.pdf` - Privacy-accuracy trade-off curve
  - `roc_curves.pdf` - ROC curves for membership inference attacks

### Original Materials
- `[1028]量化离群点可分离性理论-修改稿-1.md` - Original Chinese draft
- `Recommendations for USENIXSecurity Submission.md` - Submission guidelines

---

## Compilation Instructions

### Prerequisites
- LaTeX distribution (TeX Live 2020 or later recommended)
- BibTeX for bibliography processing
- Required LaTeX packages (all standard in modern distributions):
  - `amsmath`, `amssymb`, `amsthm`
  - `algorithm`, `algorithmic`
  - `graphicx`, `booktabs`, `multirow`

### Build Commands

#### Option 1: Using pdflatex (recommended)
```bash
cd papers
pdflatex functional_outlier_separability.tex
bibtex functional_outlier_separability
pdflatex functional_outlier_separability.tex
pdflatex functional_outlier_separability.tex
```

#### Option 2: Using latexmk (automated)
```bash
cd papers
latexmk -pdf functional_outlier_separability.tex
```

#### Option 3: Using make (if Makefile provided)
```bash
cd papers
make
```

The final PDF will be: `functional_outlier_separability.pdf`

---

## Paper Structure

### Main Sections (≤13 pages)
1. **Abstract** (~150 words)
2. **Introduction** (with Threat Model)
3. **Related Work** (Quantization, Privacy, Geometric Methods)
4. **Geometric Foundations** (Riemannian Manifolds, Hessian, Lattice)
5. **Functional Outlier Separability Theory** (Core Contributions)
6. **Methodology** (HOP and DQC Algorithms)
7. **Evaluation** (Experiments, Visualizations, Ablations)
8. **Discussion** (Connections, Limitations, Future Work)
9. **Conclusion**

### Mandatory Appendices (1 page each)
- **Ethical Considerations** - Required by USENIX CFP
- **Open Science** - Artifact availability statement

### References
- Comprehensive bibliography with 40+ citations
- Covers quantization, privacy, lattice theory, geometric deep learning

---

## Key Features Aligned with USENIX Requirements

### ✅ Reproducibility
- Detailed experimental setup in Section 5.1
- Hardware specifications and hyperparameters documented
- Open Science appendix lists all code and data artifacts
- Random seeds and configuration files to be released

### ✅ Visualization
- **Figure 1** (described): t-SNE visualization of parameter geometric separability
- **Figure 2** (described): Quantization error distributions (histograms)
- **Figure 3** (described): Privacy-accuracy Pareto frontier
- **Figure 4** (described): ROC curves for membership inference attacks (linear and log scale)
- **Tables**: Comparison of methods, computational efficiency, ablation studies

### ✅ Threat Model
- Explicit adversary capabilities (white-box, query access)
- Clear privacy goals (membership inference)
- Formal DP guarantees with ($\epsilon, \delta$) parameters

### ✅ Related Work
- Connections to lattice theory and CVP problem
- Discussion of privacy-preserving quantization literature
- Interpretability methods context

### ✅ Ethical Considerations
- One-page appendix addressing potential harms
- Discussion of fairness, dual-use, and environmental impact

### ✅ Open Science
- GitHub repository (to be released)
- Public datasets (GLUE, CIFAR-10, WikiText-2)
- Model checkpoints on HuggingFace
- MIA evaluation framework

---

## Submission Checklist

Before submitting to USENIX Security 2026:

- [ ] Compile successfully with no errors
- [ ] Total length: ≤13 pages (main text) + 2 pages (mandatory appendices) + references
- [ ] All figures are legible in grayscale
- [ ] Fonts embedded in PDF
- [ ] Anonymized (no author names, acknowledgments removed)
- [ ] References formatted in USENIX style
- [ ] All citations have proper URLs where applicable
- [ ] Algorithms numbered and referenced
- [ ] Equations numbered only if referenced
- [ ] Supplementary materials prepared (code repository, datasets)
- [ ] Ethical considerations addressed
- [ ] Open science artifacts listed

---

## Correspondence with Submission Guidelines

This paper follows the recommendations in `Recommendations for USENIXSecurity Submission.md`:

1. **Enhanced Reproducibility** ✓
   - Detailed dataset descriptions
   - Code repository with configuration
   - Hardware/software environment documented

2. **Quantization-Error Visualization** ✓
   - Histograms of weight/activation differences
   - Scatter plots for outliers vs. inliers

3. **HOP Interpretability Charts** ✓
   - Heatmaps showing sensitivity scores
   - Geometric projection error distributions

4. **Privacy-Accuracy Trade-off Plots** ✓
   - Bit-width vs. accuracy and privacy metrics
   - Pareto frontier comparisons

5. **Membership-Inference ROC Curves** ✓
   - Linear and log-scaled versions
   - Comparison across methods

6. **Statistical Error Bars** ✓
   - Mean ± std over 3 runs
   - Confidence intervals on key metrics

7. **Threat Model** ✓
   - Explicit adversary capabilities
   - Privacy game formalization

8. **Lattice Theory Connections** ✓
   - CVP formulation
   - Babai algorithm equivalence

9. **Professional Language** ✓
   - Formal academic tone
   - Consistent terminology
   - USENIX style formatting

---

## Contact

For questions about this submission, please contact: [anonymized for review]

Repository: [anonymized for review, will be at: github.com/...]

---

## License

The LaTeX source and supplementary materials will be released under:
- **Code**: MIT License
- **Documentation**: CC-BY 4.0

This is an academic submission to USENIX Security 2026. All rights reserved until publication.

