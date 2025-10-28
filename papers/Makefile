# Makefile for USENIX Security 2026 Submission
# Paper: Geometric Theory of Functional-Sensitive Outlier Separability

# Main document name (without .tex extension)
MAIN = functional_outlier_separability

# LaTeX compiler
LATEX = pdflatex
BIBTEX = bibtex

# Flags for pdflatex
LATEXFLAGS = -interaction=nonstopmode -file-line-error

# Output PDF
PDF = $(MAIN).pdf

# Auxiliary files
AUX = $(MAIN).aux
BBL = $(MAIN).bbl
BLG = $(MAIN).blg
LOG = $(MAIN).log
OUT = $(MAIN).out

# Default target
all: $(PDF)

# Full build with bibliography
$(PDF): $(MAIN).tex $(MAIN).bib usenix.sty
	@echo "Building $(PDF)..."
	$(LATEX) $(LATEXFLAGS) $(MAIN)
	$(BIBTEX) $(MAIN)
	$(LATEX) $(LATEXFLAGS) $(MAIN)
	$(LATEX) $(LATEXFLAGS) $(MAIN)
	@echo "Build complete: $(PDF)"

# Quick build (no bibliography update)
quick: $(MAIN).tex
	@echo "Quick build (no bibliography)..."
	$(LATEX) $(LATEXFLAGS) $(MAIN)
	@echo "Quick build complete: $(PDF)"

# Clean auxiliary files
clean:
	@echo "Cleaning auxiliary files..."
	rm -f $(AUX) $(BBL) $(BLG) $(LOG) $(OUT)
	rm -f $(MAIN).toc $(MAIN).lof $(MAIN).lot
	rm -f $(MAIN).synctex.gz
	rm -f *.aux *.log *.out *.bbl *.blg
	@echo "Clean complete."

# Clean everything including PDF
distclean: clean
	@echo "Removing PDF..."
	rm -f $(PDF)
	@echo "Distclean complete."

# View PDF (platform-dependent)
view: $(PDF)
ifeq ($(OS),Windows_NT)
	start $(PDF)
else ifeq ($(shell uname -s),Darwin)
	open $(PDF)
else
	xdg-open $(PDF) || evince $(PDF) || okular $(PDF)
endif

# Count words (approximate, excluding LaTeX commands)
wordcount: $(MAIN).tex
	@echo "Approximate word count:"
	@detex $(MAIN).tex | wc -w

# Check for common LaTeX errors
check: $(MAIN).tex
	@echo "Checking for common issues..."
	@grep -n "\\citep\|\\citet" $(MAIN).tex && echo "Warning: Use \\cite instead of natbib commands" || true
	@grep -n "\\usepackage.*times" $(MAIN).tex && echo "Warning: Do not manually load times package (USENIX style handles fonts)" || true
	@grep -n "\\vspace.*-" $(MAIN).tex && echo "Warning: Negative vspace detected (violates USENIX formatting)" || true
	@grep -n "TODO\|FIXME\|XXX" $(MAIN).tex && echo "Warning: TODO/FIXME markers found" || true
	@echo "Check complete."

# Spell check (requires aspell)
spell: $(MAIN).tex
	@echo "Running spell check..."
	@aspell --lang=en --mode=tex check $(MAIN).tex

# Help
help:
	@echo "Available targets:"
	@echo "  all        - Build PDF with bibliography (default)"
	@echo "  quick      - Quick build without bibliography update"
	@echo "  clean      - Remove auxiliary files"
	@echo "  distclean  - Remove all generated files including PDF"
	@echo "  view       - Open PDF in default viewer"
	@echo "  wordcount  - Approximate word count"
	@echo "  check      - Check for common formatting issues"
	@echo "  spell      - Run spell checker (requires aspell)"
	@echo "  help       - Show this help message"

.PHONY: all quick clean distclean view wordcount check spell help

