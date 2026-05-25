.PHONY: all clean

CHAPTERS = ch01-residual-stream.tex ch02-attention-and-gate.tex \
           ch03-ffn-or-gate.tex ch04-layers-rounds-of-bp.tex \
           ch05-scaling-numbers.tex ch06-residual-stream-vs-propositions.tex \
           ch07-evidence-from-interp.tex ch08-three-softmaxes.tex \
           ch09-grounded-vs-ungrounded.tex ch10-log-odds-algebra.tex \
           ch11-how-to-read-weights.tex ch12-or-nodes-vs-dimensions.tex \
           ch13-related-work.tex

all: index.pdf

index.pdf: index.tex refs.bib $(CHAPTERS)
	pdflatex -interaction=nonstopmode index.tex
	bibtex index
	pdflatex -interaction=nonstopmode index.tex
	pdflatex -interaction=nonstopmode index.tex

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.pdf