all:
	@(pdflatex bare_conf >> /dev/null)
	@(bibtex bare_conf >> /dev/null)
	@(pdflatex bare_conf >> /dev/null)
	@(pdflatex bare_conf >>/dev/null)

clean:
	rm -rf *.aux
	rm -rf *.bbl
	rm -rf *.blg
	rm -rf *.log
	rm -rf *.out
