.PHONY: default

default: report.pdf
	printf '\a'


report.pdf: report.md 
	pandoc report.md -s --pdf-engine=xelatex --highlight-style tango -o report.pdf

