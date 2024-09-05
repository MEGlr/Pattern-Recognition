.PHONY: default

default: report.pdf
	printf '\a'


report.pdf: report.md 
	pandoc report.md -s --filter pandoc-include-code --pdf-engine=xelatex --highlight-style tango -o report.pdf

