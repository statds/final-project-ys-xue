source := proposal_yx.tex
out := $(patsubst %.tex,%.pdf,$(source))
bib := $(patsubst %.tex,%.bib,$(source))
rev := b09de09
latest := $(shell git log --format="%h" | head -n 1)


.PHONY: all
all: $(out)

%.pdf: %.tex $(bib)
	@latexmk $? -pdf

## compare modification with the latest version checked into git
## for comparison of specific commits:
##	latexdiff-vc --git -r old_githash -r new_githash --pdf source.tex
#Add a comment to this line
diff:
	latexdiff-vc --git --pdf --force $(source) -r $(rev)
#Add a comment to this line

log:
	@git log --pretty=format:"%h by %an at %ar: %s" $(source) | head -n 15

.PNONY: clean
clean:
	rm -rf *~ .*~ .\#* .Rhistory *.aux *.bbl *.blg *.out *.log *.toc\
		*.fff *.fdb_latexmk *.synctex.gz *.fls *.ttt *diff* *oldtmp*
	rm proposal_yx.pdf
