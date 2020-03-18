mkdir -p tmp/
latexmk -auxdir=tmp -outdir=tmp -xelatex -pdf mdarrin-bissect-graph.tex
cp tmp/mdarrin-bissect-graph.pdf mdarrin-bissect-graph.pdf
rm -r tmp/*
