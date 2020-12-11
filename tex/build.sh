cd /Users/maksim/dev_projects/pytorch_abstraction_comparison/tex
source venv/bin/activate
#pdflatex -shell-escape poster; bibtex poster; pdflatex -shell-escape poster; pdflatex -shell-escape poster
rm *.aux *.bbl *.out *.out
pdflatex -shell-escape main; bibtex main; pdflatex -shell-escape main; pdflatex -shell-escape main
