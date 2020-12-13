cd /Users/maksim/dev_projects/pytorch_abstraction_comparison/tex
source venv/bin/activate
#pdflatex -shell-escape poster; bibtex poster; pdflatex -shell-escape poster; pdflatex -shell-escape poster
rm *.aux *.bbl *.out *.out
pdflatex -shell-escape main; bibtex main; pdflatex -shell-escape main; pdflatex -shell-escape main
docker run -ti --rm -v $(pwd):/pdf pdf2htmlex/pdf2htmlex:0.18.8.alpha-master-20200623-Ubuntu-eoan-x86_64 --fit-width 1024 /pdf/main.pdf /pdf/main.html
