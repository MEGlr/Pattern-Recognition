#/bin/bash

# patrec3_ΑριθμόςΜητρώου1ουΣυνεργάτη_ΑριθμόςΜητρώου2ουΣυνεργάτη.zip

[ -e "patrec3_el18176_el18052.zip" ] && rm patrec3_el18176_el18052.zip

zip patrec3_el18176_el18052.zip report/report.pdf plots/README.md input/README.md README.md prelab_part1.py main_lab.py ./help_code/*.py ./help_code/README.md

