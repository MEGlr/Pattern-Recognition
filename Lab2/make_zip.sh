#/bin/bash

# patrec2_ΑριθμόςΜητρώου1ουΣυνεργάτη_ΑριθμόςΜητρώου2ουΣυνεργάτη.zip

[ -e "patrec2_el18176_el18052.zip" ] && rm patrec2_el18176_el18052.zip

zip patrec2_el18176_el18052.zip report/report.pdf checkpoints/.gitkeep dataset/.gitkeep README.md *.py ./helper_scripts/*.py


