#!/bin/bash

python3 -c "print('Restore.sh has been executed.')"

python src/preprocessing/noise2asterisk.py
python src/preprocessing/asterisk2GTandLnoise_Tnoise.py
python src/preprocessing/asterisk2recover_Tnoise.py
python src/preprocessing/asterisk2recover_Tnoise_only.py