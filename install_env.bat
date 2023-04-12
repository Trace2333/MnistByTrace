@echo off
echo Automatically install required python environments...

conda create -n mnist python==3.7.0

pip install -r requirements.txt

echo Install complete!!!