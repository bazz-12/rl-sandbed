@echo off

call conda activate sandbed
cd /d "C:\Users\robin\projects\sandbed"

echo ===== Starting training =====
python go2_train.py -e go2-p16
if errorlevel 1 goto error

echo.
echo [1/7] Training P16 with height command...
python go2_train.py -e go2-p16h --height-cmd
if errorlevel 1 goto error

echo.
echo [2/7] Training P16 with LSTM...
python go2_train.py -e go2-p16r --rnn lstm
if errorlevel 1 goto error

echo.
echo [3/7] Training P16 with GRU...
python go2_train.py -e go2-p16g --rnn gru
if errorlevel 1 goto error

echo.
echo [4/7] Training P16 with height command and LSTM...
python go2_train.py -e go2-p16hr --height-cmd --rnn lstm
if errorlevel 1 goto error

echo.
echo [5/7] Training P16 with LSTM hidden size 128...
python go2_train.py -e go2-p16r128 --rnn lstm --rnn-hidden 128
if errorlevel 1 goto error

echo.
echo [6/7] Training P16 with 96 steps per environment...
python go2_train.py -e go2-p16r96 --rnn lstm --steps-per-env 96
if errorlevel 1 goto error

echo.
echo [7/7] Training P16 height + LSTM for 1500 iterations...
python go2_train.py -e go2-p16hr15 --height-cmd --rnn lstm --max-iterations 1500
if errorlevel 1 goto error

echo.
echo ==========================================
echo All seven training experiments completed!
echo ==========================================
pause
exit /b 0

:error
echo.
echo ==========================================
echo ERROR: A training command failed.
echo Remaining experiments were not started.
echo ==========================================
pause
exit /b 1