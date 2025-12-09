# Script de Instalacion Manual para EpigrafIA
# Usar: .\install.ps1

Write-Host "==================================" -ForegroundColor Cyan
Write-Host "Instalando EpigrafIA Dependencias" -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Cyan

# 1. Actualizar pip
Write-Host "`n[1/5] Actualizando pip..." -ForegroundColor Yellow
py -3.12 -m pip install --upgrade pip

# 2. Eliminar TensorFlow incompleto si existe
Write-Host "`n[2/5] Limpiando instalacion anterior..." -ForegroundColor Yellow
py -3.12 -m pip uninstall -y tensorflow tensorflowjs tf-keras

# 3. Instalar TensorFlow completo con todas sus dependencias
Write-Host "`n[3/5] Instalando TensorFlow 2.20.0..." -ForegroundColor Yellow
py -3.12 -m pip install tensorflow==2.20.0

# 4. Instalar TensorFlowJS (sin tensorflow-decision-forests)
Write-Host "`n[4/5] Instalando TensorFlowJS..." -ForegroundColor Yellow
py -3.12 -m pip install --no-deps tensorflowjs
py -3.12 -m pip install tensorflow-hub six

# 5. Instalar librerias de audio y ciencia de datos
Write-Host "`n[5/5] Instalando librerias de audio, ML y visualizacion..." -ForegroundColor Yellow
py -3.12 -m pip install librosa soundfile audioread
py -3.12 -m pip install numpy pandas scikit-learn
py -3.12 -m pip install matplotlib seaborn
py -3.12 -m pip install tqdm joblib pyyaml

Write-Host "`n==================================" -ForegroundColor Green
Write-Host " Instalacion completada!" -ForegroundColor Green
Write-Host "==================================" -ForegroundColor Green

Write-Host "`nVerificando TensorFlow..." -ForegroundColor Cyan
py -3.12 -c "import tensorflow as tf; print('TensorFlow', tf.__version__, 'OK')"

Write-Host "`nVerificando Librosa..." -ForegroundColor Cyan
py -3.12 -c "import librosa; print('Librosa', librosa.__version__, 'OK')"

Write-Host "`nPara usar Jupyter Notebooks, instalalo con:" -ForegroundColor Magenta
Write-Host "  py -3.12 -m pip install notebook" -ForegroundColor White

Write-Host "`nAhora puedes entrenar modelos o ejecutar el frontend!" -ForegroundColor Green
