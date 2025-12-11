# 🚀 Guía de Despliegue: EpigrafIA en Hugging Face Spaces + Vercel

## Arquitectura

```
┌─────────────────┐         ┌──────────────────────┐
│   VERCEL        │         │  HUGGING FACE SPACES │
│   (Frontend)    │ ──────► │  (Backend + ML)      │
│   Astro + CSS   │   API   │  FastAPI + TensorFlow│
└─────────────────┘         └──────────────────────┘
```

## Paso 1: Crear Space en Hugging Face

1. Ve a https://huggingface.co/spaces
2. Click "Create new Space"
3. Configuración:
   - **Space name**: `epigrafia` (o el nombre que quieras)
   - **License**: MIT
   - **SDK**: Docker
   - **Hardware**: CPU basic (gratuito)

4. Clona tu nuevo Space:
   ```bash
   git clone https://huggingface.co/spaces/TU_USUARIO/epigrafia
   cd epigrafia
   ```

## Paso 2: Copiar archivos al Space

Copia estos archivos desde tu proyecto:

```bash
# Desde la carpeta backend/
cp backend/hf_app.py ./app.py
cp backend/hf_requirements.txt ./requirements.txt

# Crea el Dockerfile
cat > Dockerfile << 'EOF'
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY models/ ./models/

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
EOF

# Copia tu modelo entrenado
mkdir -p models
cp outputs/models_trained/language_model_best.keras models/
```

## Paso 3: Crear README.md para HF

```bash
cat > README.md << 'EOF'
---
title: EpigrafIA
emoji: 🎤
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# 🎤 EpigrafIA API

API de detección de idioma mediante Deep Learning.

## Endpoints

- `POST /api/predict` - Analizar audio
- `GET /api/health` - Health check
EOF
```

## Paso 4: Subir a Hugging Face

```bash
git add .
git commit -m "Initial deployment"
git push
```

El Space se construirá automáticamente. Una vez listo, tendrás una URL como:
```
https://TU_USUARIO-epigrafia.hf.space
```

## Paso 5: Actualizar Frontend para usar HF

Edita `frontend/src/config.ts` o donde configures la URL del API:

```typescript
// Producción: Hugging Face Spaces
export const API_URL = import.meta.env.PROD 
  ? 'https://TU_USUARIO-epigrafia.hf.space'
  : 'http://localhost:8000';
```

## Paso 6: Desplegar Frontend en Vercel

1. Conecta tu repo a Vercel (https://vercel.com/new)
2. Configura:
   - **Framework**: Astro
   - **Root Directory**: `frontend`
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`

3. Añade variable de entorno en Vercel Dashboard → Settings → Environment Variables:
   ```
   PUBLIC_HF_API_URL=https://TU_USUARIO-epigrafia.hf.space
   ```
   ⚠️ IMPORTANTE: El nombre debe ser exactamente `PUBLIC_HF_API_URL`

## ✅ Resultado Final

- **Frontend**: `https://tu-proyecto.vercel.app`
- **Backend API**: `https://tu-usuario-epigrafia.hf.space`

Ambos gratuitos, funcionando en PC y móvil.

## Troubleshooting

### El modelo no carga
- Verifica que `language_model_best.keras` está en `models/`
- El archivo debe estar en el repo de HF (usa Git LFS para archivos >10MB)

### CORS errors
- El backend ya tiene CORS configurado para `*`
- Si necesitas restringir, edita `app.py`

### Cold starts lentos
- HF Spaces gratuito tiene cold starts de ~30s
- El modelo TensorFlow tarda en cargar la primera vez
