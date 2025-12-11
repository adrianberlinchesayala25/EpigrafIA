# 🚀 Guía de Despliegue en Vercel - EpigrafIA

## 📋 Requisitos Previos

- Cuenta en [Vercel](https://vercel.com) (gratuita)
- Repositorio en GitHub/GitLab/Bitbucket
- Node.js 18+ instalado localmente

## ⚡ Despliegue Rápido

### 1️⃣ Preparar el Repositorio

```bash
# Asegúrate de tener todos los archivos necesarios
git add .
git commit -m "Add Vercel deployment configuration"
git push origin main
```

### 2️⃣ Conectar con Vercel

1. Ve a [vercel.com/new](https://vercel.com/new)
2. Conecta tu repositorio de GitHub
3. Selecciona el repositorio **EpigrafIA**
4. Vercel detectará automáticamente la configuración

### 3️⃣ Configuración del Proyecto

Vercel debería detectar automáticamente estos valores (si no, configúralos manualmente):

| Campo | Valor |
|-------|-------|
| **Framework Preset** | Astro |
| **Root Directory** | `.` (raíz) |
| **Build Command** | `cd frontend && npm run build` |
| **Output Directory** | `frontend/dist` |
| **Install Command** | `cd frontend && npm install` |

### 4️⃣ Variables de Entorno (Opcional)

Si necesitas configurar variables:

```
# En el dashboard de Vercel → Settings → Environment Variables
PYTHON_VERSION=3.11
```

### 5️⃣ Desplegar

Haz clic en **Deploy** y espera a que termine el proceso.

---

## 📁 Estructura de Archivos Creados

```
EpigrafIA/
├── vercel.json           # Configuración principal de Vercel
├── .vercelignore         # Archivos a ignorar en el despliegue
├── api/                  # Serverless Functions (Python)
│   ├── __init__.py
│   ├── requirements.txt  # Dependencias Python
│   ├── predict.py        # Endpoint /api/predict
│   ├── analyze.py        # Endpoint /api/analyze
│   └── health.py         # Endpoint /api/health
└── frontend/             # Frontend Astro
    ├── astro.config.mjs  # Configuración con adaptador Vercel
    └── package.json      # Con @astrojs/vercel
```

---

## 🔗 Endpoints Disponibles

Una vez desplegado, tendrás estos endpoints:

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/` | GET | Frontend (interfaz web) |
| `/api/health` | GET | Estado del servidor |
| `/api/predict` | POST | Predicción de idioma (audio) |
| `/api/analyze` | POST | Alias de predict |

### Ejemplo de uso:

```javascript
// Frontend - llamada al API
const response = await fetch('/api/predict', {
  method: 'POST',
  body: formData  // FormData con archivo de audio
});
const result = await response.json();
```

---

## ⚠️ Limitaciones del Plan Gratuito

### Vercel Hobby (Gratis):
- ✅ 100GB de ancho de banda/mes
- ✅ Dominios personalizados ilimitados
- ✅ SSL automático
- ⚠️ Serverless Functions: máx 10s de ejecución
- ⚠️ 1024MB de memoria por función
- ⚠️ 250MB máximo de tamaño de función

### Sobre TensorFlow:
> **Nota Importante**: TensorFlow completo (~500MB) es muy grande para Vercel serverless gratuito.

**Alternativas**:
1. **Usar TensorFlow.js en el cliente** (recomendado para este proyecto)
2. Usar `tflite-runtime` en lugar de TensorFlow completo
3. Desplegar el modelo en un servicio externo (Hugging Face, AWS Lambda)

---

## 🛠️ Desarrollo Local

Para probar antes de desplegar:

```bash
# Terminal 1 - Frontend
cd frontend
npm install
npm run dev

# Terminal 2 - Backend (para desarrollo)
cd backend
pip install -r ../requirements.txt
python main.py
```

---

## 🐛 Solución de Problemas

### Error: "Function too large"
- Reduce las dependencias en `api/requirements.txt`
- Considera usar TensorFlow.js en el frontend

### Error: "Build failed"
```bash
# Verifica que el build funciona localmente
cd frontend
npm install
npm run build
```

### Error: "Python runtime not found"
- Asegúrate de que `vercel.json` tiene `"runtime": "python3.11"`

### CORS errors
- Los headers CORS ya están configurados en `vercel.json`
- Verifica que usas rutas relativas (`/api/predict`) no absolutas

---

## 📱 Verificar Despliegue

Una vez desplegado:

1. Abre `https://tu-proyecto.vercel.app`
2. Verifica que la interfaz carga correctamente
3. Prueba `/api/health` en el navegador
4. Prueba la funcionalidad de grabación de audio

---

## 🔄 Actualizaciones

Cada `git push` a la rama principal desplegará automáticamente una nueva versión.

```bash
git add .
git commit -m "Update feature X"
git push origin main
# Vercel despliega automáticamente ✨
```

---

## 📞 Soporte

- [Documentación de Vercel](https://vercel.com/docs)
- [Guía de Python en Vercel](https://vercel.com/docs/functions/runtimes/python)
- [Astro + Vercel](https://docs.astro.build/en/guides/deploy/vercel/)
