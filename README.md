# 🎤 EpigrafIA - Detección Inteligente de Voz

<div align="center">

![EpigrafIA Logo](frontend/public/LOGOTyA_tfg.svg)

**Reconocimiento de idioma y acento usando Deep Learning, ejecutándose 100% en el navegador**

[![TensorFlow.js](https://img.shields.io/badge/TensorFlow.js-4.15.0-orange?logo=tensorflow)](https://www.tensorflow.org/js)
[![Astro](https://img.shields.io/badge/Astro-5.16.4-blueviolet?logo=astro)](https://astro.build)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind-4.1.17-38bdf8?logo=tailwindcss)](https://tailwindcss.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[Demo en Vivo](#) | [Documentación](#-características) | [Contribuir](#-contribución)

</div>

---

## 🌟 Características

### 🎯 Detección de Idioma
Identifica automáticamente entre **4 idiomas principales**:
- 🇪🇸 **Español**
- 🇬🇧 **Inglés**
- 🇫🇷 **Francés**
- 🇩🇪 **Alemán**

### 🗣️ Profiling de Acento
Reconoce **8 acentos diferentes** con alta precisión:
- 🇪🇸 Español (España) vs 🇲🇽 (México)
- 🇬🇧 Inglés (UK) vs 🇺🇸 (USA)
- 🇫🇷 Francés (Francia) vs 🇨🇦 (Quebec)
- 🇩🇪 Alemán (Alemania) vs 🇦🇹 (Austria)

### ⚡ Tecnología de Vanguardia
- ✅ **100% Client-Side** - Sin backend, sin APIs externas
- ✅ **Deep Learning en el Navegador** - TensorFlow.js para inferencia en tiempo real
- ✅ **Redes Neuronales CNN** - Arquitectura optimizada para audio
- ✅ **MFCC Features** - Análisis espectral avanzado del audio
- ✅ **Interfaz Moderna** - Diseño responsive con Tailwind CSS 4
- ✅ **Visualización en Tiempo Real** - Waveform animado del audio

---

## 🚀 Inicio Rápido

### Prerrequisitos

- **Node.js** >= 18.0.0
- **Python** >= 3.9 (solo para entrenamiento)
- **npm** o **yarn**

### Instalación Frontend

```bash
# Clonar repositorio
git clone https://github.com/adrianberlinchesayala25/EpigrafIA.git
cd EpigrafIA/frontend

# Instalar dependencias
npm install

# Ejecutar en desarrollo
npm run dev
```

Abre [http://localhost:4321](http://localhost:4321) en tu navegador 🎉

### Instalación Backend (Entrenamiento)

```bash
# Volver al root del proyecto
cd ..

# Instalar dependencias Python
pip install -r requirements.txt

# Ejecutar notebooks de entrenamiento
jupyter notebook notebooks/train_language_model.ipynb
```

---

## 📁 Estructura del Proyecto

```
EpigrafIA/
├── 📂 frontend/              # Aplicación web (Astro + Tailwind)
│   ├── src/
│   │   ├── pages/
│   │   │   └── index.astro   # Página principal
│   │   ├── components/       # Componentes reutilizables
│   │   └── utils/
│   │       ├── modelLoader.js        # Carga de modelos TF.js
│   │       └── audioProcessing.js    # Extracción de MFCC
│   ├── public/
│   │   ├── models/           # Modelos TensorFlow.js
│   │   │   ├── language/     # Modelo de idiomas
│   │   │   └── accent/       # Modelo de acentos
│   │   └── LOGOTyA_tfg.svg   # Logo animado
│   └── package.json
│
├── 📂 notebooks/             # Entrenamiento de modelos
│   ├── train_language_model.ipynb
│   └── train_accent_model.ipynb
│
├── 📂 data/                  # Datasets (no incluido en repo)
│   └── Common Voice/
│       ├── Audios Español/   (2000 audios + validated.tsv)
│       ├── Audios Ingles/    (2000 audios + validated.tsv)
│       ├── Audios Frances/   (2000 audios + validated.tsv)
│       └── Audios Aleman/    (2000 audios + validated.tsv)
│
├── requirements.txt          # Dependencias Python
└── README.md                 # Este archivo
```

---

## 🧠 Arquitectura de los Modelos

### Red Neuronal Convolucional (CNN)

Ambos modelos (idiomas y acentos) utilizan una arquitectura CNN optimizada:

```python
Input: (130, 120) 
  ↓
Conv1D (64 filters) → ReLU → BatchNorm → MaxPool → Dropout(0.3)
  ↓
Conv1D (128 filters) → ReLU → BatchNorm → MaxPool → Dropout(0.3)
  ↓
Conv1D (256 filters) → ReLU → BatchNorm → GlobalAvgPool
  ↓
Dense (128) → ReLU → Dropout(0.4)
  ↓
Dense (num_classes) → Softmax
```

**Features de entrada:**
- **40 MFCC** + **40 Delta-MFCC** + **40 Delta²-MFCC**
- Ventanas de **3 segundos** a **16kHz**
- **130 time steps** por audio

**Precisión alcanzada:**
- 🎯 **Idiomas:** ~92% accuracy
- 🗣️ **Acentos:** ~85% accuracy

---

## 🎨 Flujo de Uso

1. **Grabar Audio** 🎙️ o **Subir Archivo** 📁
2. **Visualización de Waveform** 🌊
3. **Análisis Neural** 🧠
4. **Resultados Instantáneos** con probabilidades ⚡

---

## 🛠️ Comandos Disponibles

### Frontend

| Comando | Acción |
|---------|--------|
| `npm install` | Instalar dependencias |
| `npm run dev` | Servidor de desarrollo (puerto 4321) |
| `npm run build` | Build para producción |
| `npm run preview` | Preview del build |

### Backend (Entrenamiento)

| Comando | Acción |
|---------|--------|
| `pip install -r requirements.txt` | Instalar librerías Python |
| `jupyter notebook` | Abrir notebooks de entrenamiento |
| `python -m tensorflowjs_converter ...` | Convertir modelos a TF.js |

---

## 📊 Dataset

El proyecto utiliza el dataset **Common Voice de Mozilla**, con:

- ✅ **8,000 audios** totales (2,000 por idioma)
- ✅ **Validados manualmente** (`validated.tsv`)
- ✅ **Metadatos completos** (duración, votos, etc.)
- ✅ **Multi-speaker** para generalización

### Descarga del Dataset

1. Ve a [Mozilla Common Voice](https://commonvoice.mozilla.org/datasets)
2. Descarga los idiomas: Español, Inglés, Francés, Alemán
3. Coloca los audios en `data/Common Voice/Audios {Idioma}/`

---

## 🔬 Tecnologías Utilizadas

### Frontend
- **Astro 5** - Framework web moderno
- **Tailwind CSS 4** - Estilos utility-first
- **TensorFlow.js** - Inferencia de ML en el navegador
- **Web Audio API** - Grabación y procesamiento de audio
- **Canvas API** - Visualización de waveforms

### Backend / Training
- **TensorFlow 2.15** - Entrenamiento de modelos
- **Librosa** - Procesamiento de audio
- **NumPy & Pandas** - Manipulación de datos
- **Scikit-learn** - Métricas y validación
- **Matplotlib & Seaborn** - Visualización de resultados

---

## 🤝 Contribución

¡Las contribuciones son bienvenidas! Por favor:

1. Fork el proyecto
2. Crea una rama (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add: Amazing Feature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

---

## 📄 Licencia

Este proyecto está bajo la licencia **MIT**. Ver [LICENSE](LICENSE) para más detalles.

---

## 👨‍💻 Autor

**Adrián Berlinches Ayala**

- GitHub: [@adrianberlinchesayala25](https://github.com/adrianberlinchesayala25)
- Email: [berlinchesayalaadrian@gmail.com]

---

## 🙏 Agradecimientos

- **Mozilla Common Voice** por el dataset público
- **TensorFlow.js** por hacer posible ML en el navegador
- **Astro Team** por el increíble framework
- Comunidad de **Deep Learning en Audio**

---

<div align="center">

**⭐ Si te gusta este proyecto, dale una estrella en GitHub! ⭐**

Hecho con ❤️ y 🎵 por Adrián Berlinches

</div>
