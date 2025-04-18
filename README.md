# 🎓 ChatBotPDFs - Asistente Académico Inteligente

[![GitHub Stars](https://img.shields.io/github/stars/SebastianVega4/ChatBotPDFs?style=for-the-badge)](https://github.com/SebastianVega4/ChatBotPDFs/stargazers)  
[![Last Commit](https://img.shields.io/github/last-commit/SebastianVega4/ChatBotPDFs?style=for-the-badge)](https://github.com/SebastianVega4/ChatBotPDFs/commits/main)  
[![Languages](https://img.shields.io/github/languages/top/SebastianVega4/ChatBotPDFs?style=for-the-badge)]  
[![License](https://img.shields.io/badge/License-GPL%203.0-blue?style=for-the-badge)](https://github.com/SebastianVega4/ChatBotPDFs/blob/main/LICENSE)

**ChatBotPDFs** es una plataforma web que, sin pretender reemplazar al bibliotecario, facilita la búsqueda de contenidos académicos mediante consulta directa sobre documentos en formato PDF, Word o texto plano. Mediante técnicas de procesamiento de lenguaje natural y aprendizaje automático, este asistente inteligente extrae, indexa y responde con precisión (y un toque de humor cuando la consulta lo pide) a las dudas de los usuarios.

---

## ✨ Características Principales

En lugar de pasar horas hojeando manualmente decenas de páginas, **ChatBotPDFs** ofrece:

- **Carga rápida** de documentos: arrastre y suelte archivos en la interfaz web; no se requiere magia, solo un navegador moderno.  
- **Normalización y fragmentación**: el contenido se limpia de ruido (palabras vacías, signos de puntuación innecesarios) y se divide en fragmentos manejables para optimizar la búsqueda.  
- **Embeddings semánticos**: emplea modelos de Sentence Transformers para representar cada fragmento en un espacio vectorial donde la similitud importa más que las coincidencias literales.  
- **Mecanismo de respuesta**: combina los fragmentos más relevantes mediante similitud de coseno y genera una respuesta coherente con un modelo BART afinado, incluyendo el nivel de confianza y las fuentes consultadas.

---

## 🏗️ Arquitectura

La solución adopta un enfoque modular que se apoya en:

1. **Flask** como servidor web ligero, responsable de las rutas de carga (`/upload`), listado de documentos (`/documents`) y servicio de chat (`/chat`).  
2. **TextProcessor**: módulo encargado de preprocesar texto, ejecutar tokenización, eliminación de stopwords y stemming para mejorar la calidad de los embeddings.  
3. **DocumentHandler**: componente que extrae texto de PDFs, DOCX y TXT, gestiona la caché de vectores (`vector_cache.pkl`) y fragmentos (`doc_cache.pkl`).  
4. **QAEngine**: motor de preguntas y respuestas que selecciona los fragmentos más relevantes y utiliza un pipeline de generación basado en BART para elaborar respuestas contextualizadas.

---

## 🛠️ Instalación y Configuración

Para poner en marcha **ChatBotPDFs**, siga estos pasos sin perder la paciencia:

```bash
# 1. Clonar el repositorio
git clone https://github.com/SebastianVega4/ChatBotPDFs.git
cd ChatBotPDFs

# 2. Crear entorno virtual y activar
python3 -m venv venv
source venv/bin/activate   # Linux/MacOS
venv\Scripts\activate.bat  # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Descargar recursos de NLTK
torch_version="$(python -c 'import torch; print(torch.__version__)')"
echo "Usando PyTorch $torch_version -- si desea aceleración GPU, instale la variante correspondiente"
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

## 🚀 Uso de la Aplicación

Una vez configurado el entorno, inicie el servidor con:

```bash
python app.py
```

Acceda a la interfaz en http://localhost:5000/, donde podrá:

Arrastrar y soltar documentos para procesarlos.

Visualizar el estado de cada carga en tiempo real.

Enviar preguntas en lenguaje natural y recibir respuestas detalladas en formato JSON, con explicación, nivel de confianza y referencias a fragmentos consultados.

---

## 📂 Estructura del Repositorio

```text
ChatBotPDFs/
├── app.py                    # Lógica principal de Flask y rutas
├── templates/
│   └── index.html            # Interfaz de usuario
├── static/
│   ├── css/                  # Estilos
│   └── js/                   # Scripts cliente
├── PDFS/                     # Mensajes de ejemplo (puede personalizarse)
├── vector_cache.pkl          # Caché de vectores de embeddings
├── doc_cache.pkl             # Caché de fragmentos de texto
├── requirements.txt          # Dependencias del proyecto
└── Readme-InstalacionesPythom.txt  # Guía de instalación de PyTorch
```

💻 Stack Tecnológico
Este proyecto combina:

Backend: Python 3.8+, Flask

Procesamiento de Texto: NLTK, Sentence Transformers

Modelos de Lenguaje: Transformers (BART)

Frontend: HTML5, CSS3, JavaScript

Caché: Pickle

👨‍💻 Autor y Contribuciones
Desarrollado por Johan Sebastián Vega Ruiz, estudiante de Ingeniería de Sistemas en la UPTC, Sogamoso.
Para sugerencias, reportes de errores o mejoras, abra un issue o envíe un pull request.

Contacto: sebastian.vegar2015@gmail.com | LinkedIn

© 2025 — GPL‑3.0 — Universidad Pedagógica y Tecnológica de Colombia (UPTC)

