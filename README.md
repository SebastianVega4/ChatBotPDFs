🎓 ChatBotPDFs - Asistente Académico Inteligente

[GitHub Stars](https://img.shields.io/github/stars/SebastianVega4/ChatBotPDFs?style=for-the-badge)
[Last Commit](https://img.shields.io/github/last-commit/SebastianVega4/ChatBotPDFs?style=for-the-badge)
[Languages](https://img.shields.io/github/languages/top/SebastianVega4/ChatBotPDFs?style=for-the-badge)
[License](https://img.shields.io/badge/License-GPL%203.0-blue?style=for-the-badge)

ChatBotPDFs es un proyecto personal que estoy desarrollando lentamente para ampliar mis conocimientos en PLN y aprendizaje automático. Es una plataforma web que, sin pretender reemplazar al bibliotecario, facilita la búsqueda de contenidos académicos mediante consulta directa sobre documentos en formato PDF, Word o texto plano. 

Aún faltan mejoras al chat bot y espero en el transcurso del tiempo dedicarle más para que avance el proyecto y mejore.

---

✨ Características Principales

En lugar de pasar horas hojeando manualmente decenas de páginas, ChatBotPDFs ofrece:

- Carga rápida de documentos: arrastre y suelte archivos en la interfaz web
- Normalización y fragmentación del contenido
- Embeddings semánticos con modelos de Sentence Transformers
- Mecanismo de respuesta con nivel de confianza y fuentes consultadas
- Interfaz sencilla pero funcional

---

🏗️ Arquitectura

La solución adopta un enfoque modular:

1. Flask como servidor web ligero
2. TextProcessor para preprocesamiento de texto
3. DocumentHandler para gestión de documentos
4. QAEngine para generación de respuestas

---

🛠️ Instalación y Configuración

Para poner en marcha ChatBotPDFs:

```bash
# 1. Clonar el repositorio
git clone https://github.com/SebastianVega4/ChatBotPDFs.git
cd ChatBotPDFs

# 2. Crear entorno virtual
python3 -m venv venv
source venv/bin/activate   # Linux/MacOS
venv\Scripts\activate.bat  # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Descargar recursos de NLTK
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

---

🚀 Uso de la Aplicación

Iniciar el servidor:

```bash
python app.py
```
Acceder a la interfaz en:
http://localhost:5000/

Funcionalidades:
- Arrastrar y soltar documentos
- Visualizar estado de carga
- Enviar preguntas en lenguaje natural
- Recibir respuestas con referencias

---

📂 Estructura del Repositorio

```text
ChatBotPDFs/
├── app.py                    # Lógica principal
├── templates/                # Interfaz de usuario
├── static/                   # Estilos y scripts
├── uploads/                  # archivos pdf
├── vector_cache.pkl          # Caché de vectores
├── doc_cache.pkl             # Caché de fragmentos
├── requirements.txt          # Dependencias
```

---

💻 Stack Tecnológico

Backend: Python 3.8+, Flask
Procesamiento: NLTK, Sentence Transformers
Modelos: Transformers (BART)
Frontend: HTML5, CSS3, JavaScript
Almacenamiento: Pickle

---

👨‍💻 Autor y Contacto

Desarrollado por Johan Sebastián Vega Ruiz, estudiante de Ingeniería de Sistemas en la UPTC, Sogamoso.

Este es un proyecto personal con fines de aprendizaje que iré mejorando con el tiempo.

Contacto:
sebastian.vegar2015@gmail.com
LinkedIn: [tu perfil]

© 2025 - Proyecto personal desarrollado en la UPTC