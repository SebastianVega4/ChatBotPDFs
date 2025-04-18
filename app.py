from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import re
import numpy as np
import nltk
import torch  # Añadir esta importación
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from transformers import pipeline
from concurrent.futures import ThreadPoolExecutor
import pdfplumber
from docx import Document
import pickle
import hashlib
from typing import List, Dict, Optional
from pathlib import Path

# Descargar recursos de NLTK al inicio
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)  # Añadir este recurso faltante

# Configuración global
class Config:
    MAX_CONTEXTS = 5  # Aumentado para mejor precisión
    MAX_FRAGMENT_SIZE = 500  # Fragmentos más grandes para mejor contexto
    N_THREADS = 4
    VECTOR_CACHE = "vector_cache.pkl"
    DOCUMENT_CACHE = "doc_cache.pkl"
    UPLOAD_FOLDER = os.path.abspath("./uploads")  # Ruta absoluta
    ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}
    HOST = "0.0.0.0"
    PORT = 5000
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # Modelo pequeño y eficiente
    GENERATION_MODEL = "facebook/bart-large-cnn"  # Para resumir/generar respuestas
    MODEL_NAME = "bert-base-multilingual-cased" 
    #SIMILARITY_THRESHOLD = 0.6  # Umbral para considerar una coincidencia relevante
    SIMILARITY_THRESHOLD = 0.4  # Reducir el umbral para mayor flexibilidad

# Descargar recursos de NLTK
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# Inicialización de Flask
app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = Config.UPLOAD_FOLDER
Path(Config.UPLOAD_FOLDER).mkdir(exist_ok=True)

class TextProcessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('spanish'))
        self.stemmer = SnowballStemmer('spanish')
        self.punctuation = re.compile(r'[^\w\s]')
        # Asegurar que los recursos de NLTK estén disponibles
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            nltk.download('punkt_tab')

    def preprocess_text(self, text: str) -> str:
        """Preprocesa el texto eliminando stopwords y aplicando stemming."""
        # Manejo de saludos básicos
        greetings = {
            'hola': 'saludo',
            'holi': 'saludo',
            'buenos días': 'saludo',
            'buenas tardes': 'saludo',
            'buenas noches': 'saludo'
        }
        """Preprocesa el texto eliminando stopwords y aplicando stemming."""
        text_lower = text.lower()
        for greeting, replacement in greetings.items():
            if greeting in text_lower:
                return replacement
        try:
            # Limpieza básica
            text = text.lower()
            text = self.punctuation.sub(' ', text)
        
            # Tokenización y stemming
            tokens = nltk.word_tokenize(text)
            tokens = [self.stemmer.stem(token) for token in tokens if token not in self.stop_words and len(token) > 2]
        
            return ' '.join(tokens)
        except Exception as e:
            print(f"Error en preprocesamiento: {str(e)}")
            return text  # Devolver texto sin procesar si hay error

    def split_text(self, text: str, max_length: int = Config.MAX_FRAGMENT_SIZE) -> List[str]:
        """Divide el texto en fragmentos manteniendo párrafos intactos cuando es posible."""
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        fragments = []
        current_fragment = []
        current_length = 0

        for para in paragraphs:
            para_length = len(para.split())
            if current_length + para_length <= max_length:
                current_fragment.append(para)
                current_length += para_length
            else:
                if current_fragment:
                    fragments.append('\n'.join(current_fragment))
                current_fragment = [para]
                current_length = para_length

        if current_fragment:
            fragments.append('\n'.join(current_fragment))

        return fragments

class DocumentHandler:
    def __init__(self):
        self.text_processor = TextProcessor()
        self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
        self.documents = []
        self.document_embeddings = None
        self.document_texts = []
        
    def _file_hash(self, file_path: str) -> str:
        """Calcula el hash de un archivo para detectar cambios."""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            print(f"Error calculando hash: {str(e)}")
            return ""

    def extract_text(self, file_path: str) -> str:
        """Extrae texto de diferentes tipos de archivos con manejo de errores."""
        try:
            ext = os.path.splitext(file_path)[1].lower()
            
            if ext == '.pdf':
                text = ''
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        try:
                            page_text = page.extract_text() or ''
                            text += page_text + '\n'
                        except Exception as e:
                            print(f"Error en página PDF: {str(e)}")
                            continue
                return text.strip()
            
            elif ext == '.docx':
                doc = Document(file_path)
                return '\n'.join(para.text for para in doc.paragraphs if para.text.strip())
            
            elif ext == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            
            else:
                raise ValueError(f"Formato no soportado: {ext}")
                
        except Exception as e:
            print(f"Error extrayendo texto de {file_path}: {str(e)}")
            return ""

    def load_documents(self, directory: str = Config.UPLOAD_FOLDER) -> bool:
        """Carga y procesa documentos con mejor manejo de errores."""
        try:
            print(f"\nIntentando cargar documentos desde: {directory}")
            
            # Verificar si el directorio existe
            if not os.path.exists(directory):
                print(f"¡Error! El directorio {directory} no existe")
                return False
                
            # Listar contenido del directorio
            print("Contenido del directorio:")
            for f in os.listdir(directory):
                print(f" - {f}")
            
            # Filtrar archivos válidos
            valid_files = [
                f for f in os.listdir(directory) 
                if any(f.lower().endswith(ext) for ext in Config.ALLOWED_EXTENSIONS)
            ]
            print(f"\nArchivos válidos encontrados: {valid_files}")
            
            if not valid_files:
                print("No hay archivos con extensiones válidas (.pdf, .docx, .txt)")
                return False
                
            # Procesar cada archivo
            self.documents = []
            for filename in valid_files:
                file_path = os.path.join(directory, filename)
                try:
                    text = self.extract_text(file_path)
                    if text.strip():
                        fragments = self.text_processor.split_text(text)
                        
                        self.documents.append({
                            'file_name': filename,
                            'content': text,
                            'fragments': fragments
                        })
                        print(f"Documento procesado: {filename}")
                except Exception as e:
                    print(f"Error procesando {filename}: {str(e)}")
                    continue
            
            if not self.documents:
                print("No se pudieron procesar documentos válidos")
                return False
                
            # Generar embeddings para todos los fragmentos
            all_fragments = [f for doc in self.documents for f in doc['fragments']]
            if all_fragments:
                print("Generando embeddings para los fragmentos...")
                self.document_embeddings = self.embedding_model.encode(all_fragments)
                self.document_texts = all_fragments
                print("Embeddings generados exitosamente")
            
            print(f"Documentos cargados: {len(self.documents)}")
            return True
            
        except Exception as e:
            print(f"Error crítico en load_documents: {str(e)}")
            return False
            
    def find_relevant_contexts(self, question: str, top_k: int = 3) -> List[Dict]:
        """Encuentra los contextos más relevantes con sus puntuaciones de similitud."""
        if not self.document_texts or self.document_embeddings is None:
            return []
            
        # Embedding de la pregunta
        question_embedding = self.embedding_model.encode([question])
        
        # Calcular similitudes
        similarities = cosine_similarity(
            question_embedding,
            self.document_embeddings
        )[0]
        
        # Obtener los índices de los top_k más relevantes
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Filtrar por umbral de similitud
        results = []
        for idx in top_indices:
            if similarities[idx] >= Config.SIMILARITY_THRESHOLD:
                results.append({
                    "text": self.document_texts[idx],
                    "score": float(similarities[idx]),
                    "source": self._find_document_source(self.document_texts[idx])
                })
        
        return results
        
    def _find_document_source(self, fragment: str) -> str:
        """Encuentra el documento de origen para un fragmento de texto."""
        for doc in self.documents:
            if fragment in doc['fragments']:
                return doc['file_name']
        return "Desconocido"


class QAEngine:
    def __init__(self):
        self.document_handler = DocumentHandler()
        self.text_processor = TextProcessor()
        
        try:
            # Cargar modelo de generación
            self.generation_pipe = pipeline(
                "text2text-generation", 
                model=Config.GENERATION_MODEL,
                device=0 if torch.cuda.is_available() else -1
            )
        except Exception as e:
            print(f"Error loading generation model: {str(e)}")
            self.generation_pipe = None
    
    def _get_confidence_level(self, score: float) -> str:
        """Devuelve un nivel de confianza legible para el usuario."""
        if score > 0.75:
            return "Alta confianza"
        elif score > 0.5:
            return "Media confianza"
        elif score > 0.3:
            return "Baja confianza"
        else:
            return "Muy baja confianza"
    
    def answer_question(self, question: str) -> Dict[str, str]:
        # Manejar saludos primero
        greetings_responses = {
            'saludo': "¡Hola! Soy un asistente universitario. ¿En qué puedo ayudarte hoy?",
            'cómo estás': "Estoy aquí para ayudarte con información académica. ¿Qué necesitas saber?",
            'quién eres': "Soy un chatbot diseñado para responder preguntas sobre documentos universitarios."
        }
        
        preprocessed = self.text_processor.preprocess_text(question)
        if preprocessed in greetings_responses:
            return {
                "answer": greetings_responses[preprocessed],
                "score": 0.9,
                "confidence": "Alta confianza",
                "sources": []
            }
        
        # Buscar contextos relevantes
        contexts = self.document_handler.find_relevant_contexts(question)
        
        if not contexts:
            return {
                "answer": "No encontré información relevante en los documentos cargados para responder a tu pregunta.",
                "score": 0.0,
                "confidence": "N/A",
                "sources": []
            }
        
        # Preparar el contexto para la generación
        context_str = "\n".join([f"Contenido: {c['text']}" for c in contexts[:2]])  # Usar máximo 2 contextos
        sources = list(set([c['source'] for c in contexts]))
        
        # Generar respuesta
        try:
            if self.generation_pipe is not None:
                generated = self.generation_pipe(
                    f"Pregunta: {question}\nContexto: {context_str}",
                    max_length=200,
                    do_sample=False
                )
                answer = generated[0]['generated_text']
            else:
                # Fallback: usar el fragmento más relevante
                answer = contexts[0]['text']
            
            return {
                "answer": answer,
                "score": float(contexts[0]['score']),
                "confidence": self._get_confidence_level(contexts[0]['score']),
                "sources": sources
            }
            
        except Exception as e:
            print(f"Error generating answer: {str(e)}")
            # Fallback: respuesta genérica
            return {
                "answer": f"Basado en los documentos: {contexts[0]['text']}",
                "score": contexts[0]['score'],
                "confidence": self._get_confidence_level(contexts[0]['score']),
                "sources": sources
            }
        
# Inicializar el motor QA
qa_engine = QAEngine()

@app.route("/")
def home():
    return render_template('index.html')

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@app.route("/upload", methods=["POST"])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No se proporcionó ningún archivo"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Nombre de archivo vacío"}), 400
    
    if file and any(file.filename.lower().endswith(ext) for ext in Config.ALLOWED_EXTENSIONS):
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        
        # Recargar documentos
        success = qa_engine.document_handler.load_documents()
        return jsonify({
            "message": "Archivo subido correctamente",
            "reloaded": success,
            "documents": [doc['file_name'] for doc in qa_engine.document_handler.documents]
        }), 200
    
    return jsonify({"error": "Tipo de archivo no permitido"}), 400

@app.route("/documents", methods=["GET"])
def list_documents():
    return jsonify({
        "documents": [doc['file_name'] for doc in qa_engine.document_handler.documents]
    })

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        if not data or "pregunta" not in data:
            return jsonify({
                "error": "Se requiere un campo 'pregunta'",
                "answer": "",
                "score": 0.0,
                "confidence": "N/A",
                "sources": []
            }), 400

        question = data["pregunta"].strip()
        if not question:
            return jsonify({
                "error": "La pregunta no puede estar vacía",
                "answer": "",
                "score": 0.0,
                "confidence": "N/A",
                "sources": []
            }), 400

        response = qa_engine.answer_question(question)
        
        # Asegurar que la respuesta tenga todos los campos necesarios
        required_fields = ["answer", "score", "confidence", "sources"]
        for field in required_fields:
            if field not in response:
                response[field] = "" if field == "answer" else 0.0 if field == "score" else "N/A" if field == "confidence" else []
        
        return jsonify(response), 200

    except Exception as e:
        print(f"Error en /chat: {str(e)}")
        return jsonify({
            "error": f"Error procesando solicitud: {str(e)}",
            "answer": "Error interno del servidor",
            "score": 0.0,
            "confidence": "N/A",
            "sources": []
        }), 500

if __name__ == "__main__":
    # Asegurar que el directorio uploads existe
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    
    # Cargar documentos al iniciar
    print("Cargando documentos...")
    if not qa_engine.document_handler.load_documents():
        print("Advertencia: No se encontraron documentos para cargar")
    else:
        print(f"Documentos cargados exitosamente: {len(qa_engine.document_handler.documents)}")
    
    app.run(host=Config.HOST, port=Config.PORT, debug=True)