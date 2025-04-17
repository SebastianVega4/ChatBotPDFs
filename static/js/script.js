class ChatUI {
  constructor() {
    this.chatLog = document.getElementById("chatLog");
    this.questionInput = document.getElementById("pregunta");
    this.sendButton = document.getElementById("sendButton");
    this.statusInfo = document.getElementById("statusInfo");
    this.fileInput = document.getElementById("fileInput");
    this.uploadButton = document.getElementById("uploadButton");
    this.documentsList = document.getElementById("documentsList");

    this.initEventListeners();
    this.loadDocumentList();
  }

  initEventListeners() {
    this.sendButton.addEventListener("click", () => this.sendQuestion());
    this.questionInput.addEventListener("keypress", (e) => {
      if (e.key === "Enter") this.sendQuestion();
    });

    this.uploadButton.addEventListener("click", () => this.uploadFile());
  }

  async loadDocumentList() {
    try {
      this.showStatus("Cargando lista de documentos...", "info");
      const response = await fetch("/documents");
      if (!response.ok) throw new Error("Error al cargar documentos");

      const data = await response.json();
      this.updateDocumentList(data.documents);

      if (data.documents.length === 0) {
        this.showStatus(
          "No hay documentos cargados. Sube archivos PDF, DOCX o TXT.",
          "warning"
        );
      } else {
        this.showStatus(
          `Lista de documentos actualizada (${data.documents.length} documentos)`,
          "success"
        );
      }
    } catch (error) {
      console.error("Error:", error);
      this.showStatus("Error al cargar la lista de documentos", "error");
    }
  }

  updateDocumentList(documents) {
    this.documentsList.innerHTML = documents.length
      ? documents.map((doc) => `<li>${doc}</li>`).join("")
      : "<li>No hay documentos cargados</li>";
  }

  async uploadFile() {
    const file = this.fileInput.files[0];
    if (!file) {
      this.showStatus("Selecciona un archivo primero", "warning");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    this.showStatus("Subiendo archivo...", "info");
    this.uploadButton.disabled = true;

    try {
      const response = await fetch("/upload", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || "Error al subir archivo");
      }

      const data = await response.json();
      this.showStatus(data.message, "success");
      this.updateDocumentList(data.documents);
    } catch (error) {
      console.error("Error:", error);
      this.showStatus(error.message, "error");
    } finally {
      this.uploadButton.disabled = false;
      this.fileInput.value = "";
    }
  }

  async sendQuestion() {
    const question = this.questionInput.value.trim();
    if (!question) return;

    this.addMessage("user", question);
    this.questionInput.value = "";
    this.showStatus("Procesando tu pregunta...", "info");
    this.sendButton.disabled = true;

    try {
      const response = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pregunta: question }),
      });

      if (!response.ok) {
        throw new Error(`Error HTTP: ${response.status}`);
      }

      const data = await response.json();

      // Manejo robusto de la respuesta
      const answer = data.answer || "No se encontró respuesta";
      const score = data.score || 0;
      const confidence = data.confidence || "N/A";
      const sources = data.sources || [];
      const error = data.error || null;

      if (error) {
        this.showStatus(error, "error");
      }

      const confidenceClass = this.getConfidenceClass(confidence);
      let sourcesText = "";

      if (sources.length > 0) {
        sourcesText = `<div class="sources">Fuente: ${sources.join(
          ", "
        )}</div>`;
      }

      this.addMessage(
        "bot",
        answer,
        confidenceClass,
        `Confianza: ${confidence} (${(score * 100).toFixed(1)}%)`,
        sourcesText
      );
    } catch (error) {
      console.error("Error:", error);
      this.addMessage(
        "bot",
        "Error al obtener respuesta del servidor",
        "error"
      );
    } finally {
      this.showStatus("", "info");
      this.sendButton.disabled = false;
      this.scrollToBottom();
    }
  }

  addMessage(
    sender,
    text,
    confidenceClass = "",
    confidenceText = "",
    extraHTML = ""
  ) {
    const messageDiv = document.createElement("div");
    messageDiv.className = `message ${sender}`;
    messageDiv.innerHTML = `
            <div class="message-content">${text}</div>
            ${
              confidenceText
                ? `<div class="confidence ${confidenceClass}">${confidenceText}</div>`
                : ""
            }
            ${extraHTML}
        `;
    this.chatLog.appendChild(messageDiv);
    this.scrollToBottom();
  }

  showStatus(text, type = "info") {
    this.statusInfo.textContent = text;
    this.statusInfo.className = `status-info ${type}`;
  }

  scrollToBottom() {
    this.chatLog.scrollTop = this.chatLog.scrollHeight;
  }

  getConfidenceClass(confidence) {
    if (!confidence) return "";
    if (confidence.includes("Alta")) return "high";
    if (confidence.includes("Media")) return "medium";
    if (confidence.includes("Baja")) return "low";
    return "very-low";
  }
}

// Initialize the chat when DOM is loaded
document.addEventListener("DOMContentLoaded", () => {
  new ChatUI();
});
