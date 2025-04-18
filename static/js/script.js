class ChatUI {
  constructor() {
    this.chatLog = document.getElementById("chatLog");
    this.questionInput = document.getElementById("pregunta");
    this.sendButton = document.getElementById("sendButton");
    this.statusInfo = document.getElementById("statusInfo");
    this.fileInput = document.getElementById("fileInput");
    this.uploadButton = document.getElementById("uploadButton");
    this.documentsList = document.getElementById("documentsList");
    this.themeToggle = document.getElementById("themeToggle");
    this.currentTheme = localStorage.getItem("theme") || "light";
    this.applyTheme();

    this.initEventListeners();
    this.loadDocumentList();
  }

  initEventListeners() {
    this.sendButton.addEventListener("click", () => this.sendQuestion());
    this.questionInput.addEventListener("keypress", (e) => {
      if (e.key === "Enter") this.sendQuestion();
    });

    this.themeToggle.addEventListener("click", () => this.toggleTheme());
    this.uploadButton.addEventListener("click", () => this.uploadFile());
  }

  toggleTheme() {
    this.currentTheme = this.currentTheme === "light" ? "dark" : "light";
    localStorage.setItem("theme", this.currentTheme);
    this.applyTheme();
    this.updateThemeIcon();
  }

  applyTheme() {
    document.documentElement.setAttribute("data-theme", this.currentTheme);
  }

  updateThemeIcon() {
    const icon = this.themeToggle.querySelector("span");
    if (this.currentTheme === "dark") {
      icon.textContent = "light_mode";
    } else {
      icon.textContent = "dark_mode";
    }
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

  // En la clase ChatUI, modifica el método uploadFile
  async uploadFile() {
    const files = this.fileInput.files;
    if (!files || files.length === 0) {
      this.showStatus("Selecciona al menos un archivo primero", "warning");
      return;
    }

    this.showStatus("Subiendo archivo(s)...", "info");
    this.uploadButton.disabled = true;

    try {
      const formData = new FormData();
      for (let i = 0; i < files.length; i++) {
        formData.append("files", files[i]); // Nota: 'files' en plural para múltiples archivos
      }

      const response = await fetch("/upload", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || "Error al subir archivo(s)");
      }

      const data = await response.json();
      this.showStatus(data.message, "success");
      this.updateDocumentList(data.documents);

      // Mostrar notificación visual de éxito
      this.showSnackbar(
        `Archivo(s) subido(s) correctamente: ${files.length}`,
        "success"
      );
    } catch (error) {
      console.error("Error:", error);
      this.showStatus(error.message, "error");
      this.showSnackbar(error.message, "error");
    } finally {
      this.uploadButton.disabled = false;
      this.fileInput.value = "";
      // Forzar recarga de la lista de documentos
      this.loadDocumentList();
    }
  }

  // Añade este método para mostrar notificaciones tipo snackbar
  showSnackbar(message, type = "info") {
    const snackbar = document.getElementById("statusInfo");
    snackbar.textContent = message;
    snackbar.className = `status-snackbar ${type}`;
    snackbar.style.display = "flex";

    setTimeout(() => {
      snackbar.style.display = "none";
    }, 5000);
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

      if (!response.ok) {
        throw new Error(data.error || "Error al obtener respuesta");
      }

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
      this.addMessage("bot", `Error: ${error.message}`, "error");
      this.showStatus(error.message, "error");
    } finally {
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
