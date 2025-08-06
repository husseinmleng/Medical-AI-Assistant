
document.addEventListener("DOMContentLoaded", () => {
    const chatInput = document.getElementById("chat-input");
    const sendButton = document.getElementById("send-button");
    const imageUpload = document.getElementById("image-upload");
    const chatMessages = document.getElementById("chat-messages");
    const chatHistory = document.getElementById("chat-history");

    let currentSessionId = null;

    // --- Event Listeners ---
    sendButton.addEventListener("click", sendMessage);
    chatInput.addEventListener("keypress", (e) => {
        if (e.key === "Enter") {
            sendMessage();
        }
    });
    imageUpload.addEventListener("change", uploadImage);

    // --- Functions ---
    async function sendMessage() {
        const message = chatInput.value.trim();
        if (!message) return;

        appendMessage(message, "user");
        chatInput.value = "";

        try {
            const response = await fetch("/chat", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ 
                    message: message,
                    session_id: currentSessionId
                }),
            });

            const data = await response.json();
            currentSessionId = data.session_id;
            displayMessages(data.messages);
            updateChatHistory();
        } catch (error) {
            console.error("Error sending message:", error);
        }
    }

    async function uploadImage() {
        const file = imageUpload.files[0];
        if (!file) return;

        const formData = new FormData();
        formData.append("file", file);

        try {
            const response = await fetch(`/upload_image?session_id=${currentSessionId || ''}`, {
                method: "POST",
                body: formData,
            });

            const data = await response.json();
            currentSessionId = data.session_id;
            displayMessages(data.messages);
            if (data.annotated_image_path) {
                appendImage(data.annotated_image_path);
            }
            updateChatHistory();
        } catch (error) {
            console.error("Error uploading image:", error);
        }
    }

    function displayMessages(messages) {
        chatMessages.innerHTML = "";
        messages.forEach(msg => {
            appendMessage(msg.content, msg.role);
        });
    }

    function appendMessage(content, role) {
        const messageElement = document.createElement("div");
        messageElement.classList.add("message", role === "user" ? "user-message" : "assistant-message");
        messageElement.innerText = content;
        chatMessages.appendChild(messageElement);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function appendImage(imagePath) {
        const imageElement = document.createElement("img");
        imageElement.src = imagePath;
        imageElement.classList.add("message", "assistant-message");
        chatMessages.appendChild(imageElement);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    async function updateChatHistory() {
        // This is a placeholder for fetching and displaying chat history.
        // A more complete implementation would fetch a list of conversations
        // from the backend.
        if (currentSessionId) {
            const historyResponse = await fetch(`/history/${currentSessionId}`);
            const historyData = await historyResponse.json();
            
            const historyElement = document.createElement('div');
            historyElement.innerText = historyData.title;
            historyElement.classList.add('history-item');
            historyElement.onclick = () => switchConversation(currentSessionId);
            chatHistory.appendChild(historyElement);
        }
    }

    async function switchConversation(sessionId) {
        const historyResponse = await fetch(`/history/${sessionId}`);
        const historyData = await historyResponse.json();
        currentSessionId = sessionId;
        displayMessages(historyData.messages);
    }
});
