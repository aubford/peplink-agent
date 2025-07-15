let currentThreadId = null
let isStreaming = false

async function loadData() {
  try {
    const response = await fetch("/api/threads")
    const data = await response.json()

    const threadsList = document.getElementById("threadsList")
    threadsList.innerHTML = ""

    data.threads.forEach(thread => {
      const threadElement = document.createElement("div")
      threadElement.className = "thread-item"
      threadElement.setAttribute("data-thread-id", thread.thread_id)
      threadElement.onclick = e => {
        // Don't select thread if delete button was clicked
        if (e.target.closest(".delete-thread-btn")) {
          return
        }
        e.preventDefault()
        selectThread(thread.thread_id, threadElement)
      }

      threadElement.innerHTML = `
          <div class="thread-content">
              <div class="thread-title">${thread.title}</div>
              <div class="thread-meta">${thread.message_count} messages</div>
          </div>
          <button class="delete-thread-btn" onclick="deleteThread('${thread.thread_id}', event)" title="Delete conversation">
              🗑️
          </button>
      `

      threadsList.appendChild(threadElement)
    })

    // Only auto-select if we're reloading threads for an existing conversation
    // Otherwise, show the no messages screen by default
    if (currentThreadId && data.threads.find(t => t.thread_id === currentThreadId)) {
      // Maintain current selection if it still exists
      await selectThread(currentThreadId)
    } else if (data.threads.length === 0) {
      // Show no messages screen if no threads exist
      showNoMessagesScreen()
    } else {
      // Show no messages screen by default when app loads
      showNoMessagesScreen()
    }
  } catch (error) {
    console.error("Error loading threads:", error)
    showError("Failed to load conversations")
  }
}

// Load threads on page load
document.addEventListener("DOMContentLoaded", function () {
  loadData()
  loadTestsetSuggestions()
  setupTextareaAutoResize()
})

async function sendMessage(event) {
  event.preventDefault()

  if (isStreaming) {
    return
  }

  const messageInput = document.getElementById("messageInput")
  const message = messageInput.value.trim()

  if (!message) return

  // Generate a new thread ID if we don't have one
  if (!currentThreadId) {
    currentThreadId = generateUUID()
    document.getElementById("currentThread").textContent = `Thread: ${currentThreadId}`
  }

  // Add user message to chat
  addMessage(message, "user")
  messageInput.value = ""
  messageInput.style.height = "auto"

  // Disable input and show loading
  setStreamingState(true)

  try {
    // Create assistant message element for streaming
    const assistantMessage = addMessage("", "assistant", true)

    // Start streaming
    const response = await fetch("/api/chat/stream", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        message: message,
        thread_id: currentThreadId,
      }),
    })

    const reader = response.body.getReader()
    const decoder = new TextDecoder()

    while (true) {
      const { done, value } = await reader.read()

      if (done) break

      const chunk = decoder.decode(value)
      const lines = chunk.split("\n")

      for (const line of lines) {
        if (line.startsWith("data: ")) {
          try {
            const data = JSON.parse(line.slice(6))

            if (data.type === "token") {
              assistantMessage.textContent += data.content
              scrollToBottom()
            } else if (data.type === "complete") {
              assistantMessage.classList.remove("streaming")

              // Always reload threads to update message count in sidebar
              await loadData()
            } else if (data.type === "error") {
              showError(`Error: ${data.message}`)
              assistantMessage.textContent =
                "Sorry, I encountered an error processing your request."
              assistantMessage.classList.remove("streaming")
            }
          } catch (e) {
            // Ignore JSON parse errors for incomplete chunks
          }
        }
      }
    }
  } catch (error) {
    console.error("Error sending message:", error)
    showError("Failed to send message")
  } finally {
    setStreamingState(false)
  }
}

function showNoMessagesScreen() {
  currentThreadId = null

  const chatMessages = document.getElementById("chatMessages")
  const welcomeMessage = document.getElementById("welcomeMessage")

  // Clear any existing messages
  const existingMessages = chatMessages.querySelectorAll(".message-wrapper")
  existingMessages.forEach(msg => msg.remove())

  // Show welcome message
  chatMessages.classList.remove("has-messages")
  welcomeMessage.style.display = "block"

  document.getElementById("currentThread").textContent = "Start a new conversation"
}

function startNewChat() {
  // Show no messages screen without creating a thread yet
  showNoMessagesScreen()

  // Close sidebar on mobile
  if (window.innerWidth <= 768) {
    toggleSidebar()
  }
}

async function selectThread(threadId, clickedElement = null) {
  currentThreadId = threadId

  // Update UI - remove active class from all threads
  document.querySelectorAll(".thread-item").forEach(item => {
    item.classList.remove("active")
  })

  // Add active class to the selected thread
  if (clickedElement) {
    clickedElement.classList.add("active")
  } else {
    // Find thread by data attribute when called programmatically
    const targetThread = document.querySelector(`[data-thread-id="${threadId}"]`)
    if (targetThread) {
      targetThread.classList.add("active")
    }
  }

  document.getElementById("currentThread").textContent = `Thread: ${threadId}`

  // Load thread history
  await loadThreadHistory(threadId)

  // Close sidebar on mobile after selecting thread
  if (window.innerWidth <= 768) {
    toggleSidebar()
  }
}

async function sendSuggestion(message) {
  if (isStreaming) return

  const messageInput = document.getElementById("messageInput")
  messageInput.value = message

  // Trigger the send message function
  const event = new Event("submit")
  document.querySelector(".chat-input-form").dispatchEvent(event)
}

async function loadThreadHistory(threadId) {
  try {
    const response = await fetch(`/api/threads/${threadId}/history`)
    const data = await response.json()

    const chatMessages = document.getElementById("chatMessages")
    const welcomeMessage = document.getElementById("welcomeMessage")

    // Clear existing messages except welcome
    const existingMessages = chatMessages.querySelectorAll(".message-wrapper")
    existingMessages.forEach(msg => msg.remove())

    if (data.messages.length === 0) {
      chatMessages.classList.remove("has-messages")
      welcomeMessage.style.display = "block"
    } else {
      chatMessages.classList.add("has-messages")
      welcomeMessage.style.display = "none"

      data.messages.forEach(msg => {
        addMessage(msg.content, msg.type === "human" ? "user" : "assistant")
      })
    }

    scrollToBottom()
  } catch (error) {
    console.error("Error loading thread history:", error)
    showError("Failed to load conversation history")
  }
}

function addMessage(content, type, streaming = false) {
  const chatMessages = document.getElementById("chatMessages")
  const welcomeMessage = document.getElementById("welcomeMessage")

  // Hide welcome message when first message is added
  if (!chatMessages.classList.contains("has-messages")) {
    chatMessages.classList.add("has-messages")
    welcomeMessage.style.display = "none"
  }

  const messageWrapper = document.createElement("div")
  messageWrapper.className = `message-wrapper ${type}`

  const messageContent = document.createElement("div")
  messageContent.className = "message-content"

  const avatar = document.createElement("div")
  avatar.className = `message-avatar ${type}`
  avatar.textContent = type === "user" ? "U" : "AI"

  const messageText = document.createElement("div")
  messageText.className = "message-text"
  if (streaming) {
    messageText.classList.add("streaming")
  }
  messageText.textContent = content

  messageContent.appendChild(avatar)
  messageContent.appendChild(messageText)
  messageWrapper.appendChild(messageContent)
  chatMessages.appendChild(messageWrapper)

  scrollToBottom()
  return messageText
}

async function deleteThread(threadId, event) {
  event.stopPropagation() // Prevent thread selection

  try {
    const response = await fetch(`/api/threads/${threadId}`, {
      method: "DELETE",
    })

    if (!response.ok) {
      throw new Error("Failed to delete thread")
    }

    // If we deleted the currently selected thread, show no messages screen
    if (currentThreadId === threadId) {
      showNoMessagesScreen()
    }

    // Reload threads to update the sidebar
    await loadData()
  } catch (error) {
    console.error("Error deleting thread:", error)
    showError("Failed to delete conversation")
  }
}

function setStreamingState(streaming) {
  isStreaming = streaming
  const sendBtn = document.getElementById("sendBtn")
  const messageInput = document.getElementById("messageInput")

  sendBtn.disabled = streaming
  messageInput.disabled = streaming
}

function scrollToBottom() {
  const chatMessages = document.getElementById("chatMessages")
  chatMessages.scrollTop = chatMessages.scrollHeight
}

function showError(message) {
  const chatMessages = document.getElementById("chatMessages")
  const errorElement = document.createElement("div")
  errorElement.className = "error"
  errorElement.textContent = message
  chatMessages.appendChild(errorElement)
  scrollToBottom()

  // Remove error after 5 seconds
  setTimeout(() => {
    errorElement.remove()
  }, 15000)
}

// Handle Enter key in textarea
document.getElementById("messageInput").addEventListener("keydown", function (event) {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault()
    sendMessage(event)
  }
})

// Handle window resize
window.addEventListener("resize", function () {
  if (window.innerWidth > 768) {
    const sidebar = document.getElementById("sidebar")
    const overlay = document.getElementById("sidebarOverlay")
    sidebar.classList.remove("open")
    overlay.classList.remove("show")
  }
})

async function loadTestsetSuggestions() {
  const response = await fetch("/api/testset-queries")
  const data = await response.json()

  const testsetSuggestions = document.getElementById("testsetSuggestions")

  // Add testset suggestions after the existing static ones
  data.queries.forEach(queryObj => {
    const suggestionCard = document.createElement("div")
    suggestionCard.className = "suggestion-card"
    suggestionCard.onclick = () => sendSuggestion(queryObj.query)

    suggestionCard.innerHTML = `
                <div class="suggestion-title">${queryObj.query}</div>
            `
    testsetSuggestions.appendChild(suggestionCard)
  })
}

function setupTextareaAutoResize() {
  const textarea = document.getElementById("messageInput")
  textarea.addEventListener("input", function () {
    this.style.height = "auto"
    this.style.height = Math.min(this.scrollHeight, 200) + "px"
  })
}

function toggleSidebar() {
  const sidebar = document.getElementById("sidebar")
  const overlay = document.getElementById("sidebarOverlay")

  sidebar.classList.toggle("open")
  overlay.classList.toggle("show")
}

function generateUUID() {
  return crypto.randomUUID()
}
