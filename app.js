/* ==========================================================================
   AI Search Engine — Frontend logic
   Vanilla JS, no build step. Talks to the FastAPI backend under /api/*.
   ========================================================================== */

const API = {
  health: "/api/health",
  upload: "/api/documents/upload",
  documents: "/api/documents",
  search: "/api/search",
  compare: "/api/compare",
  history: "/api/search/history",
  settings: "/api/settings",
};

// ---------------------------------------------------------------- navigation

const navItems = document.querySelectorAll(".nav-item");
const views = document.querySelectorAll(".view");

navItems.forEach((btn) => {
  btn.addEventListener("click", () => {
    const target = btn.dataset.view;
    navItems.forEach((b) => b.classList.toggle("is-active", b === btn));
    views.forEach((v) => v.classList.toggle("is-active", v.id === `view-${target}`));

    if (target === "documents") loadDocuments();
    if (target === "history") loadHistory();
    if (target === "settings") loadSettings();
  });
});

// ---------------------------------------------------------------- theme

const themeToggle = document.getElementById("themeToggle");
themeToggle.addEventListener("click", () => {
  const root = document.documentElement;
  const isLight = root.getAttribute("data-theme") === "light";
  root.setAttribute("data-theme", isLight ? "dark" : "light");
  themeToggle.textContent = isLight ? "☾" : "☀";
});

// ---------------------------------------------------------------- health

async function checkHealth() {
  const dot = document.getElementById("healthDot");
  const label = document.getElementById("healthLabel");
  const chip = document.getElementById("activeProviderChip");
  try {
    const res = await fetch(API.health);
    const data = await res.json();
    dot.classList.add("ok");
    dot.classList.remove("bad");
    label.textContent = data.openai_configured ? "online" : "online · no API key";
    chip.textContent = data.vector_db_provider;
  } catch (err) {
    dot.classList.add("bad");
    label.textContent = "offline";
  }
}

// ---------------------------------------------------------------- search

const searchForm = document.getElementById("searchForm");
const resultsList = document.getElementById("resultsList");
const answerCard = document.getElementById("answerCard");
const answerBody = document.getElementById("answerBody");
const searchMeta = document.getElementById("searchMeta");

searchForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const query = document.getElementById("searchInput").value.trim();
  if (!query) return;

  const mode = document.getElementById("searchMode").value;
  const provider = document.getElementById("searchProvider").value;
  const submitBtn = searchForm.querySelector("button[type=submit]");

  submitBtn.disabled = true;
  submitBtn.textContent = "Searching…";
  searchMeta.textContent = "";
  answerCard.hidden = true;
  resultsList.innerHTML = `<div class="empty-state">Embedding query and searching ${provider}…</div>`;

  try {
    const res = await fetch(API.search, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        mode,
        provider,
        top_k: 5,
        generate_answer: true,
      }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Search failed (${res.status})`);
    }

    const data = await res.json();
    renderSearchResults(data);
  } catch (err) {
    resultsList.innerHTML = `<div class="empty-state">⚠ ${escapeHtml(err.message)}</div>`;
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Search";
  }
});

function renderSearchResults(data) {
  searchMeta.textContent = `${data.results.length} result(s) · ${data.provider} · ${data.mode} · ${data.latency_ms}ms`;

  if (data.answer) {
    answerCard.hidden = false;
    answerBody.textContent = data.answer;
  } else {
    answerCard.hidden = true;
  }

  if (data.results.length === 0) {
    resultsList.innerHTML = `<div class="empty-state">No matching chunks found. Try uploading a relevant document first.</div>`;
    return;
  }

  resultsList.innerHTML = data.results
    .map((r) => {
      const pct = Math.round(r.score * 100);
      return `
        <div class="result-card">
          <div class="result-top">
            <span class="result-file">${escapeHtml(r.filename)}</span>
            ${scoreAxis(r.score)}
          </div>
          <div class="result-text">${escapeHtml(r.text)}</div>
        </div>
      `;
    })
    .join("");
}

function scoreAxis(score) {
  const pct = Math.max(0, Math.min(100, Math.round(score * 100)));
  return `
    <div class="score-axis">
      <div class="score-axis-track">
        <div class="score-axis-fill" style="width:${pct}%"></div>
        <div class="score-axis-dot" style="left:${pct}%"></div>
      </div>
      <span class="score-value">${pct}%</span>
    </div>
  `;
}

// ---------------------------------------------------------------- documents

const uploadZone = document.getElementById("uploadZone");
const fileInput = document.getElementById("fileInput");
const uploadStatus = document.getElementById("uploadStatus");

uploadZone.addEventListener("click", () => fileInput.click());

["dragenter", "dragover"].forEach((evt) =>
  uploadZone.addEventListener(evt, (e) => {
    e.preventDefault();
    uploadZone.classList.add("is-dragover");
  })
);
["dragleave", "drop"].forEach((evt) =>
  uploadZone.addEventListener(evt, (e) => {
    e.preventDefault();
    uploadZone.classList.remove("is-dragover");
  })
);
uploadZone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files[0];
  if (file) uploadFile(file);
});
fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) uploadFile(fileInput.files[0]);
});

async function uploadFile(file) {
  uploadStatus.classList.remove("is-error");
  uploadStatus.textContent = `Uploading ${file.name}…`;

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch(API.upload, { method: "POST", body: formData });
    const data = await res.json().catch(() => ({}));

    if (!res.ok) {
      throw new Error(data.detail || `Upload failed (${res.status})`);
    }

    uploadStatus.textContent = `✓ ${data.filename} indexed into ${data.chunk_count} chunks (${data.vector_db_provider}).`;
    loadDocuments();
  } catch (err) {
    uploadStatus.classList.add("is-error");
    uploadStatus.textContent = `⚠ ${err.message}`;
  } finally {
    fileInput.value = "";
  }
}

async function loadDocuments() {
  const tbody = document.getElementById("docTableBody");
  try {
    const res = await fetch(API.documents);
    const data = await res.json();

    if (data.documents.length === 0) {
      tbody.innerHTML = `<tr><td colspan="7" class="empty-row">No documents yet — upload one above.</td></tr>`;
      return;
    }

    tbody.innerHTML = data.documents
      .map(
        (doc) => `
        <tr>
          <td>${escapeHtml(doc.filename)}</td>
          <td>${escapeHtml(doc.file_type)}</td>
          <td><span class="status-pill ${doc.status}">${doc.status}</span></td>
          <td>${doc.chunk_count}</td>
          <td>${escapeHtml(doc.vector_db_provider)}</td>
          <td>${formatDate(doc.uploaded_at)}</td>
          <td><button class="row-delete" data-id="${doc.document_id}">Delete</button></td>
        </tr>
      `
      )
      .join("");

    tbody.querySelectorAll(".row-delete").forEach((btn) => {
      btn.addEventListener("click", () => deleteDocument(btn.dataset.id));
    });
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="7" class="empty-row">⚠ Failed to load documents.</td></tr>`;
  }
}

async function deleteDocument(documentId) {
  if (!confirm("Delete this document and its indexed vectors?")) return;
  try {
    const res = await fetch(`${API.documents}/${documentId}`, { method: "DELETE" });
    if (!res.ok) throw new Error("Delete failed");
    loadDocuments();
  } catch (err) {
    alert(err.message);
  }
}

// ---------------------------------------------------------------- compare

const compareForm = document.getElementById("compareForm");
const compareGrid = document.getElementById("compareGrid");

compareForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const query = document.getElementById("compareInput").value.trim();
  if (!query) return;

  compareGrid.innerHTML = `<div class="empty-state">Querying Chroma, Pinecone, and Qdrant…</div>`;

  try {
    const res = await fetch(API.compare, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, top_k: 5, providers: ["chroma", "pinecone", "qdrant"] }),
    });
    const data = await res.json();
    renderCompare(data);
  } catch (err) {
    compareGrid.innerHTML = `<div class="empty-state">⚠ ${escapeHtml(err.message)}</div>`;
  }
});

function renderCompare(data) {
  compareGrid.innerHTML = data.groups
    .map((g) => {
      const body = g.error
        ? `<div class="compare-item">⚠ ${escapeHtml(g.error)}</div>`
        : g.results.length === 0
        ? `<div class="compare-item">No results.</div>`
        : g.results
            .map(
              (r) => `<div class="compare-item">${escapeHtml(truncate(r.text, 140))}<br>${scoreAxis(r.score)}</div>`
            )
            .join("");

      return `
        <div class="compare-col">
          <div class="compare-col-header">
            <span>${escapeHtml(g.provider)}</span>
            <span class="compare-latency">${g.latency_ms}ms</span>
          </div>
          ${body}
        </div>
      `;
    })
    .join("");
}

// ---------------------------------------------------------------- history

async function loadHistory() {
  const tbody = document.getElementById("historyTableBody");
  try {
    const res = await fetch(API.history);
    const data = await res.json();

    if (data.length === 0) {
      tbody.innerHTML = `<tr><td colspan="5" class="empty-row">No searches yet.</td></tr>`;
      return;
    }

    tbody.innerHTML = data
      .map(
        (h) => `
        <tr>
          <td>${escapeHtml(h.query)}</td>
          <td>${escapeHtml(h.mode)}</td>
          <td>${escapeHtml(h.provider)}</td>
          <td>${h.result_count}</td>
          <td>${formatDate(h.created_at)}</td>
        </tr>
      `
      )
      .join("");
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="5" class="empty-row">⚠ Failed to load history.</td></tr>`;
  }
}

// ---------------------------------------------------------------- settings

async function loadSettings() {
  const grid = document.getElementById("settingsGrid");
  try {
    const res = await fetch(API.settings);
    const data = await res.json();

    const entries = [
      ["Vector DB Provider", data.vector_db_provider],
      ["Embedding Model", data.openai_embedding_model],
      ["Chat Model", data.openai_chat_model],
      ["Chunk Size", data.chunk_size],
      ["Chunk Overlap", data.chunk_overlap],
      ["Default Top-K", data.top_k_results],
    ];

    grid.innerHTML = entries
      .map(
        ([key, value]) => `
        <div class="settings-card">
          <div class="settings-key">${escapeHtml(key)}</div>
          <div class="settings-value">${escapeHtml(String(value))}</div>
        </div>
      `
      )
      .join("");
  } catch (err) {
    grid.innerHTML = `<div class="empty-state">⚠ Failed to load settings.</div>`;
  }
}

// ---------------------------------------------------------------- utils

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str ?? "";
  return div.innerHTML;
}

function truncate(str, len) {
  return str.length > len ? str.slice(0, len) + "…" : str;
}

function formatDate(isoString) {
  try {
    const d = new Date(isoString);
    return d.toLocaleString(undefined, {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return isoString;
  }
}

// ---------------------------------------------------------------- init

checkHealth();
loadDocuments();
