const claimInput = document.getElementById("claim-input");
const checkBtn = document.getElementById("check-btn");
const clearBtn = document.getElementById("clear-btn");
const resultPanel = document.getElementById("result-panel");
const resultBody = document.getElementById("result-body");
const statusText = document.getElementById("status-text");

const setStatus = (text) => {
  statusText.textContent = text;
};

const clearResults = () => {
  resultBody.innerHTML = "";
  resultPanel.classList.add("hidden");
};

const createLine = (label, value) => {
  const row = document.createElement("div");
  row.className = "info-line";
  const strong = document.createElement("strong");
  strong.textContent = `${label}: `;
  const text = document.createElement("span");
  text.textContent = value;
  row.appendChild(strong);
  row.appendChild(text);
  return row;
};

const createLinkLine = (label, href) => {
  const row = document.createElement("div");
  row.className = "info-line";
  const strong = document.createElement("strong");
  strong.textContent = `${label}: `;
  const link = document.createElement("a");
  link.className = "link";
  link.href = href;
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  link.textContent = href;
  row.appendChild(strong);
  row.appendChild(link);
  return row;
};

const createBadge = (label) => {
  const badge = document.createElement("span");
  badge.className = "badge";
  const upper = (label || "").toUpperCase();
  badge.textContent = upper || "UNKNOWN";
  if (upper === "SUPPORTED") {
    badge.classList.add("supported");
  } else if (upper === "REFUTED") {
    badge.classList.add("refuted");
  } else {
    badge.classList.add("unknown");
  }
  return badge;
};

const renderVerificationResult = (container, item, index, claimMap) => {
  const card = document.createElement("div");
  card.className = "card";

  const header = document.createElement("div");
  header.className = "card-header";

  const title = document.createElement("div");
  title.className = "card-title";
  title.textContent = `Claim #${index + 1}`;

  header.appendChild(title);
  card.appendChild(header);

  const metaRow = document.createElement("div");
  metaRow.className = "card-header";
  const badge = createBadge(item?.label ?? "");
  const confidence = document.createElement("div");
  confidence.className = "confidence";
  confidence.textContent = String(item?.confidence ?? "");
  metaRow.appendChild(badge);
  metaRow.appendChild(confidence);
  card.appendChild(metaRow);

  const claimText =
    item?.claim || claimMap?.[String(item?.claim_id)] || item?.claim_text || "";
  card.appendChild(createLine("Claim", claimText));

  const reasoningTitle = document.createElement("div");
  reasoningTitle.className = "section-title";
  reasoningTitle.textContent = "Reasoning:";
  card.appendChild(reasoningTitle);

  const reasoningBox = document.createElement("div");
  reasoningBox.className = "info-box";
  reasoningBox.textContent = item?.reasoning ?? "";
  card.appendChild(reasoningBox);

  card.appendChild(
    createLine("Num evidences", String(item?.num_evidences_used ?? ""))
  );

  if (Array.isArray(item?.evidence_used) && item.evidence_used.length > 0) {
    const evidenceUsedTitle = document.createElement("div");
    evidenceUsedTitle.className = "section-title";
    evidenceUsedTitle.textContent = "Evidence used";
    card.appendChild(evidenceUsedTitle);

    item.evidence_used.forEach((evidence) => {
      const block = document.createElement("div");
      block.className = "sub-card";
      block.appendChild(createLine("Evidence ID", evidence?.evidence_id ?? ""));
      block.appendChild(createLine("Snippet", evidence?.snippet ?? ""));
      block.appendChild(createLine("Relevance", evidence?.relevance ?? ""));
      card.appendChild(block);
    });
  }

  if (Array.isArray(item?.evidences) && item.evidences.length > 0) {
    const evidenceTitle = document.createElement("div");
    evidenceTitle.className = "section-title";
    evidenceTitle.textContent = "Evidences:";
    card.appendChild(evidenceTitle);

    item.evidences.forEach((evidence) => {
      const block = document.createElement("div");
      block.className = "sub-card alt";
      if (evidence?.site) {
        block.appendChild(createLinkLine("Site", evidence.site));
      } else {
        block.appendChild(createLine("Site", ""));
      }
      block.appendChild(createLine("Reason", evidence?.reason ?? ""));
      card.appendChild(block);
    });
  }

  container.appendChild(card);
};

const renderResults = (data) => {
  resultBody.innerHTML = "";

  if (Array.isArray(data)) {
    data.forEach((item, index) => {
      renderVerificationResult(resultBody, item, index, null);
    });
    return;
  }

  if (data && typeof data === "object") {
    const claimMap = {};
    if (Array.isArray(data.claims)) {
      data.claims.forEach((claim) => {
        if (claim?.claim_id != null) {
          claimMap[String(claim.claim_id)] = claim.claim_text || "";
        }
      });
    }

    if (Array.isArray(data.verification_results)) {
      data.verification_results.forEach((item, index) => {
        renderVerificationResult(resultBody, item, index, claimMap);
      });
      return;
    }
  }

  resultBody.appendChild(createLine("Error", "Unexpected response format."));
};

checkBtn.addEventListener("click", async () => {
  const text = claimInput.value.trim();
  if (!text) {
    alert("Vui lòng nhập văn bản để kiểm tra.");
    return;
  }

  setStatus("Đang kiểm tra...");
  resultPanel.classList.remove("hidden");
  resultBody.innerHTML = "";

  try {
    const response = await fetch("https://factcheck.thanhpx.work/fact-check", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    const data = await response.json();
    renderResults(data);
    setStatus("Hoàn tất.");
  } catch (error) {
    alert(error?.message || "Không thể gọi API.");
    resultBody.innerHTML = "";
    resultBody.appendChild(createLine("Error", error?.message || "API error."));
    setStatus("Lỗi.");
  }
});

clearBtn.addEventListener("click", () => {
  claimInput.value = "";
  setStatus("");
  clearResults();
});
