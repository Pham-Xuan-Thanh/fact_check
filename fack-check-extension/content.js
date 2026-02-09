const PANEL_ID = "fact-check-extension-panel";

function removeExistingPanel() {
  const existing = document.getElementById(PANEL_ID);
  if (existing) {
    existing.remove();
  }
}

function createPanel() {
  removeExistingPanel();

  const panel = document.createElement("div");
  panel.id = PANEL_ID;
  panel.style.position = "fixed";
  panel.style.top = "16px";
  panel.style.right = "16px";
  panel.style.width = "460px";
  panel.style.maxHeight = "80vh";
  panel.style.overflow = "auto";
  panel.style.zIndex = "2147483647";
  panel.style.background = "#ffffff";
  panel.style.border = "1px solid #ededed";
  panel.style.boxShadow = "0 12px 30px rgba(0,0,0,0.12)";
  panel.style.borderRadius = "16px";
  panel.style.fontFamily = "Inter, Arial, sans-serif";
  panel.style.fontSize = "13.5px";
  panel.style.color = "#111";

  const header = document.createElement("div");
  header.style.display = "flex";
  header.style.alignItems = "center";
  header.style.justifyContent = "space-between";
  header.style.padding = "14px 16px";
  header.style.borderBottom = "1px solid #f0f0f0";
  header.style.background = "linear-gradient(135deg, #F7F9FF 0%, #FFFFFF 65%)";
  header.style.borderTopLeftRadius = "16px";
  header.style.borderTopRightRadius = "16px";

  const title = document.createElement("div");
  title.textContent = "Fact Check Result";
  title.style.fontWeight = "700";
  title.style.fontSize = "18px";
  title.style.letterSpacing = "0.2px";
  title.style.color = "#0F172A";

  const closeBtn = document.createElement("button");
  closeBtn.textContent = "Close";
  closeBtn.style.border = "1px solid #E2E8F0";
  closeBtn.style.background = "#FFFFFF";
  closeBtn.style.padding = "6px 14px";
  closeBtn.style.borderRadius = "999px";
  closeBtn.style.fontWeight = "600";
  closeBtn.style.color = "#0F172A";
  closeBtn.style.boxShadow = "0 1px 2px rgba(15, 23, 42, 0.08)";
  closeBtn.style.cursor = "pointer";
  closeBtn.addEventListener("mouseenter", () => {
    closeBtn.style.background = "#F8FAFC";
  });
  closeBtn.addEventListener("mouseleave", () => {
    closeBtn.style.background = "#FFFFFF";
  });
  closeBtn.addEventListener("click", () => panel.remove());

  header.appendChild(title);
  header.appendChild(closeBtn);
  panel.appendChild(header);

  const body = document.createElement("div");
  body.style.padding = "12px 16px 16px";
  body.id = `${PANEL_ID}-body`;
  panel.appendChild(body);

  document.body.appendChild(panel);
  return body;
}

function addLine(container, label, value) {
  const row = document.createElement("div");
  row.style.marginBottom = "6px";

  const strong = document.createElement("span");
  strong.textContent = `${label}: `;
  strong.style.fontWeight = "bold";

  const text = document.createElement("span");
  text.textContent = value;

  row.appendChild(strong);
  row.appendChild(text);
  container.appendChild(row);
}

function addDivider(container) {
  const hr = document.createElement("div");
  hr.style.height = "1px";
  hr.style.background = "#f0f0f0";
  hr.style.margin = "14px 0";
  container.appendChild(hr);
}

function addListItem(container, text) {
  const item = document.createElement("div");
  item.textContent = `- ${text}`;
  item.style.marginLeft = "6px";
  container.appendChild(item);
}

function addLinkLine(container, label, href) {
  const row = document.createElement("div");
  row.style.marginBottom = "6px";

  const strong = document.createElement("span");
  strong.textContent = `${label}: `;
  strong.style.fontWeight = "bold";

  const link = document.createElement("a");
  link.href = href;
  link.textContent = href;
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  link.style.color = "#1a73e8";
  link.style.textDecoration = "none";

  row.appendChild(strong);
  row.appendChild(link);
  container.appendChild(row);
}

function applyCardStyle(card) {
  card.style.padding = "14px";
  card.style.marginBottom = "12px";
  card.style.background = "#ffffff";
  card.style.border = "1px solid #ededed";
  card.style.borderRadius = "16px";
  card.style.boxShadow = "0 2px 8px rgba(0,0,0,0.05)";
}

function createBadge(text) {
  const badge = document.createElement("span");
  const label = (text || "").toUpperCase();
  badge.textContent = label || "UNKNOWN";
  badge.style.padding = "4px 10px";
  badge.style.borderRadius = "999px";
  badge.style.fontWeight = "700";
  badge.style.fontSize = "12px";

  if (label === "SUPPORTED") {
    badge.style.background = "#E7F6EC";
    badge.style.color = "#1F8A3B";
  } else if (label === "REFUTED") {
    badge.style.background = "#FDECEC";
    badge.style.color = "#C62828";
  } else {
    badge.style.background = "#F1F3F4";
    badge.style.color = "#444";
  }

  return badge;
}

function createInfoBox() {
  const box = document.createElement("div");
  box.style.background = "#F3F6FB";
  box.style.border = "1px solid #E7ECF3";
  box.style.borderRadius = "12px";
  box.style.padding = "10px 12px";
  box.style.marginTop = "6px";
  return box;
}

function renderVerificationResult(body, item, index, claimMap) {
  const card = document.createElement("div");
  applyCardStyle(card);

  const claimText =
    item?.claim || claimMap?.[String(item?.claim_id)] || item?.claim_text || "";

  const headerRow = document.createElement("div");
  headerRow.style.display = "flex";
  headerRow.style.alignItems = "center";
  headerRow.style.justifyContent = "space-between";
  headerRow.style.marginBottom = "10px";

  const sectionTitle = document.createElement("div");
  sectionTitle.textContent = `Claim #${index + 1}`;
  sectionTitle.style.fontWeight = "700";
  sectionTitle.style.fontSize = "16px";

  headerRow.appendChild(sectionTitle);
  card.appendChild(headerRow);

  const metaRow = document.createElement("div");
  metaRow.style.display = "flex";
  metaRow.style.alignItems = "center";
  metaRow.style.justifyContent = "space-between";
  metaRow.style.marginBottom = "10px";

  const badge = createBadge(item?.label ?? "");
  const confidence = document.createElement("div");
  confidence.textContent = String(item?.confidence ?? "");
  confidence.style.fontWeight = "700";
  confidence.style.fontSize = "16px";

  metaRow.appendChild(badge);
  metaRow.appendChild(confidence);
  card.appendChild(metaRow);

  addLine(card, "Claim", claimText);

  const reasoningLabel = document.createElement("div");
  reasoningLabel.textContent = "Reasoning:";
  reasoningLabel.style.fontWeight = "700";
  reasoningLabel.style.marginTop = "8px";
  card.appendChild(reasoningLabel);

  const reasoningBox = createInfoBox();
  const reasoningText = document.createElement("div");
  reasoningText.textContent = item?.reasoning ?? "";
  reasoningBox.appendChild(reasoningText);
  card.appendChild(reasoningBox);

  addLine(card, "Num evidences", String(item?.num_evidences_used ?? ""));

  if (Array.isArray(item?.evidence_used) && item.evidence_used.length > 0) {
    const evidenceUsedTitle = document.createElement("div");
    evidenceUsedTitle.textContent = "Evidence used";
    evidenceUsedTitle.style.fontWeight = "700";
    evidenceUsedTitle.style.marginTop = "10px";
    evidenceUsedTitle.style.marginBottom = "6px";
    card.appendChild(evidenceUsedTitle);

    item.evidence_used.forEach((evidence) => {
      const block = document.createElement("div");
      block.style.padding = "10px 12px";
      block.style.marginBottom = "8px";
      block.style.background = "#F8FAFF";
      block.style.border = "1px solid #E6ECFF";
      block.style.borderRadius = "12px";

      addLine(block, "Evidence ID", evidence?.evidence_id ?? "");
      addLine(block, "Snippet", evidence?.snippet ?? "");
      addLine(block, "Relevance", evidence?.relevance ?? "");
      card.appendChild(block);
    });
  }

  if (Array.isArray(item?.evidences) && item.evidences.length > 0) {
    const evidenceTitle = document.createElement("div");
    evidenceTitle.textContent = "Evidences:";
    evidenceTitle.style.fontWeight = "700";
    evidenceTitle.style.marginTop = "10px";
    evidenceTitle.style.marginBottom = "6px";
    card.appendChild(evidenceTitle);

    item.evidences.forEach((evidence) => {
      const block = document.createElement("div");
      block.style.padding = "10px 12px";
      block.style.marginBottom = "8px";
      block.style.background = "#ffffff";
      block.style.border = "1px solid #ededed";
      block.style.borderRadius = "12px";

      if (evidence?.site) {
        addLinkLine(block, "Site", evidence.site);
      } else {
        addLine(block, "Site", "");
      }
      addLine(block, "Reason", evidence?.reason ?? "");
      card.appendChild(block);
    });
  }

  body.appendChild(card);
}

function renderResult(data, selectedText) {
  const body = createPanel();

  if (selectedText) {
    const selectedTitle = document.createElement("div");
    selectedTitle.textContent = "Selected";
    selectedTitle.style.fontWeight = "700";
    selectedTitle.style.marginBottom = "6px";
    body.appendChild(selectedTitle);

    const selectedBox = createInfoBox();
    const selectedValue = document.createElement("div");
    selectedValue.textContent = selectedText;
    selectedBox.appendChild(selectedValue);
    body.appendChild(selectedBox);

    addDivider(body);
  }

  if (Array.isArray(data)) {
    data.forEach((item, index) => {
      renderVerificationResult(body, item, index, null);
      if (index < data.length - 1) {
        addDivider(body);
      }
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
        renderVerificationResult(body, item, index, claimMap);
        if (index < data.verification_results.length - 1) {
          addDivider(body);
        }
      });
      return;
    }
  }

  addLine(body, "Error", "Unexpected response format.");
}

function renderMessage(message, title) {
  const body = createPanel();
  const header = document.createElement("div");
  header.textContent = title;
  header.style.fontWeight = "bold";
  header.style.marginBottom = "6px";
  body.appendChild(header);
  addLine(body, "Message", message);
}

chrome.runtime.onMessage.addListener((message) => {
  if (message?.type === "fact-check-result") {
    renderResult(message.payload, message.selectedText);
  }

  if (message?.type === "fact-check-error") {
    alert(message.message || "Fact check failed.");
    renderMessage(message.message, "Fact Check Error");
  }

  if (message?.type === "fact-check-info") {
    alert(message.message || "Fact check info.");
    renderMessage(message.message, "Fact Check");
  }
});
