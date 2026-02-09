const MENU_ID = "fact-check-selection";

const sendMessageToTab = (tabId, message) =>
  new Promise((resolve) => {
    chrome.tabs.sendMessage(tabId, message, () => {
      if (chrome.runtime.lastError) {
        resolve(false);
        return;
      }
      resolve(true);
    });
  });

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: MENU_ID,
    title: "Fact check selection",
    contexts: ["selection"],
  });
});

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId !== MENU_ID || !tab?.id) {
    return;
  }

  try {
    const [result] = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: () => window.getSelection().toString(),
    });
    const selectedText = result?.result?.trim() || "";

    if (!selectedText) {
      await sendMessageToTab(tab.id, {
        type: "fact-check-info",
        message: "No text selected.",
      });
      return;
    }

    const response = await fetch("http://localhost:5000/fact-check", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: selectedText }),
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    const data = await response.json();
    await sendMessageToTab(tab.id, {
      type: "fact-check-result",
      payload: data,
      selectedText,
    });
  } catch (error) {
    await sendMessageToTab(tab.id, {
      type: "fact-check-error",
      message: error?.message || "Unknown error.",
    });
  }
});
