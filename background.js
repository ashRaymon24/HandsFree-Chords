chrome.action.onClicked.addListener((tab) => {
    // Inject the content script into the active tab.
    // This keeps the extension quiet until the user turns it on.
    chrome.scripting.executeScript({
      target: { tabId: tab.id },
      files: ['content.js']
    });
    console.log(`Injected content script into tab ${tab.id}`);
});