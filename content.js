let songSections = [];
let currentSectionIndex = -1;
let mappedDOMElements = [];

function normalizeText(value) {
    return (value || "").replace(/\s+/g, " ").trim().toLowerCase();
}

function isVerseSection(section) {
    return normalizeText(section && section.type).includes("verse");
}

function isChorusSection(section) {
    return normalizeText(section && section.type).includes("chorus");
}

function escapeRegExp(value) {
    return (value || "").replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function findFlexibleTextMatch(rawText, queryText) {
    const queryTokens = normalizeText(queryText).split(" ").filter(Boolean);
    if (queryTokens.length === 0) {
        return null;
    }

    const pattern = queryTokens.map(escapeRegExp).join("\\s+");
    const regex = new RegExp(pattern, "i");
    const match = rawText.match(regex);
    if (!match) {
        return null;
    }

    return {
        index: match.index,
        length: match[0].length
    };
}

function findTextNodeMatch(anchorText, textNodes, startIndex = 0) {
    const normalizedAnchor = normalizeText(anchorText);

    for (let idx = startIndex; idx < textNodes.length; idx++) {
        const node = textNodes[idx];
        const rawText = node.nodeValue || "";
        const normalizedNode = normalizeText(rawText);

        if (normalizedNode.includes(normalizedAnchor)) {
            const exactMatch = findFlexibleTextMatch(rawText, anchorText);
            return {
                textNode: node,
                matchIndex: exactMatch ? exactMatch.index : rawText.toLowerCase().indexOf(normalizedAnchor),
                matchLength: exactMatch ? exactMatch.length : anchorText.length,
                exact: true
            };
        }
    }

    return null;
}

async function sendPageText() {
    try {
        console.log("Loaded. Reading song text...");
        
        let contentContainer = document.querySelector("pre, article, main, [class*='chord'], [class*='lyric'], [id*='song']");
        const pageText = contentContainer ? contentContainer.innerText : document.body.innerText;
        
        console.log("Sending text to the local parser...");
        const response = await fetch("http://127.0.0.1:5000/parse-song", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ pageText })
        });

        if (!response.ok) throw new Error(`Server error: ${response.status}`);

        const songStructure = await response.json();
        console.log("Song structure ready.");

        songSections = songStructure.sections || [];
        currentSectionIndex = -1; 

        mapUniversalAnchors();

    } catch (err) {
        console.error("Song parse failed:", err);
    }
}

function mapUniversalAnchors() {
    mappedDOMElements = [];
    if (songSections.length === 0) return;

    // Grab the page text once, then line it up with the parsed sections.

    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, null, false);
    let currentNode;
    const textNodes = [];

    while (currentNode = walker.nextNode()) {
        const parent = currentNode.parentElement;
        if (parent && !['SCRIPT', 'STYLE', 'NOSCRIPT', 'NAV', 'HEADER', 'FOOTER'].includes(parent.tagName)) {
            textNodes.push(currentNode);
        }
    }

    let searchStartIndex = 0;

    songSections.forEach((section) => {
        const anchorText = section.text_anchor.trim();
        if (!anchorText) {
            mappedDOMElements.push(null);
            return;
        }

        const matchedElement = findTextNodeMatch(anchorText, textNodes, searchStartIndex);
        if (matchedElement) {
            matchedElement.anchorText = anchorText;
            matchedElement.matchLength = matchedElement.matchLength || anchorText.length;
            searchStartIndex = textNodes.indexOf(matchedElement.textNode) + 1;
        }

        mappedDOMElements.push(matchedElement);
    });

    console.log(`Mapped ${mappedDOMElements.filter(el => el !== null).length}/${songSections.length} sections.`);
}


function scrollToSectionIndex(index) {
    const target = mappedDOMElements[index];
    if (!target || !target.textNode) {
        return false;
    }

    const textNode = target.textNode;
    const startIndex = Number.isInteger(target.matchIndex) ? target.matchIndex : -1;
    const matchLength = Number.isInteger(target.matchLength) && target.matchLength > 0 ? target.matchLength : (target.anchorText || "").length;

    if (startIndex < 0 || matchLength <= 0) {
        return false;
    }

    try {
        // Put the lyric near the middle of the screen.
        const range = document.createRange();
        range.setStart(textNode, startIndex);
        range.setEnd(textNode, startIndex + matchLength);

        const rect = range.getBoundingClientRect();
        const targetTop = window.scrollY + rect.top - (window.innerHeight * 0.5) + (rect.height * 0.5);

        window.scrollTo({
            top: Math.max(0, targetTop),
            behavior: "smooth"
        });

        const selection = window.getSelection();
        selection.removeAllRanges();
        selection.addRange(range);

        setTimeout(() => {
            selection.removeAllRanges();
        }, 1200);

        return true;

    } catch (err) {
        console.warn("Unable to scroll/highlight lyric:", err);
        return false;
    }
}

function findSectionIndex(sectionFilter, startIndex, step) {
    if (songSections.length === 0) return -1;

    for (let idx = startIndex; idx >= 0 && idx < songSections.length; idx += step) {
        if (sectionFilter(songSections[idx])) {
            return idx;
        }
    }

    return -1;
}

function jumpToSection(sectionFilter, searchStartIndex, step, wrapIndex, label) {
    // One helper keeps verse and chorus jumps behaving the same way.
    const nextIndex = findSectionIndex(sectionFilter, searchStartIndex, step);
    const wrappedIndex = wrapIndex();
    const targetIndex = nextIndex !== -1 ? nextIndex : wrappedIndex;

    if (targetIndex === -1) {
        return;
    }

    currentSectionIndex = targetIndex;
    console.log(`${label}: ${targetIndex}`);
    scrollToSectionIndex(targetIndex);
}

function navigateNextVerse() {
    jumpToSection(
        isVerseSection,
        currentSectionIndex + 1,
        1,
        () => songSections.findIndex(isVerseSection),
        "next verse"
    );
}

function navigatePreviousVerse() {
    jumpToSection(
        isVerseSection,
        currentSectionIndex - 1,
        -1,
        () => {
            const reversedIndex = [...songSections].reverse().findIndex(isVerseSection);
            return reversedIndex === -1 ? -1 : songSections.length - 1 - reversedIndex;
        },
        "previous verse"
    );
}

function navigateToChorus() {
    jumpToSection(
        isChorusSection,
        currentSectionIndex + 1,
        1,
        () => songSections.findIndex(isChorusSection),
        "chorus bypass"
    );
}

async function pollForGestures() {
    try {
        // Read the next queued gesture from Flask.
        const response = await fetch("http://127.0.0.1:5000/get-gesture");
        if (response.ok) {
            const data = await response.json();
            if (data.gesture === "DOWN") {
                navigateNextVerse();
            } else if (data.gesture === "UP") {
                navigatePreviousVerse();
            } else if (data.gesture === "LEFT" || data.gesture === "RIGHT") {
                navigateToChorus();
            }
        }
    } catch (err) {}
}

// Execute immediately when manually injected by toolbar icon click
sendPageText();
setInterval(pollForGestures, 250);