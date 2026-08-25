# HandsFreeChords

HandsFreeChords is a browser extension that lets musicians control online chord sheets and lyrics using head movements, so they don't need to stop playing to scroll.

## How it works

The project has three main parts:

* **Browser extension** – extracts the song text, maps the AI-generated sections to the webpage, and handles scrolling.
* **Flask backend** – processes requests from the extension and handles gesture communication.
* **Computer vision + AI** – MediaPipe/OpenCV detects head movements, while Gemini identifies song sections such as verses and choruses.

```text
Song webpage
     ↓
Browser extension
     ↓
Flask backend
     ↓
AI song parsing + gesture detection
     ↓
Browser navigation
```

## Controls

| Gesture      | Action         |
| ------------ | -------------- |
| UP           | Previous verse |
| DOWN         | Next verse     |
| LEFT / RIGHT | Jump to chorus |

UP/DOWN navigation is based on the user's current position on the page, so it can find the next or previous verse even when the user is currently viewing a chorus, bridge, or other section.

## Tech Stack

* **JavaScript** – Chrome extension and DOM manipulation
* **Python / Flask** – backend
* **OpenCV + MediaPipe** – head movement detection
* **Google Gemini** – song structure parsing

## Running

Start the Flask server:

```bash
python app.py
```

The extension expects the backend at:

```text
http://127.0.0.1:5000
```

Then load the extension through Chrome/Edge's **Developer Mode → Load unpacked**.

## Current Status

The core system is working, including:

* AI song section parsing
* DOM text mapping
* Head movement detection
* Hands-free verse navigation
* Chorus navigation
* Lyric highlighting

The project is still being developed, with improvements planned for gesture reliability, webpage compatibility, and more accurate section detection.
