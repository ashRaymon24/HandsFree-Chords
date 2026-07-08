from flask import Flask, request, jsonify
from flask_cors import CORS
from parser import parseSongStructure 

app = Flask(__name__)
CORS(app)  # Let the extension call this API.

# Latest queued gesture.
current_gesture = "NONE"

@app.route("/parse-song", methods=["POST"])
def parse_song():
    data = request.get_json()
    if not data or "pageText" not in data:
        return jsonify({"error": "pageText missing"}), 400
    
    try:
        # Parse the lyric text and hand the structured result back to the extension.
        song_structure = parseSongStructure(data["pageText"])
        if hasattr(song_structure, "model_dump"):
            return jsonify(song_structure.model_dump()), 200
        return jsonify(song_structure), 200
    except Exception as e:
        print(f"Parse error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/trigger-gesture", methods=["POST"])
def trigger_gesture():
    global current_gesture
    data = request.get_json()
    if data and "gesture" in data:
        # Keep only the latest gesture so the browser sees one clean action.
        current_gesture = data["gesture"]
        print(f"Gesture queued: {current_gesture}")
        return jsonify({"status": "buffered"}), 200
    return jsonify({"error": "Invalid payload"}), 400

@app.route("/get-gesture", methods=["GET"])
def get_gesture():
    global current_gesture
    # Return it once, then clear it.
    gesture_to_return = current_gesture
    current_gesture = "NONE" 
    return jsonify({"gesture": gesture_to_return}), 200

if __name__ == "__main__":
    app.run(port=5000, debug=True)