from flask import Flask, request, jsonify
import easyocr

app = Flask(__name__)
reader = easyocr.Reader(['en', 'th'])

@app.route('/api/easy-ocr', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "Require field image in request"}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({"error": "File not found"}), 400

    try:
        results = reader.readtext(image_file.read(), detail=0)
        lines = [{"line": idx + 1, "text": line}
                 for idx, line in enumerate(results)]

        return jsonify({
            "lines": lines
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
