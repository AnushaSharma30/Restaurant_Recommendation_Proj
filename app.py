# app.py (The Correct Version for Render Deployment)

from flask import Flask, jsonify, request
from engine import RecommendationEngine # Import our refactored engine
# app.py (add two new lines)


from flask_cors import CORS # <-- IMPORT CORS

 # <-- ENABLE CORS FOR YOUR APP

# ... (the rest of your app.py remains the same) ...

app = Flask(__name__)
CORS(app)
# --- Load the engine once when the server starts ---
# This is the corrected line. We call it with NO arguments.
# It will automatically find the DATABASE_URL from the Render environment.
print("Loading Recommendation Engine...")
engine = RecommendationEngine()

@app.route("/")
def index():
    return "<h1>Recommendation Engine API is running!</h1><p>Usage: /recommend/&lt;user_id&gt;</p>"

@app.route("/recommend/<int:user_id>", methods=['GET'])
def recommend(user_id):
    if engine.db_engine is None:
        return jsonify({"error": "Server is not ready, database connection failed."}), 500
    
    n_recs = request.args.get('n', default=5, type=int)
    
    recommendations = engine.get_recommendations(user_id, n_recommendations=n_recs)
    
    if recommendations is None:
        return jsonify({"error": f"User with user_id {user_id} not found."}), 404
    
    if recommendations.empty:
        return jsonify({
            "message": "No new recommendations found for this user in their local area.",
            "recommendations": []
        })
    
    return jsonify(recommendations.to_dict(orient='records'))

if __name__ == '__main__':
    # This part is only for local testing, Gunicorn runs the app in production.
    app.run(host='0.0.0.0', port=5000)