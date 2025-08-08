# app.py
from flask import Flask, jsonify, request
from engine import RecommendationEngine # Import our refactored engine

app = Flask(__name__)

# --- Load the engine once when the server starts ---
# This is a critical optimization. The model is loaded into memory only once.
print("Loading Recommendation Engine... This may take a moment.")
engine = RecommendationEngine(server_name="localhost\\SQLEXPRESS", db_name="RestaurantReco")

@app.route("/")
def index():
    return "<h1>Recommendation Engine API is running!</h1><p>Usage: /recommend/&lt;user_id&gt;</p>"

@app.route("/recommend/<int:user_id>", methods=['GET'])
def recommend(user_id):
    if engine.db_engine is None:
        return jsonify({"error": "Server is not ready, database connection failed."}), 500

    # Get optional parameters from the URL (e.g., /recommend/666?n=3)
    n_recs = request.args.get('n', default=5, type=int)

    # Call our engine's main function
    recommendations = engine.get_recommendations(user_id, n_recommendations=n_recs)

    if recommendations is None:
        return jsonify({"error": f"User with user_id {user_id} not found."}), 404

    if recommendations.empty:
        return jsonify({
            "message": "No new recommendations found for this user in their local area.",
            "recommendations": []
        })

    # Convert the final DataFrame to a JSON response
    return jsonify(recommendations.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False) # Set debug=False for production