"""
Vector Search Routes
Clean API endpoints for vector/semantic search operations
"""
from flask import Blueprint, request, jsonify
from services.vector_search import vector_search_service
from services.summary import summary_service

# Create blueprint
vector_bp = Blueprint('vector_search', __name__)

@vector_bp.route('/similar/<int:summary_id>')
def similar_api(summary_id):
    """
    Simple similar summaries API - direct service call
    GET /similar/24?count=8
    """
    count = int(request.args.get("count", 5))
    
    # Validate count parameter
    if count < 1 or count > 50:
        return jsonify({"error": "count must be between 1 and 50"}), 400
    
    items = vector_search_service.find_similar(summary_id, count)
    
    return jsonify({
        "summary_id": summary_id,
        "count": count,
        "results": items
    })

@vector_bp.route('/semantic-search', methods=['POST'])
def semantic_search_rpc():
    """
    Semantic search using direct embedding input
    POST /semantic-search
    {"embedding": [...], "threshold": 0.75, "count": 10}
    """
    body = request.get_json(force=True) or {}
    embedding = body.get("embedding")
    threshold = float(body.get("threshold", 0.75))
    count = int(body.get("count", 10))

    # Validate embedding
    if not embedding or len(embedding) != 1536:
        return jsonify({"error": "embedding (1536 floats) required"}), 400
    
    # Validate parameters
    if threshold < 0.0 or threshold > 1.0:
        return jsonify({"error": "threshold must be between 0.0 and 1.0"}), 400
    
    if count < 1 or count > 50:
        return jsonify({"error": "count must be between 1 and 50"}), 400

    items = vector_search_service.search_by_embedding(embedding, threshold, count)
    return jsonify({"results": items})

@vector_bp.route('/semantic_search', methods=['POST'])
def semantic_search():
    """
    Text-based semantic search - generates embeddings automatically
    POST /semantic_search
    {"query": "machine learning", "threshold": 0.75, "limit": 10}
    """
    try:
        if not vector_search_service.is_available():
            return jsonify({'status': 'error', 'message': 'Vector search not available'}), 400
        
        data = request.get_json()
        query = data.get('query', '').strip()
        threshold = float(data.get('threshold', 0.75))
        limit = int(data.get('limit', 10))
        
        if not query:
            return jsonify({'status': 'error', 'message': 'Query cannot be empty'}), 400
        
        # Use service layer for text search
        similar_summaries = vector_search_service.search_by_text(query, threshold, limit)
        
        return jsonify({
            'status': 'success',
            'query': query,
            'threshold': threshold,
            'results': similar_summaries,
            'count': len(similar_summaries)
        })
        
    except Exception as e:
        print(f"[ERROR] Semantic search failed: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@vector_bp.route('/vector-search/health')
def vector_search_health():
    """Health check for vector search functionality"""
    try:
        # Use DAL health check
        from db.queries import dal
        health_result = dal.health_check()
        
        # Add service-level checks
        health_result['vector_search_available'] = vector_search_service.is_available()
        
        return jsonify(health_result)
        
    except Exception as e:
        return jsonify({
            "status": "error", 
            "message": f"Health check failed: {e}"
        }), 500