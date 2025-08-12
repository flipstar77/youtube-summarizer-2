import os
import pytest

try:
    from supabase import create_client
except Exception:
    create_client = None  # so we can skip gracefully


REQUIRED_KEYS = ("SUPABASE_URL", "SUPABASE_ANON_KEY")

pytestmark = pytest.mark.skipif(
    any(os.getenv(k) in (None, "") for k in REQUIRED_KEYS) or create_client is None,
    reason="Supabase creds or client not available",
)

@pytest.fixture(scope="session")
def sb():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_ANON_KEY")
    return create_client(url, key)


def test_search_by_query_embedding_rpc(sb):
    # Null vector just to check the RPC shape; not meaningful search
    q = [0.0] * 1536
    res = sb.rpc(
        "search_summaries_by_similarity",
        {"query_embedding": q, "match_threshold": 0.0, "match_count": 3},
    ).execute()

    assert isinstance(res.data, (list, type(None)))
    if res.data:
        row = res.data[0]
        # expected stable columns
        for k in ("id", "title", "summary", "url", "video_id", "summary_type", "created_at", "similarity"):
            assert k in row


def test_find_similar_summaries_rpc(sb):
    # pick a row that has an embedding; if none exist, skip
    probe = sb.table("summaries").select("id").is_("embedding", None, negate=True).limit(1).execute().data
    if not probe:
        pytest.skip("no rows with embedding → nothing to test")
        return

    summary_id = probe[0]["id"]
    res = sb.rpc("find_similar_summaries", {"summary_id": summary_id, "match_count": 3}).execute()

    assert isinstance(res.data, (list, type(None)))
    # If there are very few rows with embeddings, result may be empty; that's fine.
    if res.data:
        row = res.data[0]
        for k in ("id", "title", "summary", "url", "video_id", "summary_type", "created_at", "similarity"):
            assert k in row


if __name__ == "__main__":
    # Run tests directly if called as script
    import sys
    
    try:
        if any(os.getenv(k) in (None, "") for k in REQUIRED_KEYS):
            print("SKIP - Supabase credentials not available")
            sys.exit(0)
            
        if create_client is None:
            print("SKIP - Supabase client not available")
            sys.exit(0)
            
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_ANON_KEY")
        client = create_client(url, key)
        
        print("Running vector RPC smoke tests...")
        
        # Test 1
        q = [0.0] * 1536
        res = client.rpc("search_summaries_by_similarity", {
            "query_embedding": q, "match_threshold": 0.0, "match_count": 3
        }).execute()
        print("OK - search_summaries_by_similarity function works")
        
        # Test 2
        probe = client.table("summaries").select("id").is_("embedding", None, negate=True).limit(1).execute().data
        if probe:
            summary_id = probe[0]["id"]
            res = client.rpc("find_similar_summaries", {"summary_id": summary_id, "match_count": 3}).execute()
            print("OK - find_similar_summaries function works")
        else:
            print("INFO - No summaries with embeddings to test find_similar_summaries")
        
        print("SUCCESS - All vector RPC smoke tests passed!")
        
    except Exception as e:
        print(f"ERROR - Smoke test failed: {e}")
        sys.exit(1)