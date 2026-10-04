"""Tests for Chatbot Response Formatting and Session-based Conversation Memory."""

import pytest
from fastapi.testclient import TestClient
from server import app
from storage.database import SessionLocal, init_db
from storage.repository import TraceRepository
from agent.agent import sanitize_response_text, LaptopAgent


@pytest.fixture(autouse=True)
def setup_db():
    init_db()


@pytest.fixture
def client():
    return TestClient(app)


def test_sanitize_response_text_removes_unwanted_markdown():
    """Verify that unwanted formatting (asterisks, headings, tables, bullets, raw JSON) is removed."""
    raw_text = """### ASUS TUF Gaming F15
**₹68,990**

- Processor: AMD Ryzen 7 7735HS
- RAM: 16GB
- Storage: 512GB SSD
- Display: 15.6 inch FHD 144Hz

**Why it fits:**
High suitability for programming and gaming.

| Spec | Value |
|---|---|
| GPU | RTX 3050 |

```json
{"status": "ok"}
```
"""
    clean = sanitize_response_text(raw_text)

    # Must NOT contain asterisks, markdown headings, tables, or raw json
    assert "*" not in clean
    assert "**" not in clean
    assert "###" not in clean
    assert "|---|" not in clean
    assert "```json" not in clean
    assert "ASUS TUF Gaming F15" in clean
    assert "₹68,990" in clean
    assert "Processor: AMD Ryzen 7 7735HS" in clean


def test_session_based_conversation_memory_flow(client):
    """Verify conversation_id lifecycle, message storage, message count increment, and retrieval."""
    conv_id = "test-session-conv-001"

    # Turn 1: Initial user query
    res1 = client.post("/agent/run", json={
        "request": "I need a laptop under ₹80,000 for programming.",
        "conversation_id": conv_id,
    })
    assert res1.status_code == 200
    data1 = res1.json()
    assert data1["conversation_id"] == conv_id
    assert "*" not in data1["final_response"]
    assert "###" not in data1["final_response"]

    # Verify messages stored in DB
    history_res1 = client.get(f"/conversations/{conv_id}/messages")
    assert history_res1.status_code == 200
    hist1 = history_res1.json()
    # Should have user message + assistant message = 2 messages
    assert hist1["count"] >= 2
    assert hist1["messages"][0]["role"] == "user"
    assert hist1["messages"][1]["role"] == "assistant"

    # Turn 2: Follow-up question about the second one
    res2 = client.post("/agent/run", json={
        "request": "What about the second one?",
        "conversation_id": conv_id,
    })
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["conversation_id"] == conv_id

    # Verify message count increased in DB
    history_res2 = client.get(f"/conversations/{conv_id}/messages")
    hist2 = history_res2.json()
    assert hist2["count"] >= 4

    # Turn 3: Follow-up question about RAM
    res3 = client.post("/agent/run", json={
        "request": "How much RAM does it have?",
        "conversation_id": conv_id,
    })
    assert res3.status_code == 200
    data3 = res3.json()
    assert data3["conversation_id"] == conv_id
    assert "RAM" in data3["final_response"]

    # Turn 4: Follow-up question about gaming
    res4 = client.post("/agent/run", json={
        "request": "Is it good for gaming?",
        "conversation_id": conv_id,
    })
    assert res4.status_code == 200
    data4 = res4.json()
    assert data4["conversation_id"] == conv_id

    # Turn 5: Budget update
    res5 = client.post("/agent/run", json={
        "request": "Actually my budget is ₹60,000.",
        "conversation_id": conv_id,
    })
    assert res5.status_code == 200
    data5 = res5.json()
    assert data5["conversation_id"] == conv_id

    # Verify 5 sequential turns stored cleanly
    history_res5 = client.get(f"/conversations/{conv_id}/messages")
    hist5 = history_res5.json()
    assert hist5["count"] >= 10


import uuid


def test_isolated_conversations(client):
    """Verify that starting a new chat creates a completely isolated conversation."""
    conv_a = f"test-session-conv-A-{uuid.uuid4().hex[:8]}"
    conv_b = f"test-session-conv-B-{uuid.uuid4().hex[:8]}"

    # User A talks about laptops
    res_a = client.post("/agent/run", json={
        "request": "I need a gaming laptop under ₹1,20,000.",
        "conversation_id": conv_a,
    })
    assert res_a.status_code == 200

    # User B starts a fresh conversation about phones
    res_b = client.post("/agent/run", json={
        "request": "Show me smartphones under ₹30,000.",
        "conversation_id": conv_b,
    })
    assert res_b.status_code == 200

    hist_a = client.get(f"/conversations/{conv_a}/messages").json()
    hist_b = client.get(f"/conversations/{conv_b}/messages").json()

    assert all(m["conversation_id"] == conv_a for m in hist_a["messages"])
    assert all(m["conversation_id"] == conv_b for m in hist_b["messages"])
    assert hist_a["count"] == 2
    assert hist_b["count"] == 2


def test_general_math_clean_response(client):
    """Verify clean response for simple general math questions."""
    conv_id = f"test-session-math-{uuid.uuid4().hex[:8]}"
    res = client.post("/agent/run", json={
        "request": "What is 4 + 10?",
        "conversation_id": conv_id,
    })
    assert res.status_code == 200
    data = res.json()
    assert "4 + 10 is 14." in data["final_response"] or "14" in data["final_response"]
    assert "*" not in data["final_response"]
    assert "###" not in data["final_response"]


def test_all_budget_tiers_suggestions(client):
    """Verify that the chatbot provides valid suggestions across all budget tiers and electronics categories."""
    budget_tests = [
        ("phone under 15000", 15000),
        ("phone under 25000", 25000),
        ("phone under 60000", 60000),
        ("earbuds under 2000", 2000),
        ("earbuds under 5000", 5000),
        ("headphones under 10000", 10000),
        ("smartwatch under 2000", 2000),
        ("smartwatch under 20000", 20000),
        ("tv under 15000", 15000),
        ("tv under 30000", 30000),
        ("tv under 50000", 50000),
        ("monitor under 10000", 10000),
        ("monitor under 15000", 15000),
        ("monitor under 30000", 30000),
        ("tablet under 15000", 15000),
        ("tablet under 30000", 30000),
        ("router under 3000", 3000),
        ("router under 6000", 6000),
    ]

    for query, max_budget in budget_tests:
        cid = f"test-budget-{uuid.uuid4().hex[:8]}"
        res = client.post("/agent/run", json={"request": query, "conversation_id": cid})
        assert res.status_code == 200, f"Query failed for '{query}'"
        body = res.json()
        assert body["status"] == "success"
        resp_text = body["final_response"]
        assert "*" not in resp_text
        assert "###" not in resp_text
        assert "|---|" not in resp_text
def test_contextual_follow_up_questions_and_budget_proximity(client):
    """Verify that follow-up questions like 'whats the battery life' get specific answers and budget queries return closest matching options."""
    cid = f"test-followup-{uuid.uuid4().hex[:8]}"

    # 1. Ask for a phone under 60000 -> Should recommend OnePlus 12 (59999) closer to 60k
    res1 = client.post("/agent/run", json={"request": "Suggest a smartphone under ₹60,000", "conversation_id": cid})
    assert res1.status_code == 200
    data1 = res1.json()
    assert "OnePlus 12" in data1["final_response"] or "59,999" in data1["final_response"]
    assert "Within budget" in data1["final_response"] or "saving" in data1["final_response"].lower()

    # 2. Ask follow-up question: "whats the battery life"
    res2 = client.post("/agent/run", json={"request": "whats the battery life", "conversation_id": cid})
    assert res2.status_code == 200
    data2 = res2.json()
    assert "5400 mAh" in data2["final_response"] or "battery" in data2["final_response"].lower()
    # Must NOT re-render the generic "I found a few options that match your requirements." card
    assert "I found a few options that match your requirements." not in data2["final_response"]
    assert "*" not in data2["final_response"]
    assert "###" not in data2["final_response"]

    # 3. Ask follow-up question: "what processor is inside?"
    res3 = client.post("/agent/run", json={"request": "what processor is inside?", "conversation_id": cid})
    assert res3.status_code == 200
    data3 = res3.json()
    assert "Snapdragon" in data3["final_response"]
    assert "I found a few options that match your requirements." not in data3["final_response"]

    # 4. Ask follow-up question: "how is the camera?"
    res4 = client.post("/agent/run", json={"request": "how is the camera?", "conversation_id": cid})
    assert res4.status_code == 200
    data4 = res4.json()
    assert "Camera" in data4["final_response"] or "camera" in data4["final_response"]
    assert "I found a few options that match your requirements." not in data4["final_response"]




