import httpx

base_url = "http://localhost:8000"

with httpx.Client(base_url=base_url, timeout=30.0) as client:
    print("Running Round 1 - Data Layer...")
    r = client.get("/")
    print("1. GET /:", r.json())

    with open("lead101_realtime_5000.csv", "rb") as f:
        files = {"file": ("lead101_realtime_5000.csv", f, "text/csv")}
        r = client.post("/api/leads/upload", files=files)
        print("2. POST /api/leads/upload:", r.json())

    r = client.get("/api/leads", params={"page": 1, "page_size": 10})
    data = r.json()
    print("3. GET /api/leads: total=", data["total"], "len(data)=", len(data["data"]))
    lead_id = data["data"][0]["lead_id"]

    r = client.get("/api/leads", params={"source": "referral", "converted": 1})
    data = r.json()
    print("4. GET /api/leads (filtered): total=", data["total"])

    r = client.get("/api/leads/stats/summary")
    print("5. GET /api/leads/stats/summary:", r.json())

    r = client.get(f"/api/leads/{lead_id}")
    print(f"6. GET /api/leads/ID: lead_id=", r.json()["lead_id"])

    print("\nRunning Round 2 - Analytics Layer...")
    r = client.get("/api/analytics/source")
    print("7. GET /api/analytics/source:", [d["source"] for d in r.json()][:3], "...")

    r = client.get("/api/analytics/city")
    print("8. GET /api/analytics/city:", [d["city"] for d in r.json()][:3], "...")

    r = client.get("/api/analytics/course")
    print("9. GET /api/analytics/course:", [d["course"] for d in r.json()][:3], "...")

    r = client.get("/api/analytics/funnel")
    print("10. GET /api/analytics/funnel:", [d["stage"] for d in r.json()])

    r = client.get("/api/analytics/counselors")
    print("11. GET /api/analytics/counselors: count=", len(r.json()))

    r = client.get("/api/analytics/monthly")
    print("12. GET /api/analytics/monthly: count=", len(r.json()))

    r = client.get("/api/analytics/stuck-leads")
    print("13. GET /api/analytics/stuck-leads: count=", r.json()["count"])

    r = client.get("/api/analytics/overview")
    print("14. GET /api/analytics/overview: keys=", list(r.json().keys()))

    print("\nRunning Round 3 - Prediction Layer...")
    predict_payload_high = {
      "source": "referral",
      "course_interest": "B.Tech CS",
      "city": "Chennai",
      "total_calls": 8,
      "total_whatsapp_messages": 15,
      "total_emails": 4,
      "email_opened": 1,
      "whatsapp_replied": 1,
      "form_completion_percentage": 85,
      "response_time_hours": 1.5,
      "days_since_last_interaction": 2,
      "current_stage": "application",
      "days_in_inquiry_stage": 3,
      "days_in_engagement_stage": 4,
      "days_in_application_stage": 2,
      "days_in_verification_stage": 0,
      "counselor_id": "C001"
    }
    r = client.post("/api/predict", json=predict_payload_high)
    print("15. POST /api/predict (high engagement):", r.json())

    predict_payload_low = {
      "source": "website",
      "course_interest": "B.Com",
      "city": "Mumbai",
      "total_calls": 0,
      "total_whatsapp_messages": 0,
      "total_emails": 0,
      "email_opened": 0,
      "whatsapp_replied": 0,
      "form_completion_percentage": 10,
      "response_time_hours": None,
      "days_since_last_interaction": 25,
      "current_stage": "inquiry",
      "days_in_inquiry_stage": 16,
      "days_in_engagement_stage": 0,
      "days_in_application_stage": 0,
      "days_in_verification_stage": 0,
      "counselor_id": "C005"
    }
    r = client.post("/api/predict", json=predict_payload_low)
    print("16. POST /api/predict (low engagement):", r.json())

    r = client.post("/api/predict/batch")
    print("17. POST /api/predict/batch:", r.json())

    r = client.get(f"/api/predict/{lead_id}")
    print(f"18. GET /api/predict/{lead_id}:", r.json())

    print("\nRunning Round 4 - Edge Cases...")
    r = client.get("/api/leads/LEAD99999")
    print("19. GET /api/leads/LEAD99999 status:", r.status_code)

    r = client.get("/api/predict/LEAD99999")
    print("20. GET /api/predict/LEAD99999 status:", r.status_code)

    with open("lead101_realtime_5000.csv", "rb") as f:
        files = {"file": ("lead101_realtime_5000.csv", f, "text/csv")}
        r = client.post("/api/leads/upload", files=files)
        print("21. POST /api/leads/upload (re-upload):", r.json())
