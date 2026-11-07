"""
SEGMENTIQ: FastAPI Application Service Entry Point.

Serves processed customer behavioral metrics, segmentation summaries,
cluster profiles, and e-commerce marketing recommendations to the dashboard.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router, get_processed_data

app = FastAPI(
    title="SEGMENTIQ API",
    version="1.0.0",
    description="Customer Segmentation & Retention Engine Backend API",
)

# Enable CORS for all origins, methods, and headers to support local frontend dev servers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register endpoints under /api/v1
app.include_router(router)


@app.on_event("startup")
def startup_event():
    """Preload processed customer dataset into memory on service startup."""
    df = get_processed_data()
    if df.empty:
        print("[WARNING] Processed dataset not found or empty. Please execute `python run.py` first.")
    else:
        print(f"[INFO] SEGMENTIQ API preloaded {len(df):,} customer records successfully.")


@app.get("/", tags=["Root"])
def root():
    """Service metadata and documentation links."""
    return {
        "service": "SEGMENTIQ Customer Analytics API",
        "version": "1.0.0",
        "status": "online",
        "documentation": "/docs",
        "openapi": "/openapi.json",
        "api_v1": {
            "health": "/api/v1/health",
            "summary": "/api/v1/summary",
            "customers": "/api/v1/customers",
            "customer_detail": "/api/v1/customers/{customer_id}",
        },
    }
