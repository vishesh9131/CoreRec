"""
Model Server for Production Serving

FastAPI-based REST API server for serving recommendation models.

Author: Vishesh Yadav (mail: sciencely98@gmail.com)
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import logging

# Optional FastAPI import
try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse
    import uvicorn

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


class PredictionRequest(BaseModel):
    """Request schema for predictions."""

    user_id: Any
    item_id: Any
    context: Optional[Dict[str, Any]] = None


class RecommendationRequest(BaseModel):
    """Request schema for recommendations."""

    user_id: Any
    top_k: int = 10
    exclude_items: List[Any] = []
    context: Optional[Dict[str, Any]] = None


class BatchPredictionRequest(BaseModel):
    """Request schema for batch predictions."""

    pairs: List[tuple]  # List of (user_id, item_id) tuples


class BatchRecommendationRequest(BaseModel):
    """Request schema for batch recommendations."""

    user_ids: List[Any]
    top_k: int = 10


class ModelServer:
    """
    Production-ready model serving infrastructure.

    Provides REST API endpoints for:
    - Single predictions
    - Batch predictions
    - Recommendations
    - Batch recommendations
    - Health checks
    - Model metadata

    Example:
        from corerec.serving import ModelServer
        from corerec.engines.unionizedFilterEngine.nn_base.ncf import NCF

        model = NCF.load('models/ncf_model.pkl')
        server = ModelServer(model, host="0.0.0.0", port=8000)
        server.start()  # Server starts at http://0.0.0.0:8000

        # API Endpoints:
        # POST /predict - Single prediction
        # POST /recommend - Single recommendation
        # POST /batch/predict - Batch predictions
        # POST /batch/recommend - Batch recommendations
        # GET /health - Health check
        # GET /info - Model info

    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """

    def __init__(
            self,
            model,
            host: str = "0.0.0.0",
            port: int = 8000,
            enable_docs: bool = True):
        """
        Initialize model server.

        Args:
            model: Trained recommendation model with predict/recommend methods
            host: Server host address
            port: Server port
            enable_docs: Whether to enable API documentation

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        if not FASTAPI_AVAILABLE:
            raise ImportError(
                "FastAPI not installed. Install with: pip install fastapi uvicorn")

        self.model = model
        self.host = host
        self.port = port

        # Create FastAPI app
        self.app = FastAPI(
            title="CoreRec Model Server",
            description="Production serving for CoreRec recommendation models",
            version="1.0.0",
            docs_url="/docs" if enable_docs else None,
        )

        # Setup logging
        self.logger = logging.getLogger("CoreRecServer")
        self.logger.setLevel(logging.INFO)

        # Setup routes
        self._setup_routes()

    def _setup_routes(self):
        """Setup API routes."""

        @self.app.post("/predict")
        async def predict(request: PredictionRequest):
            """
            Predict score for a single user-item pair.

            Request Body:
                {
                    "user_id": 123,
                    "item_id": 456,
                    "context": {}  // optional
                }

            Response:
                {
                    "user_id": 123,
                    "item_id": 456,
                    "score": 0.8523
                }
            """
            try:
                score = self.model.predict(request.user_id, request.item_id)
                return {
                    "user_id": request.user_id,
                    "item_id": request.item_id,
                    "score": float(score),
                }
            except Exception as e:
                self.logger.error(f"Prediction error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/recommend")
        async def recommend(request: RecommendationRequest):
            """
            Generate recommendations for a user.

            Request Body:
                {
                    "user_id": 123,
                    "top_k": 10,
                    "exclude_items": [1, 2, 3]  // optional
                }

            Response:
                {
                    "user_id": 123,
                    "recommendations": [456, 789, 101, ...],
                    "scores": [0.95, 0.92, 0.89, ...]  // if available
                }
            """
            try:
                recs = self.model.recommend(
                    request.user_id,
                    top_k=request.top_k,
                    exclude_items=request.exclude_items)
                return {
                    "user_id": request.user_id,
                    "recommendations": recs,
                    "top_k": request.top_k}
            except Exception as e:
                self.logger.error(f"Recommendation error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/batch/predict")
        async def batch_predict(request: BatchPredictionRequest):
            """Batch predictions for multiple user-item pairs."""
            try:
                if hasattr(self.model, "batch_predict"):
                    scores = self.model.batch_predict(request.pairs)
                else:
                    scores = [self.model.predict(u, i)
                              for u, i in request.pairs]

                return {
                    "predictions": [
                        {"user_id": u, "item_id": i, "score": float(s)}
                        for (u, i), s in zip(request.pairs, scores)
                    ]
                }
            except Exception as e:
                self.logger.error(f"Batch prediction error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/batch/recommend")
        async def batch_recommend(request: BatchRecommendationRequest):
            """Batch recommendations for multiple users."""
            try:
                if hasattr(self.model, "batch_recommend"):
                    recs = self.model.batch_recommend(
                        request.user_ids, request.top_k)
                else:
                    recs = {
                        uid: self.model.recommend(
                            uid, request.top_k) for uid in request.user_ids}

                return {"recommendations": recs}
            except Exception as e:
                self.logger.error(f"Batch recommendation error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/health")
        async def health():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "model_loaded": self.model is not None,
                "model_fitted": getattr(self.model, "is_fitted", True),
            }

        @self.app.get("/info")
        async def info():
            """Get model information."""
            try:
                if hasattr(self.model, "get_model_info"):
                    return self.model.get_model_info()
                else:
                    return {
                        "model_type": self.model.__class__.__name__,
                        "model_name": getattr(self.model, "name", "Unknown"),
                    }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.exception_handler(Exception)
        async def global_exception_handler(request: Request, exc: Exception):
            """Global exception handler."""
            self.logger.error(f"Unhandled exception: {exc}")
            return JSONResponse(status_code=500, content={"detail": str(exc)})

    def start(self, reload: bool = False):
        """
        Start the server.

        Args:
            reload: Enable auto-reload (development mode)

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        self.logger.info(
            f"Starting CoreRec Model Server on {
                self.host}:{
                self.port}")
        self.logger.info(f"Model: {self.model.__class__.__name__}")
        self.logger.info(f"API Docs: http://{self.host}:{self.port}/docs")

        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            reload=reload,
            log_level="info")
