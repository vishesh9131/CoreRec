"""
Model Server for Production Serving

FastAPI-based REST API server for serving recommendation models.

Author: Vishesh Yadav (mail: sciencely98@gmail.com)
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import logging

try:
    import numpy as np
except ImportError:
    np = None


def _to_json_safe(value: Any) -> Any:
    """Convert numpy scalars and nested lists to JSON-serializable Python types."""
    if np is not None and isinstance(value, np.generic):
        return value.item()
    if isinstance(value, list):
        return [_to_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_json_safe(v) for k, v in value.items()}
    return value

def _no_nan(value: Any) -> Any:
    """NaN/inf -> None: JSON has no NaN, and metrics on an empty window are NaN."""
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        return None
    if isinstance(value, dict):
        return {k: _no_nan(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_no_nan(v) for v in value]
    return _to_json_safe(value)


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


class FeedbackRequest(BaseModel):
    """A user acted on a recommended item."""

    user_id: Any
    item_id: Any
    event: str = "click"
    request_id: Optional[str] = None


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
        from corerec.engines import DCN

        model = DCN.load('artifacts/dcn')
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
            enable_docs: bool = True,
            metadata: Optional[Dict[str, Any]] = None,
            fallback_items: Optional[List[Any]] = None,
            feedback_log: Any = None,
            traffic: Optional[Dict[str, float]] = None,
            reload_fn: Any = None,
            admin_token: Optional[str] = None,
            feedback_token: Optional[str] = None):
        """
        Initialize model server.

        Args:
            model: Trained recommendation model with predict/recommend methods
            host: Server host address
            port: Server port
            enable_docs: Whether to enable API documentation
            metadata: Extra fields reported by GET /info under "artifact"
                (for example the training manifest)
            fallback_items: Items, best first, returned to users the model
                cannot answer (unknown users, empty results). Responses say
                which path answered in their "source" field.
            feedback_log: path or FeedbackLog. Enables impression logging,
                POST /feedback and GET /metrics.
            traffic: for A/B tests pass ``model`` as {"name": model, ...} and the
                share of users each gets, e.g. {"control": 0.9, "treatment": 0.1}.
                Users are assigned by a stable hash, so each sees one variant.
            reload_fn: zero-argument callable returning a fresh model; enables
                POST /reload (e.g. after ``corerec retrain`` replaced the artifact).
            admin_token: when set, POST /reload needs ``Authorization: Bearer <token>``.
            feedback_token: when set, POST /feedback needs ``Authorization: Bearer <token>``.
                Without them anyone who reaches the port can swap the model or
                write clicks that retraining learns from; set both unless the
                port is private.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        if not FASTAPI_AVAILABLE:
            raise ImportError(
                "FastAPI not installed. Install with: pip install fastapi uvicorn")

        from corerec.serving.feedback import FeedbackLog

        self.models = dict(model) if isinstance(model, dict) else {"default": model}
        if traffic is not None and set(traffic) != set(self.models):
            raise ValueError(f"traffic names {sorted(traffic)} must match models {sorted(self.models)}")
        self.traffic = traffic or {name: 1.0 for name in self.models}
        self.model = next(iter(self.models.values()))
        self.feedback_log = (feedback_log if feedback_log is None or isinstance(feedback_log, FeedbackLog)
                             else FeedbackLog(feedback_log))
        self.reload_fn = reload_fn
        self.admin_token = admin_token
        self.feedback_token = feedback_token
        self.host = host
        self.port = port
        self.metadata = metadata
        self.fallback_items = list(fallback_items) if fallback_items else None

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

    def _variant(self, user_id: Any) -> str:
        if len(self.models) == 1:
            return next(iter(self.models))
        from corerec.serving.feedback import assign_variant

        return assign_variant(user_id, self.traffic)

    def _recommend(self, user_id: Any, top_k: int, exclude_items: List[Any], variant: Optional[str] = None):
        """Return (items, source); source is "model" or "fallback"."""
        from corerec.api.exceptions import RecommendationError

        model = self.models[variant or self._variant(user_id)]
        try:
            recs = model.recommend(user_id, top_k=top_k, exclude_items=exclude_items)
        except RecommendationError:
            # Raised for users the model never saw; anything else is a real error.
            if self.fallback_items is None:
                raise
            recs = []
        if recs or self.fallback_items is None:
            return recs, "model"
        excluded = set(exclude_items or ())
        return [i for i in self.fallback_items if i not in excluded][:top_k], "fallback"

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
                model = self.models[self._variant(request.user_id)]
                score = model.predict(request.user_id, request.item_id)
                return {
                    "user_id": _to_json_safe(request.user_id),
                    "item_id": _to_json_safe(request.item_id),
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
                variant = self._variant(request.user_id)
                recs, source = self._recommend(
                    request.user_id, request.top_k, request.exclude_items, variant)
                body = {
                    "user_id": _to_json_safe(request.user_id),
                    "recommendations": _to_json_safe(recs),
                    "top_k": request.top_k,
                    "source": source,
                    "variant": variant}
                if self.feedback_log is not None:
                    # echo request_id back on POST /feedback to attribute the click
                    body["request_id"] = self.feedback_log.impression(
                        request.user_id, recs, variant=variant, source=source)
                return body
            except Exception as e:
                self.logger.error(f"Recommendation error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/batch/predict")
        async def batch_predict(request: BatchPredictionRequest):
            """Batch predictions for multiple user-item pairs."""
            try:
                if len(self.models) == 1 and hasattr(self.model, "batch_predict"):
                    scores = self.model.batch_predict(request.pairs)
                else:
                    scores = [self.models[self._variant(u)].predict(u, i)
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
                if len(self.models) == 1 and hasattr(self.model, "batch_recommend"):
                    recs = self.model.batch_recommend(
                        request.user_ids, request.top_k)
                else:
                    recs = {
                        uid: self._recommend(uid, request.top_k, [])[0]
                        for uid in request.user_ids}

                return {"recommendations": recs}
            except Exception as e:
                self.logger.error(f"Batch recommendation error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        def _check_token(http: "Request", token: Optional[str]) -> None:
            if token is None:
                return
            import hmac

            sent = http.headers.get("authorization", "")
            # constant-time compare, so the token can't be guessed byte by byte
            if not hmac.compare_digest(sent.encode(), f"Bearer {token}".encode()):
                raise HTTPException(status_code=401, detail="missing or wrong bearer token",
                                    headers={"WWW-Authenticate": "Bearer"})

        @self.app.post("/feedback")
        async def feedback(request: FeedbackRequest, http: Request):
            """Record that a user clicked (or bought, ...) an item.

            Pass the request_id from the /recommend response so the click is
            credited to the list and variant that showed the item.
            """
            _check_token(http, self.feedback_token)
            if self.feedback_log is None:
                raise HTTPException(status_code=404, detail="feedback logging is off; "
                                    "start the server with a feedback log")
            self.feedback_log.feedback(request.user_id, request.item_id, request.event,
                                       request.request_id)
            return {"status": "recorded"}

        @self.app.get("/metrics")
        async def metrics(recent: int = 1000):
            """Online metrics per variant, A/B comparison and drift alerts."""
            if self.feedback_log is None:
                raise HTTPException(status_code=404, detail="feedback logging is off")
            out = {"variants": self.feedback_log.metrics(),
                   "drift": self.feedback_log.drift(recent=recent)}
            names = list(self.models)
            if len(names) == 2:
                try:
                    out["ab_test"] = self.feedback_log.compare(*names)
                except (ValueError, ZeroDivisionError):
                    out["ab_test"] = None
            for alert in out["drift"]["alerts"]:
                self.logger.warning(f"drift: {alert}")
            return _no_nan(out)

        @self.app.post("/reload")
        async def reload(http: Request):
            """Swap in a fresh model from reload_fn without restarting."""
            _check_token(http, self.admin_token)
            if self.reload_fn is None or len(self.models) != 1:
                raise HTTPException(status_code=404, detail="reload is not configured")
            name = next(iter(self.models))
            self.models[name] = self.model = self.reload_fn()
            return {"status": "reloaded", "model": type(self.model).__name__}

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
                    info = dict(self.model.get_model_info())
                else:
                    info = {
                        "model_type": self.model.__class__.__name__,
                        "model_name": getattr(self.model, "name", "Unknown"),
                    }
                if self.metadata is not None:
                    info["artifact"] = self.metadata
                return _to_json_safe(info)
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
            f"Starting CoreRec Model Server on {self.host}:{self.port}")
        self.logger.info(f"Model: {self.model.__class__.__name__}")
        self.logger.info(f"API Docs: http://{self.host}:{self.port}/docs")

        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            reload=reload,
            log_level="info")
