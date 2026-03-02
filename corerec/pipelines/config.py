"""
Pipeline Configuration

Load pipeline configurations from YAML or dict.
Enables config-driven pipeline construction.
"""

from typing import Any, Dict, Optional, Union
from pathlib import Path


def load_pipeline_config(
    source: Union[str, Path, Dict]
) -> Dict[str, Any]:
    """
    Load pipeline configuration from YAML file or dict.
    
    Args:
        source: path to YAML file or config dict
    
    Returns:
        Configuration dictionary
    
    Example YAML::

        pipeline:
          name: production_v1
          
          retrieval:
            k: 500
            fusion: rrf
            sources:
              - type: collaborative
                model: sar
                weight: 1.0
              - type: semantic
                encoder: all-MiniLM-L6-v2
                weight: 0.5
          
          ranking:
            k: 100
            type: pointwise
            model_path: models/ranker.pkl
          
          reranking:
            - type: diversity
              lambda: 0.7
            - type: business
              boost:
                - item: 123
                  multiplier: 2.0
    """
    if isinstance(source, dict):
        return source
    
    path = Path(source)
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    if path.suffix in ['.yaml', '.yml']:
        try:
            import yaml
            with open(path) as f:
                return yaml.safe_load(f)
        except ImportError:
            raise ImportError("PyYAML required for YAML config. Install: pip install pyyaml")
    
    elif path.suffix == '.json':
        import json
        with open(path) as f:
            return json.load(f)
    
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")


def build_pipeline_from_config(config: Dict[str, Any]) -> "RecommendationPipeline":
    """
    Build a pipeline from configuration dict.
    
    This is a factory function that instantiates all components
    based on the config specification.
    
    Args:
        config: pipeline configuration
    
    Returns:
        Configured RecommendationPipeline
    """
    from .orchestrator import RecommendationPipeline, PipelineConfig
    
    pipeline_config = config.get('pipeline', config)
    
    # create base config
    cfg = PipelineConfig(
        retrieval_k=pipeline_config.get('retrieval', {}).get('k', 500),
        ranking_k=pipeline_config.get('ranking', {}).get('k', 100),
        final_k=pipeline_config.get('final_k', 10),
        fusion_strategy=pipeline_config.get('retrieval', {}).get('fusion', 'rrf'),
    )
    
    pipeline = RecommendationPipeline(
        config=cfg,
        name=pipeline_config.get('name', 'pipeline'),
    )
    
    # add retrievers
    retrieval_cfg = pipeline_config.get('retrieval', {})
    for source in retrieval_cfg.get('sources', []):
        retriever = _build_retriever(source)
        if retriever:
            pipeline.add_retriever(retriever, weight=source.get('weight', 1.0))
    
    # set ranker
    ranking_cfg = pipeline_config.get('ranking', {})
    if ranking_cfg:
        ranker = _build_ranker(ranking_cfg)
        if ranker:
            pipeline.set_ranker(ranker)
    
    # add rerankers
    reranking_cfg = pipeline_config.get('reranking', [])
    for rr_cfg in reranking_cfg:
        reranker = _build_reranker(rr_cfg)
        if reranker:
            pipeline.add_reranker(reranker)
    
    return pipeline


def _build_retriever(cfg: Dict) -> Optional[Any]:
    """Build a retriever from config."""
    rtype = cfg.get('type')
    
    if rtype == 'collaborative':
        from corerec.retrieval import CollaborativeRetriever
        return CollaborativeRetriever(name=cfg.get('name', 'collaborative'))
    
    elif rtype == 'semantic':
        from corerec.retrieval import SemanticRetriever
        return SemanticRetriever(
            encoder=cfg.get('encoder'),
            name=cfg.get('name', 'semantic'),
        )
    
    elif rtype == 'popularity':
        from corerec.retrieval import PopularityRetriever
        return PopularityRetriever(name=cfg.get('name', 'popularity'))
    
    return None


def _build_ranker(cfg: Dict) -> Optional[Any]:
    """Build a ranker from config."""
    rtype = cfg.get('type')
    
    if rtype == 'pointwise':
        from corerec.ranking import PointwiseRanker
        return PointwiseRanker(name=cfg.get('name', 'pointwise'))
    
    elif rtype == 'feature_cross':
        from corerec.ranking import FeatureCrossRanker
        return FeatureCrossRanker(
            crosses=cfg.get('crosses', []),
            name=cfg.get('name', 'feature_cross'),
        )
    
    return None


def _build_reranker(cfg: Dict) -> Optional[Any]:
    """Build a reranker from config."""
    rtype = cfg.get('type')
    
    if rtype == 'diversity':
        from corerec.reranking import DiversityReranker
        return DiversityReranker(
            lambda_=cfg.get('lambda', 0.7),
            name=cfg.get('name', 'diversity'),
        )
    
    elif rtype == 'fairness':
        from corerec.reranking import FairnessReranker
        # need group_fn which can't be in config
        return None
    
    elif rtype == 'business':
        from corerec.reranking import BusinessRulesReranker
        reranker = BusinessRulesReranker(name=cfg.get('name', 'business'))
        
        # apply boosts
        for boost in cfg.get('boost', []):
            reranker.add_boost(boost['item'], boost['multiplier'])
        
        # apply blocklist
        if cfg.get('blocklist'):
            reranker.add_blocklist(cfg['blocklist'])
        
        return reranker
    
    return None
