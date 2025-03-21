# mintos life
# from corerec.engines.contentFilterEngine import *

# hardcore 
from corerec.engines.contentFilterEngine.context_personalization import (context_aware, item_profiling, user_profiling)
from corerec.engines.contentFilterEngine.embedding_representation_learning import (doc2vec,personalized_embeddings,word2vec)
from corerec.engines.contentFilterEngine.fairness_explainability import (explainable,fairness_aware,privacy_preserving)
from corerec.engines.contentFilterEngine.graph_based_algorithms import (gnn,graph_filtering,semantic_models)
from corerec.engines.contentFilterEngine.hybrid_ensemble_methods import (attention_mechanisms,ensemble_methods,hybrid_collaborative)
from corerec.engines.contentFilterEngine.learning_paradigms import (few_shot,meta_learning,transfer_learning,zero_shot)
from corerec.engines.contentFilterEngine.miscellaneous_techniques import (cold_start,feature_selection,noise_handling)
from corerec.engines.contentFilterEngine.multi_modal_cross_domain_methods import (cross_domain,cross_lingual,multi_modal)
from corerec.engines.contentFilterEngine.nn_based_algorithms import (AITM,cnn,dkn,DSSM,lstur,naml,npa,nrms,rnn,TDM,transformer,vae,WidenDeep,Word2Vec,Youtube_dnn,TRA_MIND)
from corerec.engines.contentFilterEngine.other_approaches import (ontology_based,rule_based,sentiment_analysis)
from corerec.engines.contentFilterEngine.performance_scalability import (feature_extraction,load_balancing,scalable_algorithms)
from corerec.engines.contentFilterEngine.probabilistic_statistical_methods import (bayesian,fuzzy_logic,lda,lsa)
from corerec.engines.contentFilterEngine.traditional_ml_algorithms import (decision_tree,lightgbm,LR,svm,tfidf,vw)

# these mpodules are exposed in super-set
from corerec.engines.contentFilterEngine import cr_contentFilterFactory
from corerec.engines.contentFilterEngine import tfidf_recommender
from corerec.engines.contentFilterEngine import base_recommender

