# Quick completion script for remaining batch 2 models
# Will add AutoInt, Bert4Rec, DLRM, FFM, Caser, BiVAE, AFM

REMAINING_MODELS = {
    "AutoInt": {
        "full_name": "Automatic Feature Interaction",
        "paper": "Song et al. 2019 - AutoInt: Automatic Feature Interaction Learning",
        "intro": "uses multi-head self-attention to automatically learn feature interactions of arbitrary order",
    },
    "Bert4Rec": {
        "full_name": "BERT for Sequential Recommendation",
        "paper": "Sun et al. 2019 - BERT4Rec: Sequential Recommendation with Bidirectional Encoder",
        "intro": "applies bidirectional self-attention (BERT) to model user sequences in both directions",
    },
    "DLRM": {
        "full_name": "Deep Learning Recommendation Model",
        "paper": "Naumov et al. 2019 - Deep Learning Recommendation Model (Facebook)",
        "intro": "processes categorical and continuous features separately, then combines via explicit feature interactions",
    },
    "FFM": {
        "full_name": "Field-aware Factorization Machine",
        "paper": "Juan et al. 2016 - Field-aware Factorization Machines",
        "intro": "extends FM by introducing field-aware latent factors for more expressive feature interactions",
    },
    "Caser": {
        "full_name": "Convolutional Sequence Embedding",
        "paper": "Tang & Wang 2018 - Personalized Top-N Sequential Recommendation",
        "intro": "applies CNNs to user sequences treating them as images for pattern extraction",
    },
    "BiVAE": {
        "full_name": "Bilateral Variational Autoencoder",
        "paper": "Collaborative Variational Autoencoder",
        "intro": "uses variational inference to learn user and item representations jointly",
    },
    "AFM": {
        "full_name": "Attentional Factorization Machine",
        "paper": "Xiao et al. 2017 - Attentional Factorization Machines",
        "intro": "adds attention mechanism to FM to learn importance of different feature interactions",
    },
}

print("Models to add:", len(REMAINING_MODELS))
for model_name, info in REMAINING_MODELS.items():
    print(f"  - {model_name}: {info['full_name']}")
