"""XLM-RoBERTa model implementation and Hugging Face weight adapters."""

from embex.models.xlm_roberta.loading_weights import (
    load_xlm_roberta_weights,
    update_xlm_roberta_weights,
    xlm_roberta_hf_state_dict,
    xlm_roberta_mlm_hf_state_dict,
)
from embex.models.xlm_roberta.modeling import (
    XLMRobertaConfig,
    XLMRobertaEmbedding,
    XLMRobertaMLMHead,
    create_position_ids,
    create_xlm_roberta,
)

__all__ = [
    "XLMRobertaConfig",
    "XLMRobertaEmbedding",
    "XLMRobertaMLMHead",
    "create_position_ids",
    "create_xlm_roberta",
    "load_xlm_roberta_weights",
    "update_xlm_roberta_weights",
    "xlm_roberta_hf_state_dict",
    "xlm_roberta_mlm_hf_state_dict",
]
