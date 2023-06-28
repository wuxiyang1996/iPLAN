REGISTRY = {}

from .rnn_agent import RNNAgent
from .ippo_actor import R_Actor

REGISTRY["rnn"] = RNNAgent
REGISTRY["ippo"] = R_Actor
