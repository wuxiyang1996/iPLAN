REGISTRY = {}

from .coma import COMACritic
from .ippo_critic import R_Critic

REGISTRY["coma"] = COMACritic
REGISTRY["ippo"] = R_Critic
