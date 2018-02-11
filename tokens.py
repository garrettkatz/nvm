import numpy as np

N_LAYER = 64

INSTRUCTIONS = ["NOP","SET","MOV","CMP","JMP","RET"]
CPU_LAYERS = ["OPC","OP1","OP2","OP3","CMPA","CMPB","CMPH","CMPO","MEM","MEMH","NOTI","NOTO"]
USER_LAYERS = ["REG1","REG2","REG3"]
LAYERS = CPU_LAYERS + USER_LAYERS
DEVICES = ["FEF","TC"]
VALUES = ["NULL","TRUE","FALSE","LEFT","CENTER","RIGHT"]
LABELS = ["LOOK","LOOP","GAZE"]

TOKENS = {t: np.sign(np.random.randn(N_LAYER,1)) for t in 
    INSTRUCTIONS + LAYERS + DEVICES + VALUES + LABELS
}
TOKENS["TRUE"] = np.ones((N_LAYER,1))
TOKENS["FALSE"] = -np.ones((N_LAYER,1))

PATTERNS = {tuple(np.sign(TOKENS[t]).flatten()): t for t in TOKENS}
def get_token(v):
    k = tuple(np.sign(v).flatten())
    if k in PATTERNS: return PATTERNS[k]
    else: return "?"
