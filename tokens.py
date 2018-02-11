import numpy as np

N_LAYER = 32

INSTRUCTIONS = ["NOP","SET","LOAD","GOTO","IF","RET"]
CPU_LAYERS = ["OPCODE","OPERAND1","OPERAND2","COMPAREH","MEM1","MEM2"]
USER_LAYERS = ["REGISTER1","COMPARE1","COMPARE2","COMPARE","NOT1","NOT2"]
LAYERS = CPU_LAYERS + USER_LAYERS
DEVICES = ["FEF","TC"]
VALUES = ["NULL","TRUE","FALSE","LEFT","CENTER","RIGHT"]
LABELS = ["LLEFT","LCENTER","LRIGHT"]

TOKENS = {t: np.sign(np.random.randn(N_LAYER,1)) for t in 
    INSTRUCTIONS + LAYERS + DEVICES + VALUES + LABELS
}
PATTERNS = {tuple(np.sign(TOKENS[t]).flatten()): t for t in TOKENS}
def get_token(v):
    k = tuple(np.sign(v).flatten())
    if k in PATTERNS: return PATTERNS[k]
    else: return "UNKNOWN"
