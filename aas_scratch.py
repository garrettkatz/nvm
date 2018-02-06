import numpy as np

N = 32

TOKENS = {t: np.sign(np.random.randn(N,1)) for t in [
    # instructions
    "NOP",
    "SET",
    "LOAD",
    "GOTO",
    "IF",
    "RET",
    # cpu layers
    "REGISTER1",
    "COMPARE1",
    "COMPARE2",
    "COMPARE3",
    "NOT1",
    "NOT2",
    # cortical layers
    "FEF",
    "TC",
    # values
    "NULL",
    "TRUE",
    "FALSE",
    "LEFT",
    "CENTER",
    "RIGHT",
    # labels
    "LLEFT",
    "LCENTER",
    "LRIGHT",
]}
PATTERNS = {tuple(np.sign(TOKENS[t]).flatten()): t for t in TOKENS}
get_token = lambda v: PATTERNS[tuple(np.sign(v).flatten())]
W = {}

program = [
    "SET", "FEF", "CENTER",
    "LOAD", "COMPARE1", "TC",
    "LOAD", "COMPARE2", "LEFT",
    "LOAD", "REGISTER1", "COMPARE3",
    "IF", "REGISTER1", "LLEFT",
    "LOAD", "COMPARE2", "RIGHT",
    "LOAD", "REGISTER1", "COMPARE3",
    "IF", "REGISTER1", "LRIGHT",
    "GOTO", "LCENTER", "NULL",
    "SET", "FEF", "LEFT",
    "GOTO", "LCENTER", "NULL",
    "SET", "FEF", "RIGHT",
    "GOTO", "LCENTER", "NULL",
    "RET",
]
labels = {
    "LCENTER": 0,
    "LLEFT": 27,
    "LRIGHT": 33,
}

V = 0.9*np.concatenate(tuple(TOKENS[t] for t in program), axis=1) # data
H = 0.9*np.sign(np.random.randn(*V.shape)) # hidden
VH = np.concatenate((V,H), axis=0)

X = VH[:,:-1]
Y = VH[:,1:]
W[("MEM","MEM")] = np.linalg.lstsq(X.T, np.arctanh(Y).T, rcond=None)[0].T

v = VH[:,[0]]
for t in range(len(program)):
    token = get_token(v[:N,:])
    print("%d: %s"%(t, token))
    if token == "RET": break
    v = np.tanh(W[("MEM","MEM")].dot(v))

