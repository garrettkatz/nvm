from preprocessing import preprocess
from orthogonal_patterns import hadamard, randomize_hadamard

def encode_value_tokens(nvmnet, tokens, orthogonal=False):
    if orthogonal:
        patterns = hadamard(layer.size, len(all_tokens))
        layer = nvmnet.layers[name]
        patterns = randomize_hadamard(patterns)
        patterns = (layer.activator.on + layer.activator.off)/2 + \
            (layer.activator.on - layer.activator.off)*patterns/2
        for p,token in enumerate(all_tokens):
            layer.coder.encode(token, patterns[:,[p]])


def encode_program_tokens(nvmnet, program, name, orthogonal=False):
    
    ### Preprocess program string
    lines, labels = preprocess(program, nvmnet.devices.keys())
    program_tokens = 
    opcodes, operands = set(), set(nvmnet.devices.keys())
    for line in lines:
        opcodes.add(line[0])
        operands.update(set(line[1:]))
