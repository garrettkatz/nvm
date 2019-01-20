"""
Preprocessor for VM assembly
"""

def preprocess(programs, register_names):

    lines, labels, tokens = {}, {}, set()
    for name, program in programs.items():
    
        labels[name] = {} # map label to line number
    
        # split up lines and remove blanks and full line comments
        lines[name] = [line.strip() for line in program.splitlines()
            if len(line.strip()) > 0 and line.strip()[0] != "#"]
    
        for l in range(len(lines[name])):
    
            # remove comments
            comment = lines[name][l].find("#")
            if comment > -1: lines[name][l] = lines[name][l][:comment]
    
            # split out tokens
            lines[name][l] = lines[name][l].split()
    
            # check for label
            if lines[name][l][0][-1] == ":":
                # remove and save label
                labels[name][lines[name][l][0][:-1]] = l
                lines[name][l] = lines[name][l][1:]
    
            # pad with nulls
            while len(lines[name][l]) < 3:
                lines[name][l].append("null")
    
            # replace generic instructions with value/device distinctions
            if lines[name][l][0] in ["mov", "cmp"]:
                if lines[name][l][2] in register_names: lines[name][l][0] += "d"
                else: lines[name][l][0] += "v"
            if lines[name][l][0] in ["jmp","sub"]:
                if lines[name][l][1] in register_names: lines[name][l][0] += "d"
                else: lines[name][l][0] += "v"
    
        tokens = tokens.union([tok for line in lines[name] for tok in line[1:]])
    
    return lines, labels, tokens

def measure_programs(programs, register_names, extra_tokens=[]):
    lines, labels, tokens = preprocess(programs, register_names)
    all_tokens = tokens | set(register_names + ["null"] + extra_tokens)
    for name in labels:
        all_tokens |= set(labels[name].keys())
    num_lines = sum([len(lines[name]) for name in lines])
    num_patterns = len(tokens | set(extra_tokens))
    return num_lines, num_patterns, all_tokens
