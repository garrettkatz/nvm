"""
Preprocessor for VM assembly
"""

def preprocess(program, register_names):

    labels = dict() # map label to line number

    # split up lines and remove blanks and full line comments
    lines = [line.strip() for line in program.splitlines()
        if len(line.strip()) > 0 and line.strip()[0] != "#"]

    for l in range(len(lines)):

        # remove comments
        comment = lines[l].find("#")
        if comment > -1: lines[l] = lines[l][:comment]

        # split out tokens
        lines[l] = lines[l].split()

        # check for label
        if lines[l][0][-1] == ":":
            # remove and save label
            labels[lines[l][0][:-1]] = l
            lines[l] = lines[l][1:]

        # pad with nulls
        while len(lines[l]) < 3:
            lines[l].append("null")

        # replace generic instructions with value/device distinctions
        if lines[l][0] in ["mov", "cmp"]:
            if lines[l][2] in register_names: lines[l][0] += "d"
            else: lines[l][0] += "v"
        if lines[l][0] in ["jmp","sub"]:
            if lines[l][1] in register_names: lines[l][0] += "d"
            else: lines[l][0] += "v"

    return lines, labels
