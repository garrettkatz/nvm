from random import choice, random, randint

ops = "+*"
vals = tuple(str(x) for x in range(0,19))

class Node:
    def __init__(self, val):
        self.val = val
        self.children = []

    def add_child(self, child):
        self.children.append(child)

    def __str__(self):
        return str(self.val)

def random_tree():
    root = Node(choice(ops))
    queue = [root]
    max_internal = 5
    count = 1

    while len(queue) > 0:
        node = queue[0]
        queue = queue[1:]

        if node.val in ops:
            count += 1
            if count >= max_internal:
                node.add_child(Node(choice(vals)))
                node.add_child(Node(choice(vals)))
            else:
                children = [
                    Node(choice(vals if random() < 0.5 else ops))
                        for _ in range(randint(1,3))]

                for child in children:
                    node.add_child(child)

                for c in children:
                    queue.append(c)

    def tree_string(node, output=""):
        output += "(%s" % str(node)
        if len(node.children) == 0:
            output += ","
        else:
            for c in node.children:
                output += ","
                output = tree_string(c, output)
        output += ")"
        return output

    return tree_string(root)
