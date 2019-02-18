from gen_tree import random_tree

data = random_tree()
it = iter(data)
output = ""

head = 0
memory = ["invalid" for _ in range(1000)]
pointers = {}

# counter: number of nodes created
# end:     first free memory address
# prev:    points to previous node
#              parent if create_child
#              prev sibling if create_sibling
# curr:    points to beginning of current node
ptr_reg = "null"
val_reg = "null"

def read():
    global val_reg
    val_reg = next(it)
    #print("Read %s" % val_reg)

def write():
    global val_reg, output
    #print("Write %s" % val_reg)
    output += val_reg

def ref():
    global ptr_reg, head, pointers
    pointers[ptr_reg] = head

def drf():
    global ptr_reg, head, pointers
    head = pointers[ptr_reg]

def mem_val():
    global val_reg, head, memory
    memory[head] = val_reg

def mem_ptr():
    global ptr_reg, head, memory
    memory[head] = ptr_reg

def rem_val():
    global val_reg, head, memory
    val_reg = memory[head]

def rem_ptr():
    global ptr_reg, head, memory
    ptr_reg = memory[head]

def incr():
    global val_reg
    val_reg = str(int(val_reg)+1)

def decr():
    global val_reg
    val_reg = str(int(val_reg)-1)

def nxt():
    global head
    head += 1

def prv():
    global head
    head -= 1

def print_memory():
    global memory, pointers

    print("Pointers:")
    for k in sorted(pointers.keys()):
        print("%7s : %s" % (k, pointers[k]))
    print("")

    print("Counter: %s" % memory[0])
    i = 1
    while i < pointers["end"]:
        node = memory[i:i+4]
        i += 4
        while memory[i] != "null":
            node.append(memory[i])
            i += 1
        node.append("null")
        node = [" " if x == "null" else x for x in node]
        i += 1
        print("|%s |" % (" |".join(["%3s" % x for x in node])))
    print("")

# Node: (index, parent_index, child_index, sibling_index, data)

"""
Assumes:
    counter points to counter address
    end points to free address
"""
def create_start():
    global val_reg, ptr_reg

    ### PREAMBLE #####################
    # Get current index from counter
    # Increment counter
    # Set current pointer
    # Set index pointer
    ##################################

    # Retrieve new index
    ptr_reg = "counter"
    drf()
    rem_val()

    # Increment index, save, then recover current
    incr()
    mem_val()
    decr()

    # Go to end and save curr pointer
    ptr_reg = "end"
    drf()
    ptr_reg = "curr"
    ref()

    # Write index and bind pointer
    ptr_reg = val_reg
    mem_ptr()
    ref()

    ### INIT #########################
    # Set pointers to null
    # Read and save node data
    # Update the end pointer
    ##################################

    # Go to data location (curr -> parent -> child -> sibling -> data)
    ptr_reg = "null"
    nxt()
    mem_ptr()
    nxt()
    mem_ptr()
    nxt()
    mem_ptr()
    nxt()

    # Read in data until ,
    read()
    while val_reg != ",":
        mem_val()
        nxt()
        read()
    val_reg = "null"
    mem_val()
    nxt()

    # Move end pointer
    ptr_reg = "end"
    ref()

    # Return to beginning of node
    ptr_reg = "curr"
    drf()

"""
Assumes:
    prev points to the parent node address
"""
def create_child():
    global val_reg, ptr_reg
    #print("Create child")

    create_start()

    ### CHILD NODE INIT ##############
    # Set parent index from prev
    #   "null" if parent == curr
    ##################################

    # Get parent index in val_reg
    ptr_reg = "prev"
    drf()
    rem_ptr()
    val_reg = ptr_reg

    # Return to curr
    ptr_reg = "curr"
    drf()

    # Set parent index
    # If parent == curr, use null
    rem_ptr()
    nxt()
    if val_reg != ptr_reg:
        ptr_reg = val_reg
    else:
        ptr_reg = "null"
    mem_ptr()

    # Return to curr
    ptr_reg = "curr"
    drf()

    ### Update parent #################
    ##################################

    # Get index in val_reg
    rem_ptr()
    val_reg = ptr_reg

    # Set parent's child pointer
    nxt()
    rem_ptr()
    if ptr_reg != "null":
        drf()
        nxt()
        nxt()
        ptr_reg = val_reg
        mem_ptr()

    # Go back to current node
    ptr_reg = "curr"
    drf()

    # Parse the current node's children
    create_end()


"""
Assumes:
    prev points to previous sibling address
"""
def create_sibling():
    global val_reg, ptr_reg
    #print("Create sibling")

    create_start()

    ### SIBLING NODE INIT ############
    # Set parent index from prev parent
    ##################################

    # Get prev's parent index in val_reg
    ptr_reg = "prev"
    drf()
    nxt()
    rem_ptr()
    val_reg = ptr_reg

    # Return to curr
    ptr_reg = "curr"
    drf()

    # Set parent index
    rem_ptr()
    nxt()
    ptr_reg = val_reg
    mem_ptr()

    # Return to curr
    ptr_reg = "curr"
    drf()

    ### Update prev sibling ##########
    ##################################

    # Get index in val_reg
    rem_ptr()
    val_reg = ptr_reg

    # Set prev's sibling pointer
    ptr_reg = "prev"
    drf()
    nxt()
    nxt()
    nxt()
    ptr_reg = val_reg
    mem_ptr()

    # Go back to current node
    ptr_reg = "curr"
    drf()

    # Parse the current node's children
    create_end()


"""
Assumes:
    memory head is at current node
"""
def create_end():
    global val_reg, ptr_reg

    # Move prev to curr
    ptr_reg = "prev"
    ref()

    # Check for child
    read()
    if val_reg != ")":
        create_child()

        # Loop, checking for siblings
        read()
        while val_reg != ")":
            read()
            create_sibling()
            read()

    # Set prev pointer for next node
    ptr_reg = "prev"
    ref()

    # Return head to parent (if not null)
    nxt()
    rem_ptr()
    if ptr_reg != "null":
        drf()


def print_node():
    global val_reg, ptr_reg

    # Use curr to mark begining of current node
    ptr_reg = "curr"
    ref()

    # Print (val,
    val_reg = "("
    write()

    nxt()
    nxt()
    nxt()
    nxt()
    rem_val()
    while val_reg != "null":
        write()
        nxt()
        rem_val()

    val_reg = ","
    write()

    # Return head to beginning of current node
    ptr_reg = "curr"
    drf()

    # Recurse on child if it exists
    nxt()
    nxt()
    rem_ptr()
    if ptr_reg != "null":
        drf()
        print_node()
    else:
        prv()
        prv()

    val_reg = ")"
    write()

    # Recurse on sibling if it exists
    # Otherwise, return to parent if it exists
    nxt()
    nxt()
    nxt()
    rem_ptr()
    if ptr_reg != "null":
        val_reg = ","
        write()
        drf()
        print_node()
    else:
        # Return head to parent
        prv()
        prv()

        rem_ptr()
        if ptr_reg != "null":
            drf()

print("Input data:")
print(data)
print("")

read()
if val_reg == "(":
    # Create counter pointer
    ptr_reg = "counter"
    ref()
    val_reg = "0"
    mem_val()

    # Create end pointer
    ptr_reg = "end"
    nxt()
    ref()

    # Create prev pointer (root points to itself)
    ptr_reg = "prev"
    ref()

    # Create the tree
    create_child()

    # Return to root node
    ptr_reg = "counter"
    drf()
    nxt()

    print_memory()
    print_node()

    print("Input data:")
    print(output)
    print("")
    print("Correct? %s" % (output == data))
