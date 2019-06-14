"""
Model of the memory unit of the Neural Virtual Machine (NVM).
The unit is made up of two components:
    1. Registers
    2. Memory space

Registers contain a vocabulary that is used to create pointers into the memory
    space, and to create labeled connections between memory states.  In turn,
    memory states can have registers "stored" in them for later retrieval.

The names of registers are significant because they can be referred to in NVM
    programs.  The names of memory states are arbitrary.

MEM "stores" a register in a memory state
REM retrieves the register

REF sets a register as a pointer to a memory state
DRF retrieves the memory state

MREF creates a transition between two memory states labeled using a register
MDRF retrieves the target state given a start state and a register
"""
class Memory:
    def __init__(self):
        # Register -> Memory
        self.mr = dict()

        # Memory -> Register
        self.rm = dict()

        # Memory -> Memory
        self.mm = dict()

        # Counter for memory state identifiers
        self.mem_counter = 0

    """
    Creates a new unique memory state
    """
    def gen_mem(self):
        self.mem_counter += 1
        return self.mem_counter

    """
    Binds a memory state to a register
    """
    def mem(self, mem_state, var):
        self.rm[mem_state] = var

    def rem(self, mem_state):
        return self.rm[mem_state]

    """
    Binds a register to a memory state
    """
    def ref(self, var, mem_state):
        self.mr[var] = mem_state

    def drf(self, var):
        return self.mr[var]

    """
    Binds two memory states together using a register
    """
    def mref(self, mem_state_a, var, mem_state_b):
        self.mm[(mem_state_a, var)] = mem_state_b

    def mdrf(self, mem_state_a, var):
        return self.mm[(mem_state_a, var)]


"""
Demonstration of creating multiple linked lists in memory.
Each list is associated with a register which serves as its head pointer.
Nodes are added to the list by creating transitions between memory states,
    which serve as nodes in the list.  These transitions are labeled "next".
    Each node gets a unique numbered register associated it.  The nodes are
    added round robin, so the numbers in a list will not be consecutive.

After construction, this code will iterate through each list and collect the
    numbered registers for each node, and print them all out at once.

Note: local python variables are only used to hold the Memory object, a
    counter for labeling the nodes, and a list of labels for output formatting.
    All of the list data is within the Memory object.  This makes the code
    easy to convert into an NVM program, which must use the Memory mechanism for
    storing data.
"""
def demo_lists(num_lists, length):
    if num_lists <= 0 or length <= 0:
        raise ValueError("Number of lists and list length must be positive!")

    counter = 1
    names = [str(x) for x in range(num_lists)]

    mem = Memory()

    ### Create lists
    for name in names:
        mem.ref(name, mem.gen_mem())

    # Initialize heads
    for name in names:
        mem.mem(mem.drf(name), str(counter))
        counter += 1

    # Add tail pointers
    for name in names:
        mem.ref("%s_tail" % name, mem.drf(name))

    ### Round robin on the lists adding nodes
    for _ in range(length-1):
        for name in names:
            # Create a new node
            mem.ref("new", mem.gen_mem())

            # Store counter in node
            mem.mem(mem.drf("new"), str(counter))
            counter += 1

            # Add node to end of list using tail pointer
            mem.mref(
                mem.drf("%s_tail" % name),
                "next",
                mem.drf("new"))

            # Update tail pointer
            mem.ref("%s_tail" % name, mem.drf("new"))

    ### Print the lists
    for name in names:
        # Collect values for output formatting
        vals = []

        # Create temporary pointer to head of list
        mem.ref("temp", mem.drf(name))

        # Dereference temp and read memory
        vals.append(
            mem.rem(
                mem.drf("temp")))

        # Iterate through the rest of the list
        for _ in range(length-1):
            # Jump to next memory node
            mem.ref("temp",
                mem.mdrf(
                    mem.drf("temp"), "next"))

            # Dereference temp and read memory
            vals.append(
                mem.rem(
                    mem.drf("temp")))

        # Format output and print
        print("List %s: %s" % (name,
            ','.join("%5s" % val for val in vals)))


# Demo ten lists with five elements each
demo_lists(10, 5)
