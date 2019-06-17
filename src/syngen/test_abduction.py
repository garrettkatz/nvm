from queue import Queue
from random import choice
from itertools import chain

"""
State in tree-based Finite State Machine.

Each state contains:
    * set of associated |causes| (can be empty)
    * dictionary of |transitions| keyed by input actions/causes
    * link to |parent| state (knowledge FSMs are trees)
"""
class FSMState:
    counter = 0

    def __init__(self, prev_state=None):
        self.index = FSMState.counter
        FSMState.counter += 1

        # TODO:
        # State cannot have overlap between causes and transitions.
        # This is because dictionary keys are added as explicit lists in state
        #   dictionaries in the neural implementation.
        # Although this seems like an unusual problem, because causes is a list
        #   and transitions is a dictionary, a solution is to encode the overlap
        #   in a separate list:
        #
        # causes = [...]
        # transitions = [...]
        # both = [...]
        #
        # Then, when iterating through causes or transition keys, the additional
        #   list can also be traversed.
        self.causes = set()
        self.transitions = { }
        self.parent = prev_state

    """
    Advance the machine using transition 'inp'.
    Invalid transitions will cause advance() to return None.
    """
    def advance(self, inp):
        return self.transitions.get(inp, None)

    """
    Move backwards through the machine from this state.
    If this state is the initial state, retreat() returns None.
    """
    def retreat(self):
        return self.parent

    """ Append a cause to the state. """
    def add_cause(self, cause):
        self.causes.add(cause)

    """
    Adds an effect path with a cause.
    Runs recursively, consuming an effect each time.
    Once effects are all consumed, the cause is added to the final state.
    """
    def add_path(self, cause, effects):
        try:
            inp = next(effects)
            successor = self.advance(inp)

            if successor is None:
                successor = FSMState(prev_state=self)
                self.transitions[inp] = successor

            successor.add_path(cause, effects)
        except StopIteration:
            self.add_cause(cause)

    """ Gathers all sub-states recursively. """
    def gather(self):
        return list(iter(self))

    """ Recursive iterator. """
    def __iter__(self):
        yield self
        for s in self.transitions.values():
            yield from iter(s)

    def __str__(self):
        return "s%d" % self.index


"""
Builds a Finite State Machine encoding a tree of causal knowledge.

Constructor takes a list of (cause, effect) tuples encoding knowledge:
    fsm = build_fsm(
        [
            ('X', 'ABC'),   # X causes A->B->C
            ('Y', 'AB'),    # Y causes A->B
            ('Z', 'XY'),    # Z causes X->Y
            ...
        ]
    )

To run the machine:
    state = fsm
    for v in 'ABCDE':
        state = state.advance(state)
        print(state.causes)
"""
def build_fsm(knowledge):
    init = FSMState()

    # Add a path for each cause-effect pair
    for cause,effect in knowledge:
        init.add_path(cause, iter(effect))

    return init



"""
Represents an observed instance of a cause with a given |identity| with a list
  of |effects| covering time |start_t| to |end_t| (inclusive/exclusive).

Primitive actions have no effects and take up a single timestep. Their |args|
  come directly from observation, while the |args| of higher-order causes will
  be derived from |effects|.
"""
class Cause:
    counter = 0

    def __init__(self, identity, end_t, effects=[], args={}):
        self.index = Cause.counter
        Cause.counter += 1

        self.identity = identity
        self.end_t = end_t
        self.effects = effects

        # Primitive actions have no effects and last one timestep
        # Higher-order actions start before the first effect
        if len(effects) == 0:
            self.start_t = end_t.previous
        else:
            self.start_t = effects[0].start_t

        # TODO:
        # 1. Validate that effects match cause
        # 2. Extract arguments from effects
        self.args = {}

    def __str__(self):
        return "c%d" % self.index


"""
Timepoint nodes that are chained into a reverse linked list.

Each node contains:
    * pointer to |previous| timepoint
    * |cache| mapping active FSM states to the incoming transition cause
    * set of |causes| that end at this timepoint
"""
class Timepoint:
    counter = 0

    def __init__(self, previous=None):
        self.index = Timepoint.counter
        Timepoint.counter += 1

        self.previous = previous
        self.cache = {}
        self.causes = set()

    """ Spawns a new timepoint and adds a pointer to the current timepoint.  """
    def incr(self):
        return Timepoint(self)

    """
    Adds a |cause| to this timepoint.
    Causes serve as edges in the causal graph because they contain pointers to
      their start and end timepoints.
    """
    def add_cause(self, cause):
        self.causes.add(cause)

    """
    Caches a FSM |state| with the |cause| of its incoming transition.
    This way, when a cause is completed, the effects can be recovered by tracing
      backward through the FSM using the timesteps stored in each cause/effect.
    """
    def cache_state(self, state, cause):
        # TODO:
        # The same FSM state may be cached with differentes.
        #   This must depend on properties of the FSM (to be determined).
        # Conflict can be detected in the NVM.
        # How should this be handled?
        if state in self.cache:
            pass
        self.cache[state] = cause

    """
    Traces back through the timepoints and collects the sequence of causes that
      led from to the cached |state|. This is possible because states are cached
      with their transition causes, and those causes are linked to the start
      time of their associated effects.
    """
    def traceback(self, state):
        effects = []
        cause = self.cache[state]

        while cause is not None:
            effects.insert(0, cause)
            state = state.retreat()
            cause = cause.start_t.cache.get(state, None)

        return effects

    """ Gathers all timepoints recursively. """
    def gather(self):
        return list(iter(self))

    """ Recursive iterator. """
    def __iter__(self):
        yield self
        if self.previous is not None:
            yield from iter(self.previous)

    def __str__(self):
        return "t%d" % self.index

"""
Perform cause-effect reasoning (abduction) using an input sequence of actions
  and a Finite State Machine encoding causal knowledge.

Returns a list of the shortest paths terminating at each time point.

Variables for graph generation:
    * t          : current timepoint
    * q          : action/cause queue
    * a          : observed action
    * v          : cause (may be 'a')
    * state      : fsm_state

Time complexity (worst case):
1. len(input_sequence) * len(causes) * len(FSM)
      -> Queue contains actions/causes
      -> Caches contain FSM states
2. len(input_sequence)
3. len(input_sequence) * len(causes)
4. len(input_sequence)

Space complexity (worst case):
len(input_sequence) * len(causes) * sum(len(effects))
"""
def abduce(fsm, input_sequence):
    # Current time
    curr_t = Timepoint()

    # For each action in the input sequence...
    for a in input_sequence:
        # Increment timestep
        curr_t = curr_t.incr()

        # Create a queue of action/causes to process
        # Initialize with self-cause for observed action
        q = Queue()
        q.put(Cause(a, curr_t))

        # Run until the queue is empty
        while not q.empty():
            # Retrieve action/cause (v) and add it to the graph
            v = q.get()
            curr_t.add_cause(v)

            # Advance all cached FSM states from v's start timepoint (+initial)
            for state in chain([fsm], v.start_t.cache):
                state = state.advance(v.identity)

                # If the FSM advancement was successful...
                if state is not None:
                    # Cache the new state with the transition cause
                    curr_t.cache_state(state, v)

                    # If the state is terminal, enqueue all of its causes with
                    #   the path start timepoint. Because the queue was
                    #   initialized with the observed action, only process
                    #   high-level causes.
                    for cause in state.causes:
                        q.put(Cause(cause, curr_t,
                            effects=curr_t.traceback(state)))


    # Gather timepoints for convenient iteration (reverse order)
    timepoints = curr_t.gather()

    # Use dynamic programming to compute shortest paths from the end to start
    shortest_edge =   { t : None  for t in timepoints }
    shortest_length = { t : 99999 for t in timepoints }
    shortest_length[curr_t] = 0

    for t in timepoints:
        length = 1 + shortest_length[t]

        for cause in t.causes:
            if length < shortest_length[cause.start_t]:
                shortest_length[cause.start_t] = length
                shortest_edge[cause.start_t] = cause

    # Compute the shortest path from end to each node
    shortest_path = { curr_t : "" }
    for t in timepoints[1:]:
        cause = shortest_edge[t]
        shortest_path[t] = shortest_path[cause.end_t] + cause.identity

    return timepoints, shortest_path


def run(fsm, seq, answer=None, verbose=False):
    timepoints, best_path = abduce(fsm, seq)

    if verbose:
        print("Sequence: " + seq)

        for t in timepoints:
            print("%4s : %s" % (str(t), best_path[t]))

        print("Shortest path: %d" % len(best_path[timepoints[-1]]))
        print("Num causes: %d" % sum(len(t.causes) for t in timepoints))
        print("Cache size: %d" % sum(len(t.cache) for t in timepoints))

        if answer is not None:
            print("Correct: %s" % (best_path[timepoints[-1]] == answer))
        print()

        # Label the FSM states and timepoints
        print("Caches")
        for t in timepoints:
            print(str(t), [(str(s), str(o)) for s,o in t.cache.items()])
        print()

        print("Causes")
        for t in timepoints:
            print(str(t), [str(v) for v in t.causes])
        print()

        print("FSM")
        for st in fsm.gather():
            print(str(st), [(inp, str(st2)) for inp,st2 in st.transitions.items()])
        print()

    return best_path[timepoints[-1]]

test_data = [
    (
        [
            ('X', 'ABC'),
            ('Y', 'AB'),
            ('Z', 'BC'),

            ('V', 'AZ'),
            ('W', 'YC'),

            ('S', 'D'),
            ('T', 'D'),

            ('U', 'TE'),

            ('G', 'WU'),
            ('W', 'DEC'),
            ('U', 'SA')
        ],
        'ABCDE',
        'G'
    ),
    (
        [
            ('Y', 'AB'),
            ('W', 'YC'),
            ('G', 'WU'),
            ('K', 'FG'),
        ],
        'FABCU',
        'K',
    ),
    (
        [
            ('X', 'AB'),
            ('Y', 'BC'),
            ('Z', 'XY'),

            ('S', 'AA'),
            ('T', 'BB'),
            ('U', 'CC'),

            ('G', 'SU'),
            ('H', 'TS'),
            ('I', 'XS'),
            ('J', 'UT'),

            ('S', 'GA'),
            ('T', 'BI'),
            ('U', 'TC'),
            ('G', 'CZ'),
        ],
        "".join([choice('ABC') for _ in range(20)]),
        None
    )
]

if __name__ == "__main__":
    for knowledge, seq, answer in test_data:
        run(build_fsm(knowledge), seq, answer, verbose=True)
