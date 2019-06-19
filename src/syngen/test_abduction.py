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
        self.causes = [ ]
        self.transitions = { }
        self.parent = prev_state

    """
    Advance the machine using transition 'inp'.
    Invalid transitions will cause advance() to return None.
    """
    def advance(self, inp):
        return self.transitions.get(inp, None)

    """ Recursive iterator. """
    def __iter__(self):
        yield self
        yield from chain.from_iterable(self.transitions.values())

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
    for cause,effects in knowledge:
        state = init

        for effect in effects:
            successor = state.advance(effect)

            if successor is None:
                successor = FSMState(prev_state=state)
                state.transitions[effect] = successor
            state = successor

        state.causes.append(cause)

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

    def __init__(self, identity, end_t, source_state=None, source_cause=None, args={}):
        self.index = Cause.counter
        Cause.counter += 1

        self.identity = identity
        self.end_t = end_t
        self.source_state = source_state
        self.cache = { }
        self.effects = [ ]

        if source_cause is None:
            # Primitive actions have no effects and last one timestep
            self.start_t = end_t.previous
        else:
            # Higher-order actions start before the first effect
            # Trace back to the first effect in the path
            while source_cause is not None:
                self.effects.insert(0, source_cause)
                source_cause = source_cause.cache[source_state]
                source_state = source_state.parent
            self.start_t = self.effects[0].start_t

        # TODO:
        # 1. Validate that effects match cause
        # 2. Extract arguments from effects
        self.args = {}

    """
    Caches a FSM |state| with the |cause| of its incoming transition.
    This way, when a cause is completed, the effects can be recovered by tracing
      backward through the FSM using the timesteps stored in each cause/effect.
    """
    def cache_state(self, state, cause):
        if state in self.cache:
            pass
        self.cache[state] = cause

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
        self.causes = [ ]

    """ Spawns a new timepoint and adds a pointer to the current timepoint.  """
    def incr(self):
        return Timepoint(self)

    """
    Adds a |cause| to this timepoint.
    Causes serve as edges in the causal graph because they contain pointers to
      their start and end timepoints.
    """
    def add_cause(self, cause):
        self.causes.append(cause)

    """ Recursive iterator. """
    def __iter__(self):
        yield self
        if self.previous is not None:
            yield from self.previous

    def __str__(self):
        return "t%d" % self.index

"""
Perform cause-effect reasoning (abduction) using an input sequence of actions
  and a Finite State Machine encoding causal knowledge.

Returns a list of the shortest paths terminating at each time point.

Variables for graph generation:
    * curr_t     : current timepoint
    * a          : observed action
    * q          : action/cause queue
    * v          : cause (may be 'a')
    * state      : fsm_state

    FSM transitioning (process)
    * curr_t     : current timepoint
    * v          : current cause of transition
    * pre_cause  : cause of prior transition
    * pre_state  : prior state

Time complexity (worst case):
1. Graph generation (TODO)

    For each timestep (length of action sequence):
       For each cause v generated at curr_t:
           For each cause at v's start time (v_t):
               For each cached state:
                   If terminal transition:
                       For each cause in state, create new cause:
                           For each effect in FSM path:
                               Reverse transition through FSM

    Worst case:
    O = len(action_sequence) * len(causes) * len(causes) *
        * (len(FSM) + sum(len(cause_effect) for cause_effect in knowledge))


    When the same causal identity can be decomposed into the same
        sequence of primitive actions in two different ways:

    X : A
    Y : A->X  (A->A)
    Y : X->A  (A->A)

    Then two causes can share identities and start/end timepoints, and can
        be used to transition to the same state in the same timepoint.

    If this is not the case, then the double loop over causes and caches
        is guaranteed not to exceed the total number of transitions in the
        finite state machine.

    O = len(action_sequence) * len(causes)
        * (len(FSM) + sum(len(cause_effect) for cause_effect in knowledge))

    If the start time is also cached, and the effects list does not need to
        be computed during graph generation, then Cause construction is O(1).
    If each terminal node is restricted to having a single cause, then then
        each transition is O(1):

    O = len(action_sequence) * len(causes) * len(FSM)

    By pruning the graph to keep the total number of timepoints and causes
        below a constant of memory size, the len(action_sequence) + len(causes)
        terms are eliminated, and the complexity is bounded to the size of
        the finite state machine:

    C = len(memory) >= len(action_sequence) + len(causes)
    O = C * len(FSM) = O(FSM)

    This limits the maximum length of the action sequence that can be processed,
        and limits the quality of causal explanations that can be constructed
        as the sequence gets longer.  However, as long as the length of the
        sequence fits in memory, the trivially true explanation (each action
        explains itself) can still be returned.


    Practically speaking, the total number of cached states at each timepoint
       will be much smaller than the size of the FSM, but in extremely
       degenerate cases, it may be equal. For example:

    X : A
    X : A->A
    X : A->A->A
    ...

    A sequence of A's will fill the cache with all the states in the FSM.
    Therefore, the total cache size at a timepoint depends on properties of
        the input action sequence and the knowledge base.

    My intuition is that the complexity can be reduced with additional
        guarantees about repetition in the input sequence and knowledge base.
    The strategy for determining this should be to consider the possible
        set of active FSM states at any given timepoint.

    To summarize, if:
        1. No causal identity can be decomposed into the same action sequence in
            more than one way
        2. Cause construction does not require computing effects lists
        3. Each terminal FSM node only contains a single causal identity
        4. Memory constrains action sequence length and size of Cause set

    Then the computational complexity of the algorithm is:
    O(FSM) = len(FSM)

    If 2. is not true, then the complexity is bounded by the total size of the
        knowledge base, which is the sum of effect sequence length for each
        causal relationship:
    O(knowledge) = sum(len(cause_effect) for cause_effect in knowledge)


2. Dynamic programming
   O = len(causes)
3. Computing shortest path
   O = len(input_sequence)

Space complexity:
    Memory states:
        Fixed based on knowledge base:
            States in FSM
            Unique causal identities
        Varies based on action sequence:
            Timepoints
            Causes identified during inference

        O = len(FSM) + len(set(causal_identities))
            + len(action_sequence) + len(causes)

        Bounding the sequence length number of identified causes:
        C = len(memory) >= len(action_sequence) + len(causes)
        O = len(FSM) + len(set(causal_identities)) + C


    Transitions:
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

            def process(pre_cause, pre_state):
                state = pre_state.advance(v.identity)

                # If the FSM advancement was successful...
                if state is not None:
                    # Cache the new state with the transition cause
                    v.cache_state(state, pre_cause)

                    # If the state is terminal, enqueue all of its causes
                    #   with the path start timepoint. Because the queue was
                    #   initialized with the observed action, only process
                    #   high-level causes.
                    for cause in state.causes:
                        q.put(Cause(cause, curr_t, state, v))

            # Process initial state
            process(None, fsm)

            # Process all states cached in causes at v's start time
            for pre_cause in v.start_t.causes:
                for pre_state in pre_cause.cache:
                    process(pre_cause, pre_state)



    # Gather timepoints for convenient iteration (reverse order)
    timepoints = list(iter(curr_t))

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
        print("Cache size: %d" % sum(len(c.cache) for t in timepoints for c in t.causes))

        if answer is not None:
            print("Correct: %s" % (best_path[timepoints[-1]] == answer))
        print()

        # Label the FSM states and timepoints
        print("Caches")
        for t in timepoints:
            for c in t.causes:
                print(str(c), [(str(s), str(o)) for s,o in c.cache.items()])
        print()

        print("Timepoints")
        for t in timepoints:
            print(str(t), [str(v) for v in t.causes])
        print()

        print("FSM")
        for st in iter(fsm):
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
