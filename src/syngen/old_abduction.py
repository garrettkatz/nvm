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

    Because Cause caches cannot have repeat FSM states, and FSM states only have
        one parent, the inner-most loops can be combined. The total number of
        constructed Causes constructed during iteration over a given cache is
        bounded by the number of cause-effect relations in the knowledge base.
        Because Cause construction requires tracing back through the FSM, the
        total number of operations is the cumulative length of effect sequences
        in the knowledge base:

    Worst case:
    O = len(action_sequence)
            * len(causes)
                * len(causes)
                    * (len(FSM)
                        + sum(len(effects) for (cause,effects) in knowledge))


    Identify bounds for any given timepoint based on the knowledge base only:
        1. The number of causes constructed
        2. The cumulative cache length for constructed Causes
        3. The cumulative length of effect sequences in the cause-effect pairs
            used for construction of Causes

    For the following analyses, the example knowledge base will be:
        X : A->B->B
        X : A->B
        X : B
        Y : X

    1. Cause construction

    Define set(suffix) as the set of causal action decompositions that share a
        common |suffix|.

    set(B):
        X : A->B->B
        X : A->B
        X : B
        Y : X : A->B->B
        Y : X : A->B
        Y : X : B

    Define C as the length of the largest set(suffix) for all possible suffixes
        in the knowledge base. C is the maximum number of Causes that can be
        constructed during any given timepoint. It is evident that at least one
        of the suffixes that satisfy this condition will be of length 1. Thus,
        C can be computed by looping over all primitive actions and recursively
        decomposing cause-effect relations.

    2. Cumulative cache length

    Define covers(sequence) as the complete set of covers for a given sequence.
    Define paths(covers(sequence)) as the set FSM states with a path from the
        initial state that is contained in covers(sequence).

    Define S as the length of the largest paths(covers(subsequence)) for all
        possible effects subsequences in the knowledge base. S is the maximum
        number of cached states at any given time point.  This definition is
        tautological because the abduction algorithm itself is necessary to
        compute it, but it can be computed given only the knowledge base.

    An alternative, looser bound uses set(suffix):
    Define T(causal_identity) as the number of transitions in the FSM associated
        with a causal identity.
    Define T(set(suffix)) as the sum of T(causal_identity) for the causal
        identies of all elements of set(suffix).
    Define S as the largest T(set(suffix)) for all possible suffixes in the
        knowledge base. As with C, it is evident that S will be associated with
        a suffix of length 1, and it can be computed in a similar fashion to S.
    This works by considering possible sets of causes evoked during a timestep,
        and all possible transitions that they could be effectively associated
        with. In practice, the actual number of possible transitions depends on
        the cache from the prior timestep, but using this information to bound
        the current cache size requires the previously outlined method of
        computing possible covers.

    3. Cumulative length of evoked effects sequences

    Define effects(set(suffix)) as the list of all of effects sequences
        associated with the action decompositions in set(suffix).

    effects(set(B)):
        A->B->B
        A->B
        B
        X
        X
        X

    Define len(effects(set(suffix))) as the cumulative length of effects
        sequences in effects(set(suffix)).
    Define E as the maximum len(effects(set(suffix))) for all possible suffixes
        in the knowledge base. E is the maximum number of tracebacks in the FSM
        that will occur at any given timestep.



    Given these definitions of C, S, and E:

    (C * S) is the total number of transitions in the FSM that can be attempted
        in a given timestep. This correspond to each pair (state,cause), where
        |state| is cached in a cause evoked at time t-1, and |cause| is a cause
        evoked at time t.
    Regardless of the number of attempted transitions, the number of tracebacks
        is bounded by E. For this reason, the computational complexity of each
        timestep is ((C * S) + E)

    Therefore:
        O = len(action_sequence) * ((C * S) + E)


    In the restricted case where:
        1. Effects do not need to be computed when Causes are evoked. This means
            that Cause construction is O(1), but requires that start times be
            cached along with active states
        2. Each cause-effect relation in the knowledge base is used at most once
            during a timestep
    Define max_T = sum(T(cause.identity) for (cause,effects) in knowledge)
    This is the sum of transition counts for the causal identities of each
        cause-effect relation in the knowledge base

    The complexity of the algorithm can be easily bounded by
        O = len(action_sequence) * len(knowledge) * max_T


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
