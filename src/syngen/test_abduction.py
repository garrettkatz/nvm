from queue import Queue
from random import choice
from itertools import chain

"""
State in Finite State Machine.
If state is terminal, len(causes) > 0.
If state is one action away from initial state, 'action' is not None.
"""
class FSMState:
    def __init__(self, index=0, action=None):
        self.index = index
        self.action = action
        self.causes = []

    """ Append a cause to the state. """
    def add_cause(self, cause):
        self.causes.append(cause)

    """
    Returns an arbitrary cause associated with the FSM state, or
      an action if no higher level causes are associated with this state.
    """
    def get_cause_or_action(self):
        return self.causes[0] if len(self.causes) > 0 else self.action

    def __str__(self):
        return "s%d" % self.index


"""
Finite State Machine encoding causal knowledge.

Constructor takes a list of (cause, effect) tuples encoding knowledge,
  and a list of primitive actions which cause/explain themselves.
    fsm = FSM(
        [
            ('X', 'ABC'),   # X causes A,B,C
            ('Y', 'AB'),    # Y causes A,B
            ('Z', 'XY'),    # Z causes X,Y
            ...
        ],
        'ABC'               # Primitive actions
    )

To run the machine:
    state = fsm.init
    for v in 'ABCDE':
        state = fsm.advance(state, inp)
        print(state.causes)
"""
class FSM:
    def __init__(self, knowledge, actions):
        self.init = FSMState(index=0)
        self.trans = { self.init : {} }

        # Add states for each action (actions explain themselves)
        for action in actions:
            new_state = FSMState(index=len(self.trans), action=action)
            self.trans[self.init][action] = new_state
            self.trans[new_state] = {}

        # For each cause-effect pair...
        for cause,effect in knowledge:
            # Start at initial state
            state = self.init

            # Transition through the machine, adding states as necessary
            for v in effect:
                new_state = self.advance(state, v)

                # If transition fails, add new state
                if new_state is None:
                    new_state = FSMState(index=len(self.trans))
                    self.trans[state][v] = new_state
                    self.trans[new_state] = {}
                state = new_state

            # Add cause to terminal state
            state.add_cause(cause)

    """
    Advance the machine from a starting state using transition 'inp'.
    Invalid transitions will cause advance() to return None.
    """
    def advance(self, state, inp):
        return self.trans[state].get(inp, None)


"""
Timepoint nodes that are chained into a linked list.

Each node contains:
    * time value
    * pointer to next timepoint
    * cache of active FSM states
        - each active state keeps track of the
              timepoint when its traversal began
    * set of edges linking timepoints via FSM states (with actions/causes)
        - each edge corresponds to the start and end times of a cause
"""
class Timepoint:
    def __init__(self, time=0):
        self.time = time
        self.nxt = None
        self.cache = {}
        self.edges = {}

    """
    Caches an FSM state.
    Each cached state is stored with the timestep when its traversal began.
    This way, when a terminal state is reached in the FSM, the algorithm can
      determine when the associated cause began, and extend the cached states
      from that timepoint.
    """
    def cache_state(self, fsm_state, start_t):
        # TODO:
        # The same FSM state may be cached with different starting timepoints.
        #   This must depend on properties of the FSM (to be determined).
        # Conflict can be detected in the NVM.
        # How should this be handled?
        if fsm_state in self.cache:
            pass

        self.cache[fsm_state] = start_t

    """
    Adds an edge to the causal graph.
    The edge contains an FSM state representing an action or cause that
      began immediately after the current timepoint and ended just before
      the provided end timepoint.
    """
    def add_edge(self, fsm_state, end_t):
        # TODO:
        # Multiple edges may link two timepoints.
        # Conflict can be detected in the NVM.
        # Because compatibility is determined using cached states, edges are
        #   only needed for computation of the shortest path, which may be done
        #   periodically during execution to free up memory (trim the graph).
        # For this reason, it may be arbitrary which state is retained.
        # How should this be handled?
        if end_t in self.edges:
            pass

        self.edges[end_t] = fsm_state

    """ Spawns a new timepoint and adds a pointer to the current timepoint.  """
    def incr(self):
        self.nxt = Timepoint(self.time+1)
        return self.nxt

    def __str__(self):
        return "t%d" % self.time

"""
Perform cause-effect reasoning (abduction) using an input sequence of actions
  and a Finite State Machine encoding causal knowledge.

Returns a list of the shortest paths terminating at each time point.

Variables for graph generation:
    * start_time : starting timepoint
    * t          : current timepoint
    * q          : action/cause queue
    * a          : observed action
    * v          : cause (may be 'a')
    * v_t        : timepoint of cause/action 'v'
    * state      : cached fsm_state
    * path_t     : starting timepoint of fsm path
    * new_state  : restulting fsm_state after transition
    * cause      : cause associated with terminal fsm state

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
    # Keep track of the starting timepoint
    start_time = Timepoint()

    # Current time
    t = start_time

    # For each action in the input sequence...
    for a in input_sequence:
        # Create a queue of action/causes to process
        # Initialize with observed action (start timepoint = previous t)
        q = Queue()
        q.put((a, t))

        # Increment timestep
        t = t.incr()

        # Run until the queue is empty
        while not q.empty():
            # Retrieve action/cause (v) and its start timepoint from queue (v_t)
            v,v_t = q.get()

            # Advance cached FSM states from v's start timepoint (+initial)
            # Cache the resulting states in the current timepoint
            for state,path_t in chain(v_t.cache.items(), [(fsm.init, v_t)]):
                new_state = fsm.advance(state, v)

                # If the FSM advancement was successful...
                if new_state is not None:
                    # Cache the new state with the same starting timepoint
                    t.cache_state(new_state, path_t)

                    # If the state is terminal or an action state, add it to the
                    #   graph linking path start timepoint to current timepoint
                    if new_state.get_cause_or_action() is not None:
                        path_t.add_edge(new_state,t)

                    # If the state is terminal, enqueue all of its causes with
                    #   the path start timepoint. Because the queue was
                    #   initialized with the observed action, only enqueue
                    #   high level causes.
                    for cause in new_state.causes:
                        q.put((cause, path_t))


    # Gather timepoints for convenient iteration
    timepoints = []
    t = start_time
    while t is not None:
        timepoints.append(t)
        t = t.nxt

    # Use dynamic programming to compute the shortest paths from the start
    best_length, best_edge, best_path = {}, {}, {}

    # Initialize data structures
    for t in timepoints:
        best_length[t] = 99999
        best_edge[t] = None
        best_path[t] = ""
    best_length[start_time] = 0

    # Iterate through the timepoints
    for t in timepoints:
        length = 1 + best_length[t]

        # Iterate through the outgoing edges (action/cause and end timepoint)
        #   and update best length and incoming edge for each end timepoint
        for end,state in t.edges.items():
            if length < best_length[end]:
                best_length[end] = length
                best_edge[end] = (state,t)

    # Compute the shortest path for each node
    # If the linking state has no causes, it is an action state
    for t in timepoints[1:]:
        state,prev_t = best_edge[t]
        best_path[t] = best_path[prev_t] + state.get_cause_or_action()

    return timepoints, best_path


def run(fsm, seq, answer=None, verbose=False):
    timepoints, best_path = abduce(fsm, seq)

    if verbose:
        print("Sequence: " + seq)

        for t in timepoints:
            print(t.time, best_path[t])

        print("Shortest path: %d" % len(best_path[timepoints[-1]]))
        print("Num edges: %d" % sum(len(t.edges) for t in timepoints))
        print("Cache size: %d" % sum(len(t.cache) for t in timepoints))

        if answer is not None:
            print("Correct: %s" % (best_path[timepoints[-1]] == answer))
        print()

        # Label the FSM states and timepoints
        print("Caches")
        for t in timepoints:
            print(str(t), [(str(s), str(o)) for s,o in t.cache.items()])
        print()

        print("Edges")
        for t in timepoints:
            print(str(t), [(str(o), str(s)) for o,s in t.edges.items()])
        print()

        print("FSM")
        for st,trans in fsm.trans.items():
            print(str(st), [(inp, str(st2)) for inp,st2 in trans.items()])
        print()

    return best_path[timepoints[-1]]


if __name__ == "__main__":
    run(
        FSM(
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
            actions="ABCDE"
        ),
        seq='ABCDE',
        answer='G',
        verbose=True)


    run(
        FSM(
            [
                ('Y', 'AB'),
                ('W', 'YC'),
                ('G', 'WU'),
                ('K', 'FG'),
            ],
            actions="ABCFU"
        ),
        seq='FABCU',
        answer='K',
        verbose=True)


    run(
        FSM(
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
            actions="ABC"
        ),
        seq="".join([choice('ABC') for _ in range(20)]),
        verbose=True)
