from queue import Queue
from random import choice

"""
Represents a cause-effect relation in the knowledge base.
  |cause| is an intention that results in |effects|
"""
class Relation:
    def __init__(self, cause, effects):
        self.cause = cause
        self.effects = effects

        # TODO: argument validation/extraction

"""
Knowledge base of cause-effect relations.
  |knowledge| is a list of cause-effect tuples
"""
class KnowledgeBase:
    def __init__(self, knowledge):
        self.relations = { effects[0] : [] for cause,effects in knowledge}

        for cause,effects in knowledge:
            self.relations[effects[0]].append(Relation(cause, effects))

    def evoke(self, effect):
        return self.relations.get(effect, [])

"""
Represents an identified possible cause.
  |identity| is the causal identity
  |start_time| indicates the timestep before the first effect
  |end_time| indicates the last timestep of the last effect
  |hypothesis| provides a link back to the observational support for the cause
    if the cause is self-evidential (action causes itself), |hypothesis| is null
"""
class Cause:
    counter = 0

    def __init__(self, identity, start_time, end_time, args=None):
        self.index = Cause.counter
        Cause.counter += 1

        self.identity = identity
        self.start_time = start_time
        self.end_time = end_time
        self.args = args

    def __str__(self):
        return "c%d" % self.index

"""
Represents a hypothesis that a cause-effect |relation| explains a sequence of
  effects starting at the given |start_time|.

Hypotheses keep track of the observed effects that correspond to the effects
  list in the relation.
"""
class Hypothesis:
    counter = 0

    def __init__(self, relation, effect):
        self.index = Hypothesis.counter
        Hypothesis.counter += 1

        self.relation = relation
        self.effects = [effect]
        self.args = None

    def complete(self):
        return len(self.effects) == len(self.relation.effects)

    def extend(self, effect):
        if self.complete():
            raise Exception

        if effect.identity == self.relation.effects[len(self.effects)]:
            self.effects.append(effect)
            # TODO: update arguments based on effect
            return True
        else:
            return False

    def gen_cause(self):
        if not self.complete():
            raise Exception

        return Cause(self.relation.cause,
                        self.effects[0].start_time,
                        self.effects[-1].end_time,
                        self.args)

    def __str__(self):
        return "h%d" % self.index

"""
Timepoint nodes that are chained into a reverse linked list.

Each node contains:
    * pointer to |previous| timepoint
    * set of |causes| that end at this timepoint
"""
class Timepoint:
    counter = 0

    def __init__(self, previous=None):
        self.index = Timepoint.counter
        Timepoint.counter += 1

        self.previous = previous
        self.hypotheses = set()
        self.causes = set()

    """ Spawns a new timepoint and adds a pointer to the current timepoint.  """
    def incr(self):
        return Timepoint(self)

    def get_hypotheses(self):
        return set(self.hypotheses)

    def add_hypothesis(self, to_add):
        self.hypotheses.add(to_add)

    def remove_hypothesis(self, to_remove):
        self.hypotheses.remove(to_remove)

    def get_complete_hypotheses(self):
        return set(hyp for hyp in self.hypotheses if hyp.complete())

    def add_cause(self, cause):
        self.causes.add(cause)

    """ Recursive iterator. """
    def __iter__(self):
        yield self
        if self.previous is not None:
            yield from self.previous

    def __str__(self):
        return "t%d" % self.index



def abduce(kb, input_sequence):
    # Current time
    curr_t = Timepoint()

    # For each action in the input sequence...
    for a in input_sequence:
        # Increment timestep
        curr_t = curr_t.incr()

        # Create a queue of action/causes to process
        # Initialize with self-cause for observed action
        q = Queue()
        q.put(Cause(a, curr_t.previous, curr_t))

        # Run until the queue is empty
        while not q.empty():
            # Retrieve action/cause (v) and add it to the graph
            v = q.get()
            curr_t.add_cause(v)

            # Evoke new hypotheses in current time
            for relation in kb.evoke(v.identity):
                hyp = Hypothesis(relation, v)
                if hyp.complete(): q.put(hyp.gen_cause())
                else:              curr_t.add_hypothesis(hyp)

            # Extend hypotheses to current time if possible
            for hyp in v.start_time.get_hypotheses():
                if hyp.extend(v):
                    v.start_time.remove_hypothesis(hyp)
                    if hyp.complete(): q.put(hyp.gen_cause())
                    else:              curr_t.add_hypothesis(hyp)


    # Gather timepoints for convenient iteration (reverse order)
    timepoints = list(iter(curr_t))

    # Use dynamic programming to compute shortest paths from the end to start
    shortest_edge =   { t : None  for t in timepoints }
    shortest_length = { t : 99999 for t in timepoints }
    shortest_length[curr_t] = 0

    for t in timepoints:
        length = 1 + shortest_length[t]

        for cause in t.causes:
            if length < shortest_length[cause.start_time]:
                shortest_length[cause.start_time] = length
                shortest_edge[cause.start_time] = cause

    # Compute the shortest path from end to each node
    shortest_path = { curr_t : "" }
    for t in timepoints[1:]:
        cause = shortest_edge[t]
        shortest_path[t] = shortest_path[cause.end_time] + cause.identity

    return timepoints, shortest_path


def run(kb, seq, answer=None, verbose=False):
    timepoints, best_path = abduce(kb, seq)

    if verbose:
        print("Sequence: " + seq)

        for t in timepoints:
            print("%4s : %s" % (str(t), best_path[t]))

        print("Shortest path: %d" % len(best_path[timepoints[-1]]))
        print("Num causes: %d" % sum(len(t.causes) for t in timepoints))

        if answer is not None:
            print("Correct: %s" % (best_path[timepoints[-1]] == answer))
        print()

        print("Timepoints")
        for t in timepoints:
            print(str(t))
            print("  Hypoth:", [str(h) for h in t.hypotheses])
            print("  Causes:", [str(v) for v in t.causes])
        print()

        print("Knowledge Base")
        for k,v in kb.relations.items():
            for r in iter(v):
                print(r.cause, r.effects)
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
        run(KnowledgeBase(knowledge), seq, answer, verbose=True)
