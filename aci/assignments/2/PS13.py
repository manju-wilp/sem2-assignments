# Assignment 2 - PS13 : Bayesian Network based Medical Diagnosis Assistant
# Group : G093
#
# The program reads a plain text input file that lists the conditional
# probability values for two small diagnosis networks, builds a generic
# Bayesian Network for each scenario and answers the required probabilistic
# queries using exact inference by full joint enumeration.
#
# The inference engine (class BayesianNetwork) is written once and reused for
# both scenarios, so no query result is hard coded. Only the network wiring
# (which node depends on which parent) is described per scenario.

import sys
from itertools import product

# Tolerance used when comparing two probabilities for the independence checks.
EPSILON = 1e-9

# Number of decimal places used in the printed results.
PRECISION = 4

# Required probability keys for every supported scenario. The order of the
# list is the order in which the keys are validated.
REQUIRED_KEYS = {
    "SCENARIO_1_FLU": [
        "P_F", "P_C_given_F", "P_C_given_notF",
        "P_V_given_F", "P_V_given_notF",
    ],
    "SCENARIO_2_CRITICAL_WARD": [
        "P_E", "P_A1_given_E", "P_A1_given_notE",
        "P_A2_given_E", "P_A2_given_notE",
    ],
}


class BayesianNetworkError(Exception):
    """Raised for any problem found while reading, validating or building a network."""
    pass


class Node:
    """A single binary random variable together with its conditional table."""

    def __init__(self, name, parents, cpt):
        self.name = name          # variable name, for example "F"
        self.parents = parents    # list of parent variable names
        self.cpt = cpt            # dict: tuple(parent values) -> P(node = True)

    def probability(self, value, parent_values):
        """Return P(node = value | parents = parent_values)."""
        key = tuple(parent_values)
        if key not in self.cpt:
            raise BayesianNetworkError(
                "Missing CPT entry for node '%s' with parent values %s"
                % (self.name, key))
        prob_true = self.cpt[key]
        return prob_true if value else (1.0 - prob_true)


class BayesianNetwork:
    """A binary Bayesian Network with exact inference by joint enumeration."""

    def __init__(self):
        self.nodes = []      # nodes kept in the order they are added (topological)
        self.index = {}      # variable name -> Node

    def add_node(self, name, parents, cpt):
        """Insert a new variable. Rejects a variable that already exists."""
        if name in self.index:
            raise BayesianNetworkError("Variable '%s' added more than once" % name)
        for parent in parents:
            if parent not in self.index:
                raise BayesianNetworkError(
                    "Parent '%s' of node '%s' must be added first" % (parent, name))
        node = Node(name, parents, cpt)
        self.nodes.append(node)
        self.index[name] = node

    def variables(self):
        """Return the variable names in topological order."""
        return [node.name for node in self.nodes]

    def joint_probability(self, assignment):
        """Probability of one complete assignment using the chain rule factorization."""
        result = 1.0
        for node in self.nodes:
            parent_values = [assignment[name] for name in node.parents]
            result *= node.probability(assignment[node.name], parent_values)
        return result

    def enumerate_all(self):
        """Yield every complete assignment with its joint probability."""
        names = self.variables()
        for combo in product([True, False], repeat=len(names)):
            assignment = dict(zip(names, combo))
            yield assignment, self.joint_probability(assignment)

    def probability_of(self, evidence):
        """Return P(evidence) by summing the joint over all consistent assignments."""
        total = 0.0
        for assignment, prob in self.enumerate_all():
            if all(assignment[var] == val for var, val in evidence.items()):
                total += prob
        return total

    def conditional(self, query, evidence):
        """Return P(query | evidence) computed as P(query and evidence) / P(evidence)."""
        combined = dict(evidence)
        combined.update(query)
        denominator = self.probability_of(evidence)
        if denominator == 0.0:
            raise BayesianNetworkError(
                "Cannot condition on an event with zero probability")
        return self.probability_of(combined) / denominator

    def marginally_independent(self, first, second):
        """True when P(a, b) equals P(a) * P(b) for every value combination."""
        for value_a, value_b in product([True, False], repeat=2):
            joint = self.probability_of({first: value_a, second: value_b})
            separate = (self.probability_of({first: value_a})
                        * self.probability_of({second: value_b}))
            if abs(joint - separate) > EPSILON:
                return False
        return True

    def conditionally_independent(self, first, second, given):
        """True when P(a, b | g) equals P(a | g) * P(b | g) for every combination."""
        for value_g in [True, False]:
            # Skip a value of the conditioning variable that never occurs.
            if self.probability_of({given: value_g}) == 0.0:
                continue
            for value_a, value_b in product([True, False], repeat=2):
                joint = self.conditional(
                    {first: value_a, second: value_b}, {given: value_g})
                separate = (self.conditional({first: value_a}, {given: value_g})
                            * self.conditional({second: value_b}, {given: value_g}))
                if abs(joint - separate) > EPSILON:
                    return False
        return True


def parse_input(path):
    """
    Read the input file and return an ordered list of
    (scenario_name, {key: value}) tuples.

    Raises BayesianNetworkError on an unknown scenario, a duplicate scenario,
    a duplicate key, a malformed line or a probability outside [0, 1].
    """
    try:
        with open(path, "r") as handle:
            raw_lines = handle.readlines()
    except OSError as exc:
        raise BayesianNetworkError("Unable to open input file '%s': %s" % (path, exc))

    scenarios = []
    seen_scenarios = set()
    current_name = None
    current_data = None

    for number, original in enumerate(raw_lines, start=1):
        line = original.strip()
        if not line:
            continue  # blank lines carry no information

        # A scenario header starts a new block of probability values.
        if line.startswith("SCENARIO_"):
            if line not in REQUIRED_KEYS:
                raise BayesianNetworkError(
                    "Line %d: unknown scenario identifier '%s'" % (number, line))
            if line in seen_scenarios:
                raise BayesianNetworkError(
                    "Line %d: scenario '%s' is defined more than once"
                    % (number, line))
            if current_name is not None:
                scenarios.append((current_name, current_data))
            current_name = line
            current_data = {}
            seen_scenarios.add(line)
            continue

        # Any non header line must have the form key=value.
        if "=" not in line:
            # A plain header such as the problem id is tolerated only before
            # the first scenario block. Anywhere else it is malformed.
            if current_name is None:
                continue
            raise BayesianNetworkError(
                "Line %d: malformed line '%s' (expected key=value)"
                % (number, original.rstrip()))

        key, _, value_text = line.partition("=")
        key = key.strip()
        value_text = value_text.strip()

        if current_name is None:
            raise BayesianNetworkError(
                "Line %d: key '%s' found before any scenario header"
                % (number, key))
        if not key:
            raise BayesianNetworkError(
                "Line %d: malformed line '%s' (empty key)"
                % (number, original.rstrip()))
        if key in current_data:
            raise BayesianNetworkError(
                "Line %d: duplicate key '%s' in scenario '%s'"
                % (number, key, current_name))
        try:
            value = float(value_text)
        except ValueError:
            raise BayesianNetworkError(
                "Line %d: value of '%s' is not a valid number ('%s')"
                % (number, key, value_text))
        if not 0.0 <= value <= 1.0:
            raise BayesianNetworkError(
                "Line %d: probability '%s' = %s is outside the range [0, 1]"
                % (number, key, value_text))
        current_data[key] = value

    if current_name is not None:
        scenarios.append((current_name, current_data))

    if not scenarios:
        raise BayesianNetworkError("No valid scenario block found in the input file")

    return scenarios


def validate_scenario(name, data):
    """Confirm that a scenario has exactly the keys it needs, no more and no less."""
    required = REQUIRED_KEYS[name]
    for key in required:
        if key not in data:
            raise BayesianNetworkError(
                "Scenario '%s' is missing required key '%s'" % (name, key))
    for key in data:
        if key not in required:
            raise BayesianNetworkError(
                "Scenario '%s' has an unexpected key '%s'" % (name, key))


def build_flu_network(data):
    """Wire the Flu network: F -> C and F -> V."""
    network = BayesianNetwork()
    network.add_node("F", [], {(): data["P_F"]})
    network.add_node("C", ["F"], {
        (True,): data["P_C_given_F"],
        (False,): data["P_C_given_notF"],
    })
    network.add_node("V", ["F"], {
        (True,): data["P_V_given_F"],
        (False,): data["P_V_given_notF"],
    })
    return network


def build_ward_network(data):
    """Wire the Critical Ward network: E -> A1 and E -> A2."""
    network = BayesianNetwork()
    network.add_node("E", [], {(): data["P_E"]})
    network.add_node("A1", ["E"], {
        (True,): data["P_A1_given_E"],
        (False,): data["P_A1_given_notE"],
    })
    network.add_node("A2", ["E"], {
        (True,): data["P_A2_given_E"],
        (False,): data["P_A2_given_notE"],
    })
    return network


def flag(value):
    """Convert a boolean into the single letter used in the joint table."""
    return "T" if value else "F"


def report_flu(network):
    """Produce the output block for the Flu scenario, including the joint table."""
    lines = []
    lines.append("Scenario 1 - Joint probability table")
    lines.append("F C V Probability")
    for assignment, prob in network.enumerate_all():
        lines.append("%s %s %s %.*f" % (
            flag(assignment["F"]), flag(assignment["C"]), flag(assignment["V"]),
            PRECISION, prob))

    lines.append("Scenario 1: Flu Diagnosis Network")
    lines.append("P(F | C) = %.*f"
                 % (PRECISION, network.conditional({"F": True}, {"C": True})))
    lines.append("P(F | V) = %.*f"
                 % (PRECISION, network.conditional({"F": True}, {"V": True})))
    lines.append("P(F | C and V) = %.*f"
                 % (PRECISION, network.conditional({"F": True}, {"C": True, "V": True})))

    marginal = network.marginally_independent("C", "V")
    lines.append("C and V are %s without evidence."
                 % ("independent" if marginal else "not independent"))
    conditional = network.conditionally_independent("C", "V", "F")
    lines.append("C and V are %sconditionally independent given F."
                 % ("" if conditional else "not "))
    return lines


def report_ward(network):
    """Produce the output block for the Critical Ward scenario."""
    lines = []
    lines.append("Scenario 2: Critical Ward Alert System")
    lines.append("P(E | A1) = %.*f"
                 % (PRECISION, network.conditional({"E": True}, {"A1": True})))
    lines.append("P(E | A2) = %.*f"
                 % (PRECISION, network.conditional({"E": True}, {"A2": True})))
    lines.append("P(E | A1 and A2) = %.*f"
                 % (PRECISION, network.conditional({"E": True}, {"A1": True, "A2": True})))

    marginal = network.marginally_independent("A1", "A2")
    lines.append("A1 and A2 are %s without evidence."
                 % ("independent" if marginal else "not independent"))
    conditional = network.conditionally_independent("A1", "A2", "E")
    lines.append("A1 and A2 are %sconditionally independent given EmergencyCondition."
                 % ("" if conditional else "not "))
    return lines


# Maps each scenario to the pair of functions that build and report it.
SCENARIO_HANDLERS = {
    "SCENARIO_1_FLU": (build_flu_network, report_flu),
    "SCENARIO_2_CRITICAL_WARD": (build_ward_network, report_ward),
}


def solve(input_path):
    """Read the input, run every scenario and return the list of output lines."""
    scenarios = parse_input(input_path)
    output_lines = []
    for name, data in scenarios:
        validate_scenario(name, data)
        builder, reporter = SCENARIO_HANDLERS[name]
        network = builder(data)
        output_lines.extend(reporter(network))
    return output_lines


def main(argv):
    """Entry point. Usage: python PS13.py [inputPS13.txt] [outputPS13.txt]."""
    input_path = argv[1] if len(argv) > 1 else "inputPS13.txt"
    output_path = argv[2] if len(argv) > 2 else "outputPS13.txt"

    try:
        output_lines = solve(input_path)
    except BayesianNetworkError as exc:
        output_lines = ["Error: %s" % exc]

    text = "\n".join(output_lines)
    print(text)

    try:
        with open(output_path, "w") as handle:
            handle.write(text + "\n")
    except OSError as exc:
        print("Warning: could not write output file '%s': %s" % (output_path, exc))


if __name__ == "__main__":
    main(sys.argv)


# The helper below was used during development to double check the numbers
# against the sample output. It is left commented out as required.
#
# def _debug_print(network):
#     for assignment, prob in network.enumerate_all():
#         print(assignment, round(prob, 4))
