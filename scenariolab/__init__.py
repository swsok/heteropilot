"""ScenarioLab: random-scenario power-optimal placement experiments.

DESIGN_scenariolab.md is the authoritative design. ScenarioLab is a consumer
of the planner library: it generates random ClusterSpecV2 / ServiceSpec
instances, runs the existing candidate-generation and optimization pipeline
behind a fast (surrogate) prediction path, and stores every result with its
fidelity and provenance labels. It never modifies planner or upstream code.
"""

__version__ = "0.1.0"
