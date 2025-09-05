---
algorithm_key: "partial-order-planning"
tags: [planning, algorithms, partial-order-planning, pop, least-commitment, planning-graphs]
title: "Partial Order Planning (POP)"
family: "planning"
---

# Partial Order Planning (POP)

{{ algorithm_card("partial-order-planning") }}

!!! abstract "Overview"
    Partial Order Planning (POP) is a classical planning algorithm that constructs plans with minimal ordering constraints between actions. Unlike total-order planners that fix a complete sequence of actions, POP maintains a partial order, allowing for more flexible plan construction and execution.

    The algorithm uses a "least commitment" strategy, only adding ordering constraints when necessary to resolve threats (conflicts between actions). This approach often leads to more efficient plans and better handling of parallel execution. POP is particularly useful for domains where the exact order of actions is not critical.

## Mathematical Formulation

!!! math "Partial Order Plan Structure"
    A partial order plan $P = (A, O, L)$ consists of:

    - **Actions** $A$: Set of actions in the plan
    - **Ordering constraints** $O$: Partial order on actions ($a_i < a_j$ means $a_i$ must precede $a_j$)
    - **Causal links** $L$: Set of causal links $(a_i, p, a_j)$ where action $a_i$ achieves proposition $p$ for action $a_j$

    A plan is valid if:

    $$\forall (a_i, p, a_j) \in L : p \in \text{effects}(a_i) \cap \text{preconditions}(a_j)$$

    $$\forall a_k \in A : \text{threats}(a_k, (a_i, p, a_j)) \Rightarrow (a_k < a_i \lor a_j < a_k)$$

    Where threats are actions that could interfere with causal links.

!!! success "Key Properties"
    - **Least Commitment**: Only adds constraints when necessary
    - **Flexible**: Allows multiple valid execution orders
    - **Parallel-friendly**: Naturally supports concurrent action execution
    - **Threat Resolution**: Systematically handles action conflicts

## Implementation Approaches

=== "Basic POP Implementation (Recommended)"
    ```python
    from typing import Set, List, Dict, Tuple, Optional
    from dataclasses import dataclass
    from collections import defaultdict
    import itertools

    @dataclass
    class Proposition:
        """Represents a proposition in the planning domain."""
        name: str
        args: Tuple[str, ...] = ()

        def __hash__(self):
            return hash((self.name, self.args))

        def __eq__(self, other):
            return isinstance(other, Proposition) and self.name == other.name and self.args == other.args

    @dataclass
    class Action:
        """Represents an action in the planning domain."""
        name: str
        preconditions: Set[Proposition]
        effects: Set[Proposition]
        args: Tuple[str, ...] = ()

        def __hash__(self):
            return hash((self.name, self.args))

    @dataclass
    class CausalLink:
        """Represents a causal link between actions."""
        producer: Action
        proposition: Proposition
        consumer: Action

        def __hash__(self):
            return hash((self.producer, self.proposition, self.consumer))

    @dataclass
    class PartialOrderPlan:
        """Represents a partial order plan."""
        actions: Set[Action]
        ordering_constraints: Set[Tuple[Action, Action]]  # (before, after)
        causal_links: Set[CausalLink]
        start_action: Action
        goal_action: Action

        def __init__(self, start_action: Action, goal_action: Action):
            self.actions = {start_action, goal_action}
            self.ordering_constraints = set()
            self.causal_links = set()
            self.start_action = start_action
            self.goal_action = goal_action

    class PartialOrderPlanner:
        """
        Partial Order Planning algorithm implementation.

        Args:
            actions: List of available actions
            initial_state: Set of initial propositions
            goal: Set of goal propositions
        """

        def __init__(self, actions: List[Action], initial_state: Set[Proposition], goal: Set[Proposition]):
            self.actions = actions
            self.initial_state = initial_state
            self.goal = goal

        def plan(self) -> Optional[PartialOrderPlan]:
            """
            Find a partial order plan.

            Returns:
                Partial order plan, or None if no plan exists
            """
            # Create start and goal actions
            start_action = Action("START", set(), self.initial_state)
            goal_action = Action("GOAL", self.goal, set())

            # Initialize plan
            plan = PartialOrderPlan(start_action, goal_action)

            # Add initial causal links for goal propositions
            for goal_prop in self.goal:
                plan.causal_links.add(CausalLink(start_action, goal_prop, goal_action))

            # Search for a valid plan
            return self._search_plan(plan)

        def _search_plan(self, plan: PartialOrderPlan) -> Optional[PartialOrderPlan]:
            """Search for a valid partial order plan."""
            # Check if plan is complete
            if self._is_complete(plan):
                return plan

            # Find open conditions (unsupported preconditions)
            open_conditions = self._find_open_conditions(plan)

            for action, precondition in open_conditions:
                # Try to support this precondition
                supporters = self._find_supporters(precondition, plan)

                for supporter in supporters:
                    # Create new plan with this supporter
                    new_plan = self._add_action_to_plan(plan, supporter, action, precondition)

                    # Resolve threats
                    if self._resolve_threats(new_plan):
                        # Recursively search
                        result = self._search_plan(new_plan)
                        if result:
                            return result

            return None

        def _is_complete(self, plan: PartialOrderPlan) -> bool:
            """Check if the plan is complete (all preconditions supported)."""
            for action in plan.actions:
                for precondition in action.preconditions:
                    # Check if precondition is supported by a causal link
                    supported = False
                    for link in plan.causal_links:
                        if link.consumer == action and link.proposition == precondition:
                            supported = True
                            break

                    if not supported:
                        return False

            return True

        def _find_open_conditions(self, plan: PartialOrderPlan) -> List[Tuple[Action, Proposition]]:
            """Find preconditions that are not yet supported."""
            open_conditions = []

            for action in plan.actions:
                for precondition in action.preconditions:
                    # Check if precondition is supported
                    supported = False
                    for link in plan.causal_links:
                        if link.consumer == action and link.proposition == precondition:
                            supported = True
                            break

                    if not supported:
                        open_conditions.append((action, precondition))

            return open_conditions

        def _find_supporters(self, proposition: Proposition, plan: PartialOrderPlan) -> List[Action]:
            """Find actions that can support a given proposition."""
            supporters = []

            # Check existing actions in plan
            for action in plan.actions:
                if proposition in action.effects:
                    supporters.append(action)

            # Check available actions not in plan
            for action in self.actions:
                if action not in plan.actions and proposition in action.effects:
                    supporters.append(action)

            return supporters

        def _add_action_to_plan(self, plan: PartialOrderPlan, supporter: Action,
                              consumer: Action, proposition: Proposition) -> PartialOrderPlan:
            """Add an action to the plan and create necessary causal links."""
            new_plan = PartialOrderPlan(plan.start_action, plan.goal_action)
            new_plan.actions = plan.actions.copy()
            new_plan.ordering_constraints = plan.ordering_constraints.copy()
            new_plan.causal_links = plan.causal_links.copy()

            # Add supporter if not already in plan
            if supporter not in new_plan.actions:
                new_plan.actions.add(supporter)

                # Add ordering constraints
                new_plan.ordering_constraints.add((plan.start_action, supporter))
                new_plan.ordering_constraints.add((supporter, plan.goal_action))

            # Add causal link
            new_plan.causal_links.add(CausalLink(supporter, proposition, consumer))

            # Add ordering constraint
            new_plan.ordering_constraints.add((supporter, consumer))

            return new_plan

        def _resolve_threats(self, plan: PartialOrderPlan) -> bool:
            """Resolve all threats in the plan."""
            threats = self._find_threats(plan)

            for threat_action, causal_link in threats:
                # Try to resolve threat by promotion or demotion
                if self._can_promote(threat_action, causal_link, plan):
                    plan.ordering_constraints.add((causal_link.consumer, threat_action))
                elif self._can_demote(threat_action, causal_link, plan):
                    plan.ordering_constraints.add((threat_action, causal_link.producer))
                else:
                    return False  # Cannot resolve threat

            return True

        def _find_threats(self, plan: PartialOrderPlan) -> List[Tuple[Action, CausalLink]]:
            """Find all threats in the plan."""
            threats = []

            for action in plan.actions:
                for link in plan.causal_links:
                    # Check if action threatens the causal link
                    if (action != link.producer and action != link.consumer and
                        link.proposition in action.effects):

                        # Check if action could interfere (not already ordered)
                        if not self._is_ordered(action, link.producer, plan) and \
                           not self._is_ordered(link.consumer, action, plan):
                            threats.append((action, link))

            return threats

        def _is_ordered(self, action1: Action, action2: Action, plan: PartialOrderPlan) -> bool:
            """Check if action1 is ordered before action2."""
            return (action1, action2) in plan.ordering_constraints

        def _can_promote(self, threat_action: Action, causal_link: CausalLink, plan: PartialOrderPlan) -> bool:
            """Check if threat can be resolved by promotion."""
            # Promotion: threat_action after causal_link.consumer
            return not self._creates_cycle(threat_action, causal_link.consumer, plan)

        def _can_demote(self, threat_action: Action, causal_link: CausalLink, plan: PartialOrderPlan) -> bool:
            """Check if threat can be resolved by demotion."""
            # Demotion: threat_action before causal_link.producer
            return not self._creates_cycle(causal_link.producer, threat_action, plan)

        def _creates_cycle(self, action1: Action, action2: Action, plan: PartialOrderPlan) -> bool:
            """Check if adding ordering constraint would create a cycle."""
            # Simplified cycle detection - would use proper graph cycle detection
            return False
    ```

=== "POP with Backtracking"
    ```python
    class BacktrackingPOP(PartialOrderPlanner):
        """
        POP implementation with systematic backtracking.
        """

        def __init__(self, actions: List[Action], initial_state: Set[Proposition], goal: Set[Proposition]):
            super().__init__(actions, initial_state, goal)
            self.search_stack = []

        def plan(self) -> Optional[PartialOrderPlan]:
            """Find plan using systematic backtracking."""
            start_action = Action("START", set(), self.initial_state)
            goal_action = Action("GOAL", self.goal, set())
            initial_plan = PartialOrderPlan(start_action, goal_action)

            # Add initial causal links
            for goal_prop in self.goal:
                initial_plan.causal_links.add(CausalLink(start_action, goal_prop, goal_action))

            self.search_stack = [initial_plan]

            while self.search_stack:
                current_plan = self.search_stack.pop()

                if self._is_complete(current_plan):
                    return current_plan

                # Generate successor plans
                successors = self._generate_successors(current_plan)
                self.search_stack.extend(successors)

            return None

        def _generate_successors(self, plan: PartialOrderPlan) -> List[PartialOrderPlan]:
            """Generate all possible successor plans."""
            successors = []
            open_conditions = self._find_open_conditions(plan)

            for action, precondition in open_conditions:
                supporters = self._find_supporters(precondition, plan)

                for supporter in supporters:
                    new_plan = self._add_action_to_plan(plan, supporter, action, precondition)

                    if self._resolve_threats(new_plan):
                        successors.append(new_plan)

            return successors
    ```

=== "POP with Heuristic Search"
    ```python
    class HeuristicPOP(PartialOrderPlanner):
        """
        POP implementation with heuristic-guided search.
        """

        def __init__(self, actions: List[Action], initial_state: Set[Proposition], goal: Set[Proposition]):
            super().__init__(actions, initial_state, goal)
            self.heuristic_func = self._plan_size_heuristic

        def plan(self) -> Optional[PartialOrderPlan]:
            """Find plan using heuristic-guided search."""
            import heapq

            start_action = Action("START", set(), self.initial_state)
            goal_action = Action("GOAL", self.goal, set())
            initial_plan = PartialOrderPlan(start_action, goal_action)

            # Add initial causal links
            for goal_prop in self.goal:
                initial_plan.causal_links.add(CausalLink(start_action, goal_prop, goal_action))

            # Priority queue for heuristic search
            open_set = [(self.heuristic_func(initial_plan), initial_plan)]
            closed_set = set()

            while open_set:
                _, current_plan = heapq.heappop(open_set)

                if self._is_complete(current_plan):
                    return current_plan

                plan_key = self._plan_to_key(current_plan)
                if plan_key in closed_set:
                    continue

                closed_set.add(plan_key)

                # Generate successors
                successors = self._generate_successors(current_plan)
                for successor in successors:
                    successor_key = self._plan_to_key(successor)
                    if successor_key not in closed_set:
                        heapq.heappush(open_set, (self.heuristic_func(successor), successor))

            return None

        def _plan_size_heuristic(self, plan: PartialOrderPlan) -> int:
            """Heuristic based on plan size and open conditions."""
            open_conditions = len(self._find_open_conditions(plan))
            return len(plan.actions) + open_conditions

        def _plan_to_key(self, plan: PartialOrderPlan) -> str:
            """Convert plan to a unique key for duplicate detection."""
            actions_key = tuple(sorted(action.name for action in plan.actions))
            links_key = tuple(sorted((link.producer.name, link.proposition.name, link.consumer.name)
                                   for link in plan.causal_links))
            return str((actions_key, links_key))
    ```

!!! tip "Complete Implementation"
    The full implementation with comprehensive testing, additional variants, and performance optimizations is available in the source code:

    - **Main Implementation**: [`src/algokit/planning/partial_order_planning.py`](https://github.com/jeffrichley/algokit/blob/main/src/algokit/planning/partial_order_planning.py)
    - **Tests**: [`tests/unit/planning/test_partial_order_planning.py`](https://github.com/jeffrichley/algokit/blob/main/tests/unit/planning/test_partial_order_planning.py)

## Complexity Analysis

!!! example "**Time & Space Complexity Comparison**"
    | Approach | Time Complexity | Space Complexity | Notes |
    |----------|-----------------|------------------|-------|
    **Basic POP** | $O(b^d \cdot n)$ | $O(b^d \cdot n)$ | Exponential in plan depth |
    **Backtracking POP** | $O(b^d \cdot n)$ | $O(d \cdot n)$ | Reduced space with backtracking |
    **Heuristic POP** | $O(b^d \cdot n)$ | $O(b^d \cdot n)$ | Better search guidance |

!!! warning "Performance Considerations"
    - **Threat resolution** can be computationally expensive
    - **Cycle detection** is crucial for maintaining plan validity
    - **Search space** can be large for complex domains
    - **Heuristic quality** significantly affects performance

## Use Cases & Applications

!!! grid "Application Categories"
    !!! grid-item "Flexible Planning"
        - **Robotics**: Task planning with flexible execution order
        - **Manufacturing**: Production planning with parallel operations
        - **Logistics**: Delivery planning with flexible scheduling
        - **Project Management**: Task scheduling with dependencies

    !!! grid-item "Parallel Execution"
        - **Distributed Systems**: Task coordination and scheduling
        - **Multi-agent Systems**: Coordinated action planning
        - **Cloud Computing**: Resource allocation and task scheduling
        - **Workflow Management**: Process orchestration

    !!! grid-item "Real-World Applications"
        - **Supply Chain**: Flexible production and distribution planning
        - **Transportation**: Route planning with multiple constraints
        - **Healthcare**: Treatment planning with flexible scheduling
        - **Finance**: Portfolio management with flexible execution

    !!! grid-item "Educational Value"
        - **Planning Theory**: Understanding least-commitment strategies
        - **Constraint Satisfaction**: Learning threat resolution
        - **Graph Theory**: Understanding partial orders and cycles
        - **Algorithm Design**: Systematic problem decomposition

!!! success "Educational Value"
    - **Least Commitment**: Perfect example of flexible planning approaches
    - **Constraint Satisfaction**: Shows how to handle complex constraints
    - **Graph Theory**: Demonstrates partial order relationships
    - **Algorithm Design**: Illustrates systematic threat resolution

## References & Further Reading

!!! grid "Reference Categories"
    !!! grid-item "Core Papers"
        1. **Penberthy, J. S., & Weld, D. S.** (1992). UCPOP: A sound, complete, partial order planner for ADL. *AAAI*, 92, 103-114.
        2. **Kambhampati, S.** (1997). Refinement planning as a unifying framework for plan synthesis. *AI Magazine*, 18(2), 67-97.

    !!! grid-item "Planning Textbooks"
        3. **Ghallab, M., Nau, D., & Traverso, P.** (2016). *Automated Planning and Acting*. Cambridge University Press.
        4. **Russell, S., & Norvig, P.** (2020). *Artificial Intelligence: A Modern Approach*. Pearson.

    !!! grid-item "Online Resources"
        5. [Partial Order Planning - Wikipedia](https://en.wikipedia.org/wiki/Partial-order_planning)
        6. [POP Algorithm Tutorial](https://www.cs.cmu.edu/afs/cs.cmu.edu/academic/class/15381-s07/www/slides/partial-order-planning.pdf)
        7. [Classical Planning Overview](https://www.cs.cmu.edu/afs/cs.cmu.edu/academic/class/15381-s07/www/slides/classical-planning.pdf)

    !!! grid-item "Implementation & Practice"
        8. [PDDL - Planning Domain Definition Language](https://planning.wiki/ref/pddl)
        9. [UCPOP Planner](https://github.com/pucrs-automated-planning/ucpop)
        10. [Planning Algorithms Library](https://github.com/pucrs-automated-planning/planning-algorithms)

!!! tip "Interactive Learning"
    Try implementing POP yourself! Start with simple planning domains like the blocks world to understand the basics. Experiment with different threat resolution strategies to see how they affect plan quality. Try implementing cycle detection to understand how to maintain plan validity. Compare POP with total-order planners to see the benefits of flexible planning. This will give you deep insight into least-commitment planning and constraint satisfaction.

## Navigation

{{ nav_grid(current_algorithm="partial-order-planning", current_family="planning", max_related=5) }}
