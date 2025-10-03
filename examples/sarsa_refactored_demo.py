#!/usr/bin/env python3
"""Demo script showcasing the refactored SARSA agent features."""

import numpy as np
from algokit.algorithms.reinforcement_learning.sarsa import SarsaAgent


def main() -> None:
    """Demonstrate the refactored SARSA agent features."""
    print("🚀 SARSA Agent Refactored Demo")
    print("=" * 50)

    # Create a simple environment: 3x3 grid world
    # States: 0-8 (3x3 grid), Actions: 0-3 (up, down, left, right)
    state_space_size = 9
    action_space_size = 4

    print("\n1. Basic Agent Creation")
    print("-" * 30)
    agent = SarsaAgent(
        state_space_size=state_space_size,
        action_space_size=action_space_size,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon_start=0.9,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        random_seed=42,
    )
    print(f"Created agent: {agent}")

    print("\n2. Random Tie-Breaking Demo")
    print("-" * 30)
    # Set all Q-values to the same value to create ties
    agent.q_table[0, :] = 1.0
    agent.epsilon = 0.0  # No exploration

    print("Actions selected with tied Q-values (should be random):")
    actions = [agent.select_action(0) for _ in range(10)]
    print(f"Actions: {actions}")
    unique_actions = set(actions)
    print(f"Unique actions: {unique_actions} (shows random tie-breaking)")

    print("\n3. Reproducibility Demo")
    print("-" * 30)
    agent1 = SarsaAgent(state_space_size=3, action_space_size=2)
    agent2 = SarsaAgent(state_space_size=3, action_space_size=2)

    agent1.set_seed(123)
    agent2.set_seed(123)

    actions1 = [agent1.select_action(0) for _ in range(5)]
    actions2 = [agent2.select_action(0) for _ in range(5)]

    print(f"Agent 1 actions: {actions1}")
    print(f"Agent 2 actions: {actions2}")
    print(f"Actions identical: {actions1 == actions2}")

    print("\n4. Expected SARSA Demo")
    print("-" * 30)
    expected_agent = SarsaAgent(
        state_space_size=3,
        action_space_size=2,
        use_expected_sarsa=True,
        random_seed=42,
    )

    print(f"Expected SARSA enabled: {expected_agent.use_expected_sarsa}")

    # Perform some updates
    for _ in range(5):
        next_action = expected_agent.step(state=0, action=0, reward=1.0, next_state=1, done=False)
        print(f"Next action: {next_action}")

    print("After updates:")
    print(f"Q-table shape: {expected_agent.q_table.shape}")

    print("\n5. Debug Logging Demo")
    print("-" * 30)
    debug_agent = SarsaAgent(
        state_space_size=3,
        action_space_size=2,
        debug=True,
    )

    print("Performing step with debug logging:")
    next_action = debug_agent.step(state=0, action=0, reward=1.0, next_state=1, done=False)
    print(f"Next action: {next_action}")
    print("(Debug output would appear above if logging is configured)")

    print("\n6. Introspection Methods Demo")
    print("-" * 30)
    # Set some Q-values for demonstration
    agent.q_table[0, 0] = 0.5
    agent.q_table[0, 1] = 0.8
    agent.q_table[0, 2] = 0.3
    agent.q_table[0, 3] = 0.6

    print(f"Q-value for state 0, action 1: {agent.get_q_value(0, 1)}")
    print(f"Action values for state 0: {agent.get_action_values(0)}")

    policy = agent.get_policy()
    print(f"Greedy policy: {policy}")

    print(f"Policy entropy: {agent.get_policy_entropy():.3f}")
    print(f"Average Q-value magnitude: {agent.get_average_q_magnitude():.3f}")

    print("\n7. Pretty Print Policy Demo")
    print("-" * 30)
    state_names = ["Top-Left", "Top-Center", "Top-Right",
                   "Mid-Left", "Mid-Center", "Mid-Right",
                   "Bot-Left", "Bot-Center", "Bot-Right"]
    action_names = ["Up", "Down", "Left", "Right"]

    policy_str = agent.pretty_print_policy(state_names, action_names)
    print(policy_str)

    print("\n8. Epsilon Scheduling Demo")
    print("-" * 30)
    print(f"Initial epsilon: {agent.epsilon:.3f}")
    for i in range(5):
        agent.decay_epsilon()
        print(f"After decay {i+1}: {agent.epsilon:.3f}")

    print("\n9. On-Policy Behavior Demo")
    print("-" * 30)
    on_policy_agent = SarsaAgent(
        state_space_size=3,
        action_space_size=2,
        epsilon_start=0.8,  # High exploration
        random_seed=42,
    )

    print("Performing SARSA step (on-policy):")
    next_action = on_policy_agent.step(state=0, action=0, reward=1.0, next_state=1, done=False)
    print(f"Next action chosen by current epsilon-greedy policy: {next_action}")

    print("\n10. Parameter Validation Demo")
    print("-" * 30)
    try:
        invalid_agent = SarsaAgent(
            state_space_size=3,
            action_space_size=2,
            learning_rate=0.0,  # Invalid: must be > 0
        )
    except ValueError as e:
        print(f"Caught expected error: {e}")

    print("\n11. Backward Compatibility Demo")
    print("-" * 30)
    # Test old parameter names
    old_style_agent = SarsaAgent(
        state_space_size=3,
        action_space_size=2,
        epsilon=0.5,      # Old parameter name
        epsilon_min=0.1, # Old parameter name
    )
    print(f"Old-style agent created successfully: epsilon={old_style_agent.epsilon}")

    # Test old method names
    action = old_style_agent.get_action(0)  # Old method name
    old_style_agent.update_q_value(0, 0, 1.0, 1, 1, False)  # Old method name
    print(f"Old-style methods work: action={action}")

    print("\n12. Expected SARSA vs Standard SARSA Demo")
    print("-" * 30)
    standard_agent = SarsaAgent(
        state_space_size=3,
        action_space_size=2,
        epsilon_start=0.5,
        use_expected_sarsa=False,
        random_seed=42,
    )
    expected_agent = SarsaAgent(
        state_space_size=3,
        action_space_size=2,
        epsilon_start=0.5,
        use_expected_sarsa=True,
        random_seed=42,
    )

    # Perform same updates
    standard_agent.update_q_value(0, 0, 1.0, 1, 1, False)
    expected_agent.update_q_value(0, 0, 1.0, 1, 1, False)

    print(f"Standard SARSA Q-value: {standard_agent.q_table[0, 0]:.3f}")
    print(f"Expected SARSA Q-value: {expected_agent.q_table[0, 0]:.3f}")
    print(f"Values differ: {not np.allclose(standard_agent.q_table, expected_agent.q_table)}")

    print("\n✅ Demo completed successfully!")
    print("\nKey Features Demonstrated:")
    print("• Random tie-breaking in action selection")
    print("• Reproducible results with set_seed()")
    print("• Expected SARSA for bias reduction")
    print("• Debug logging for TD error tracking")
    print("• Introspection methods for Q-values and policies")
    print("• Pretty printing with custom state/action names")
    print("• Configurable epsilon scheduling")
    print("• Strict on-policy behavior enforcement")
    print("• Strict parameter validation")
    print("• Full backward compatibility")
    print("• Policy entropy and Q-value magnitude analysis")


if __name__ == "__main__":
    main()
