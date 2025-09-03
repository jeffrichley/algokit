# Dynamic Movement Primitives (DMPs)

## Overview
Dynamic Movement Primitives (DMPs) are a framework for learning and reproducing complex motor skills in robotics and humanoid systems. DMPs represent movements as dynamical systems that can be learned from demonstrations, modified through parameters, and adapted to new situations. They provide a robust way to encode, learn, and generalize complex motor behaviors.

## Key Concepts
- **Movement Primitives**: Basic building blocks of complex motor behaviors
- **Dynamical Systems**: Mathematical models that evolve over time
- **Learning from Demonstration**: Acquiring skills by observing expert behavior
- **Generalization**: Adapting learned movements to new contexts
- **Temporal Scaling**: Adjusting movement speed while preserving shape
- **Spatial Scaling**: Modifying movement amplitude and direction

## Comparison Table
| Algorithm | Complexity | Strengths | Weaknesses | Applications |
|-----------|------------|-----------|------------|--------------|
| Basic DMPs | O(n) | Simple, interpretable, real-time | Limited to single trajectories | Robot learning, motion planning |
| Adaptive DMPs | O(n²) | Online adaptation, robust to changes | Higher computational cost | Human-robot interaction, adaptive control |
| Multi-dimensional DMPs | O(n) | Coordinated multi-joint movements | More parameters, complex training | Humanoid robotics, bimanual tasks |
| Probabilistic DMPs | O(n²) | Uncertainty quantification, multiple solutions | Statistical interpretation needed | Safe robotics, human-like motion |
| DMPs with Obstacle Avoidance | O(n²) | Reactive obstacle avoidance, safety | Real-time constraints, path planning | Mobile robotics, navigation |

## Algorithms in This Family
- [Basic DMPs](../algorithms/dmps/basic-dmps.md)
- [Adaptive DMPs](../algorithms/dmps/adaptive-dmps.md)
- [Multi-dimensional DMPs](../algorithms/dmps/multi-dimensional-dmps.md)
- [Probabilistic DMPs](../algorithms/dmps/probabilistic-dmps.md)
- [DMPs with Obstacle Avoidance](../algorithms/dmps/dmps-obstacle-avoidance.md)
