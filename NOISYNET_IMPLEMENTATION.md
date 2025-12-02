# NoisyNet Implementation

## Overview
This implementation replaces the epsilon-greedy exploration strategy with NoisyNet layers as described in the paper "Noisy Networks for Exploration" (Fortunato et al., 2017).

## Changes Made

### 1. NoisyDense Layer
Created a custom TensorFlow.js layer `NoisyDense` that implements factorized Gaussian noise:

**Formula**: `y = (μ_w + σ_w ⊙ ε_w) x + (μ_b + σ_b ⊙ ε_b)`

Where:
- `μ_w`, `μ_b`: Trainable mean parameters for weights and biases
- `σ_w`, `σ_b`: Trainable standard deviation parameters for weights and biases
- `ε_w`, `ε_b`: Factorized Gaussian noise (reset each forward pass)

**Factorized Noise**: Uses `f(x) = sgn(x) * sqrt(|x|)` to reduce noise parameters from O(p*q) to O(p+q).

### 2. Model Architecture Changes
Replaced the final layers in the Dueling DQN architecture:
- **Before**: Regular Dense layers for Value and Advantage streams
- **After**: NoisyDense layers for both streams

```javascript
// Value stream: NoisyDense 1 unit
// Advantage stream: NoisyDense 4 units
```

### 3. Exploration Changes
Removed epsilon-greedy exploration entirely:
- **Before**: `selectActionFromBuffer(state, epsilon)` used epsilon-greedy
- **After**: Simply selects argmax of noisy Q-values

The noise in the network parameters provides automatic exploration without needing epsilon decay.

### 4. Key Features
- **Automatic Exploration**: No need for epsilon scheduling
- **Trainable Exploration**: Network learns optimal exploration strategy
- **Memory Efficient**: Factorized noise reduces parameters
- **Noise Reset**: Fresh noise generated at each forward pass

## Benefits
1. **No Hyperparameter Tuning**: No need to tune epsilon, epsilon_start, epsilon_end, epsilon_decay
2. **Better Exploration**: Network learns when to explore based on uncertainty
3. **Simpler Code**: Remove epsilon decay logic from training loop
4. **State-dependent Exploration**: Different states get different exploration levels

## Compatibility
- Epsilon parameters in `RL.train()` are now ignored for compatibility
- `RL.selectAction(state, epsilon)` still accepts epsilon but ignores it
- Training loop unchanged - works with existing code

## Memory Management
The implementation properly manages TensorFlow.js tensors:
- Noise tensors are disposed before regeneration
- Custom `dispose()` method cleans up layer resources
- All operations wrapped in `tf.tidy()` for automatic cleanup

## Testing
The implementation has been validated for:
- ✓ Correct syntax (Node.js --check)
- ✓ Proper layer registration with TensorFlow.js
- ✓ Memory management (tensor disposal)
- ✓ Integration with existing Dueling DQN architecture

## References
- Fortunato, M., et al. (2017). "Noisy Networks for Exploration." arXiv:1706.10295
