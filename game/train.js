/**
 * Minimal Training Loop for Fruit Merge RL
 * 
 * A simple Q-learning training module using TensorFlow.js.
 * This is a minimal implementation to verify that states, actions,
 * rewards, and Q-target training behave correctly.
 * 
 * @module train
 */

/**
 * Initialize the Q-network model.
 * 
 * Creates a tiny neural network with:
 * - Input: state size (from RL.getState().length)
 * - 2 hidden dense layers with 32 units each, ReLU activation
 * - Output: 4 units (Q-values for 4 actions)
 * 
 * Uses Adam optimizer with learning rate 0.001 and MSE loss.
 * Exposes the model on window.RL.model.
 */
export function initModel() {
    // Get state size from RL interface
    const stateSize = window.RL.getState().length;
    
    // Build model
    const model = tf.sequential();
    
    // Input layer + first hidden layer
    model.add(tf.layers.dense({
        inputShape: [stateSize],
        units: 32,
        activation: 'relu'
    }));
    
    // Second hidden layer
    model.add(tf.layers.dense({
        units: 32,
        activation: 'relu'
    }));
    
    // Output layer (4 Q-values for 4 actions)
    model.add(tf.layers.dense({
        units: 4,
        activation: 'linear'
    }));
    
    // Compile with Adam optimizer and MSE loss
    model.compile({
        optimizer: tf.train.adam(0.001),
        loss: 'meanSquaredError'
    });
    
    // Expose model on window.RL
    window.RL.model = model;
    
    console.log('[Train] Model initialized');
    console.log('[Train] State size:', stateSize);
    model.summary();
    
    return model;
}

/**
 * Predict the best action for a given state.
 * 
 * Runs a forward pass on the model and returns the index
 * of the action with the highest Q-value.
 * Pure exploitation (argmax) - no epsilon-greedy exploration.
 * 
 * @param {number[]} state - The current state vector
 * @returns {number} Action index (0-3)
 */
export function predictAction(state) {
    const model = window.RL.model;
    if (!model) {
        throw new Error('Model not initialized. Call initModel() first.');
    }
    
    // Convert state to tensor and add batch dimension
    const stateTensor = tf.tensor2d([state], [1, state.length]);
    
    // Forward pass
    const qValues = model.predict(stateTensor);
    
    // Get argmax
    const actionIndex = qValues.argMax(1).dataSync()[0];
    
    // Clean up tensors
    stateTensor.dispose();
    qValues.dispose();
    
    return actionIndex;
}

/**
 * Train the model on a single transition.
 * 
 * Computes target Q-value for action a:
 * - If done: target = r
 * - Else: target = r + 0.95 * max(Q(s2))
 * 
 * Then trains the model on this single sample using model.fit().
 * 
 * @param {number[]} s - Current state
 * @param {number} a - Action taken
 * @param {number} r - Reward received
 * @param {number[]} s2 - Next state
 * @param {boolean} done - Whether episode is terminated
 * @returns {Promise<number>} Training loss
 */
export async function trainOneStep(s, a, r, s2, done) {
    const model = window.RL.model;
    if (!model) {
        throw new Error('Model not initialized. Call initModel() first.');
    }
    
    const gamma = 0.95; // Discount factor
    
    // Convert states to tensors
    const sTensor = tf.tensor2d([s], [1, s.length]);
    const s2Tensor = tf.tensor2d([s2], [1, s2.length]);
    
    // Get current Q-values for state s
    const currentQ = model.predict(sTensor);
    const currentQData = currentQ.dataSync().slice();
    
    // Compute target Q-value for action a
    let target;
    if (done) {
        target = r;
    } else {
        // Get max Q-value for next state
        const nextQ = model.predict(s2Tensor);
        const maxNextQ = nextQ.max().dataSync()[0];
        target = r + gamma * maxNextQ;
        nextQ.dispose();
    }
    
    // Update Q[a] with target
    currentQData[a] = target;
    
    // Create target tensor
    const targetTensor = tf.tensor2d([currentQData], [1, 4]);
    
    // Train the model
    const result = await model.fit(sTensor, targetTensor, {
        epochs: 1,
        verbose: 0
    });
    
    const loss = result.history.loss[0];
    
    // Clean up tensors
    sTensor.dispose();
    s2Tensor.dispose();
    currentQ.dispose();
    targetTensor.dispose();
    
    return loss;
}

/**
 * Run training for multiple episodes.
 * 
 * For each episode:
 * 1. Reset the game using RL.resetEpisode()
 * 2. Loop until done:
 *    - Get state via RL.getState()
 *    - Choose action via predictAction()
 *    - Execute action via RL.step()
 *    - Advance physics via RL.stepPhysics()
 *    - Tick cooldown via RL.tickCooldown()
 *    - Get reward and check if done
 *    - Train using trainOneStep()
 * 3. Log final reward and steps
 * 
 * Note: This is intentionally slow (no batching, no replay buffer)
 * but sufficient to verify the training loop works.
 * 
 * @param {number} numEpisodes - Number of episodes to run
 * @returns {Promise<Object[]>} Array of episode results
 */
export async function trainEpisodes(numEpisodes) {
    const model = window.RL.model;
    if (!model) {
        throw new Error('Model not initialized. Call initModel() first.');
    }
    
    // Check required RL methods
    if (typeof window.RL.setHeadlessMode !== 'function' ||
        typeof window.RL.stepPhysics !== 'function' ||
        typeof window.RL.tickCooldown !== 'function' ||
        typeof window.RL.getRunner !== 'function' ||
        typeof window.RL.getRender !== 'function') {
        throw new Error('RL interface incomplete. Required: setHeadlessMode, stepPhysics, tickCooldown, getRunner, getRender');
    }
    
    console.log(`[Train] Starting training for ${numEpisodes} episodes`);
    
    const results = [];
    
    // Enable headless mode to disable rendering and audio
    window.RL.setHeadlessMode(true);
    
    // Stop the normal game runner and render to take control of physics updates
    const runner = window.RL.getRunner();
    const render = window.RL.getRender();
    
    // Note: Matter.js is expected to be available globally when this code runs
    // These checks are defensive in case the global isn't available
    if (typeof Matter !== 'undefined') {
        if (runner && Matter.Runner) {
            Matter.Runner.stop(runner);
        }
        if (render && Matter.Render) {
            Matter.Render.stop(render);
        }
    }
    
    try {
        for (let ep = 0; ep < numEpisodes; ep++) {
            // Reset episode
            window.RL.resetEpisode();
            
            let steps = 0;
            let totalReward = 0;
            let done = false;
            
            // Get initial state
            let state = window.RL.getState();
            
            while (!done) {
                // Choose action
                const action = predictAction(state);
                
                // Execute action and get reward
                const reward = window.RL.step(action);
                
                // Advance physics simulation
                window.RL.stepPhysics();
                
                // Tick the step-based cooldown counter
                window.RL.tickCooldown();
                
                // Check if terminal
                done = window.RL.isTerminal();
                
                // Get next state
                const nextState = window.RL.getState();
                
                // Train on this transition
                await trainOneStep(state, action, reward, nextState, done);
                
                // Update for next iteration
                state = nextState;
                totalReward += reward;
                steps++;
                
                // Yield to event loop periodically
                if (steps % 50 === 0) {
                    await new Promise(resolve => setTimeout(resolve, 0));
                }
            }
            
            const episodeResult = {
                episode: ep,
                steps: steps,
                totalReward: totalReward
            };
            
            results.push(episodeResult);
            
            console.log(`[Train] Episode ${ep}: steps=${steps}, reward=${totalReward.toFixed(2)}`);
        }
    } finally {
        // Restore normal mode
        window.RL.setHeadlessMode(false);
        
        // Reset the game to leave it in a clean state with rendering restored
        // RL.reset() calls handleRestart() which calls initGame()
        // initGame() restarts Render.run() and Runner.run() automatically
        try {
            if (typeof window.RL.reset === 'function') {
                window.RL.reset();
            }
        } catch (e) {
            console.warn('[Train] Failed to reset game after training:', e);
        }
    }
    
    console.log('[Train] Training completed');
    
    return results;
}
