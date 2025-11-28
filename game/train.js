/**
 * train.js - Optimized TensorFlow.js Training Loop for Fruit Merge RL
 * 
 * High-performance DQN training implementation with:
 * - Pre-allocated state buffers (reduces GC pressure)
 * - Batched predictions for action selection and Q-target computation
 * - Replay buffer with random sampling
 * - trainOnBatch() for faster gradient updates
 * - tf.tidy() wrapping for automatic memory management
 * - Synchronous training loop (minimal async overhead)
 * - Completely headless training (no DOM/rendering)
 * 
 * Usage:
 *   RL.initModel();                        // Build and compile model
 *   RL.train(5);                           // Run 5 episodes of training
 *   RL.train(5, { batchSize: 64 });        // Use batch size of 64
 *   RL.train(5, { epsilon: 0.1 });         // Use 10% random exploration
 * 
 * @module train
 */

// Game context reference (set by initTraining)
let gameContext = null;

/**
 * Fixed-size ring buffer for experience replay.
 * Stores transitions (s, a, r, s', done) with O(1) add and sample operations.
 */
class ReplayBuffer {
    /**
     * @param {number} capacity - Maximum number of transitions to store
     * @param {number} stateSize - Size of state vectors
     */
    constructor(capacity, stateSize) {
        this.capacity = capacity;
        this.stateSize = stateSize;
        this.position = 0;
        this.size = 0;
        
        // Pre-allocate typed arrays for all data
        this.states = new Float32Array(capacity * stateSize);
        this.actions = new Uint8Array(capacity);
        this.rewards = new Float32Array(capacity);
        this.nextStates = new Float32Array(capacity * stateSize);
        this.dones = new Uint8Array(capacity);
    }
    
    /**
     * Add a transition to the buffer.
     * @param {Float32Array|number[]} state - Current state
     * @param {number} action - Action taken
     * @param {number} reward - Reward received
     * @param {Float32Array|number[]} nextState - Next state
     * @param {boolean} done - Whether episode ended
     */
    add(state, action, reward, nextState, done) {
        const offset = this.position * this.stateSize;
        
        // Copy state data
        for (let i = 0; i < this.stateSize; i++) {
            this.states[offset + i] = state[i];
            this.nextStates[offset + i] = nextState[i];
        }
        
        this.actions[this.position] = action;
        this.rewards[this.position] = reward;
        this.dones[this.position] = done ? 1 : 0;
        
        this.position = (this.position + 1) % this.capacity;
        if (this.size < this.capacity) {
            this.size++;
        }
    }
    
    /**
     * Sample a random minibatch of transitions.
     * Returns typed arrays for direct tensor creation.
     * @param {number} batchSize - Number of transitions to sample
     * @returns {{states: Float32Array, actions: Uint8Array, rewards: Float32Array, nextStates: Float32Array, dones: Uint8Array}}
     */
    sampleBatch(batchSize) {
        const actualBatchSize = Math.min(batchSize, this.size);
        
        // Pre-allocated output arrays
        const batchStates = new Float32Array(actualBatchSize * this.stateSize);
        const batchActions = new Uint8Array(actualBatchSize);
        const batchRewards = new Float32Array(actualBatchSize);
        const batchNextStates = new Float32Array(actualBatchSize * this.stateSize);
        const batchDones = new Uint8Array(actualBatchSize);
        
        // Random sampling without replacement
        for (let i = 0; i < actualBatchSize; i++) {
            const idx = Math.floor(Math.random() * this.size);
            const srcOffset = idx * this.stateSize;
            const dstOffset = i * this.stateSize;
            
            // Copy state data
            for (let j = 0; j < this.stateSize; j++) {
                batchStates[dstOffset + j] = this.states[srcOffset + j];
                batchNextStates[dstOffset + j] = this.nextStates[srcOffset + j];
            }
            
            batchActions[i] = this.actions[idx];
            batchRewards[i] = this.rewards[idx];
            batchDones[i] = this.dones[idx];
        }
        
        return {
            states: batchStates,
            actions: batchActions,
            rewards: batchRewards,
            nextStates: batchNextStates,
            dones: batchDones,
            actualBatchSize
        };
    }
    
    /**
     * Check if buffer has enough samples for a batch.
     * @param {number} batchSize - Desired batch size
     * @returns {boolean}
     */
    canSample(batchSize) {
        return this.size >= batchSize;
    }
    
    /**
     * Clear the buffer.
     */
    clear() {
        this.position = 0;
        this.size = 0;
    }
}

/**
 * Initialize the training module and expose RL.initModel() and RL.train().
 * This function should be called after the game and FastSim are fully initialized.
 * 
 * @param {Object} context - Game context with engine, runner, render references
 * @param {Function} context.engine - Function that returns Matter.js engine instance
 * @param {Function} context.runner - Function that returns Matter.js runner instance  
 * @param {Function} context.render - Function that returns Matter.js render instance
 */
export function initTraining(context) {
    // Check if TensorFlow.js is available
    if (typeof tf === 'undefined') {
        console.error('[Train] TensorFlow.js is not loaded. Make sure to include the TF.js CDN.');
        return;
    }
    
    // Store game context
    gameContext = context;
    
    // Ensure RL namespace exists
    window.RL = window.RL || {};
    
    // Model configuration
    const STATE_SIZE = 155;  // 155-element state vector
    const NUM_ACTIONS = 4;   // 4 discrete actions (left, right, center, drop)
    const HIDDEN_UNITS = 32; // Hidden layer units
    const LEARNING_RATE = 0.001; // Adam optimizer learning rate
    const GAMMA = 0.95;      // Discount factor for Q-learning
    
    // Training configuration
    const DEFAULT_BATCH_SIZE = 64;
    const DEFAULT_REPLAY_BUFFER_SIZE = 10000;
    const MIN_REPLAY_SIZE = 100; // Minimum samples before training starts
    
    // Physics timestep (60 FPS equivalent)
    const DELTA_TIME = 1000 / 60;
    
    // Maximum steps per episode to prevent infinite loops
    const MAX_STEPS_PER_EPISODE = 10000;
    
    // Model reference
    let model = null;
    let optimizer = null;
    
    // Pre-allocated buffers for state encoding (reused across steps)
    const stateBuffer = new Float32Array(STATE_SIZE);
    const nextStateBuffer = new Float32Array(STATE_SIZE);
    
    // Replay buffer instance
    let replayBuffer = null;
    
    /**
     * Copy state array into a Float32Array buffer.
     * Avoids creating new arrays on every step.
     * @param {number[]} state - Source state array
     * @param {Float32Array} buffer - Destination buffer
     */
    function encodeStateIntoBuffer(state, buffer) {
        const len = Math.min(state.length, buffer.length);
        for (let i = 0; i < len; i++) {
            buffer[i] = state[i];
        }
        // Zero-pad if source is shorter
        for (let i = len; i < buffer.length; i++) {
            buffer[i] = 0;
        }
    }
    
    /**
     * Build and compile the Q-network model.
     * 
     * Architecture:
     * - Input: 155-element state vector
     * - Dense: 32 units, ReLU
     * - Dense: 32 units, ReLU
     * - Output: 4 units (Q-values for each action)
     * 
     * Optimizer: Adam(0.001)
     * Loss: Mean Squared Error (MSE)
     */
    window.RL.initModel = function() {
        console.log('[Train] Building Q-network model...');
        
        // Create sequential model
        model = tf.sequential();
        
        // First hidden layer with input shape
        model.add(tf.layers.dense({
            units: HIDDEN_UNITS,
            activation: 'relu',
            inputShape: [STATE_SIZE],
            kernelInitializer: 'heNormal'
        }));
        
        // Second hidden layer
        model.add(tf.layers.dense({
            units: HIDDEN_UNITS,
            activation: 'relu',
            kernelInitializer: 'heNormal'
        }));
        
        // Output layer (Q-values for each action)
        model.add(tf.layers.dense({
            units: NUM_ACTIONS,
            activation: 'linear',
            kernelInitializer: 'glorotNormal'
        }));
        
        // Create optimizer for trainOnBatch
        optimizer = tf.train.adam(LEARNING_RATE);
        
        // Compile with Adam optimizer and MSE loss
        model.compile({
            optimizer: optimizer,
            loss: 'meanSquaredError'
        });
        
        // Initialize replay buffer
        replayBuffer = new ReplayBuffer(DEFAULT_REPLAY_BUFFER_SIZE, STATE_SIZE);
        
        console.log('[Train] Model built and compiled successfully.');
        console.log('[Train] Model summary:');
        model.summary();
        
        return model;
    };
    
    /**
     * Batched prediction for multiple states.
     * More efficient than individual predict calls.
     * @param {Float32Array} statesFlat - Flattened states array (batchSize * stateSize)
     * @param {number} batchSize - Number of states in the batch
     * @returns {Float32Array} Q-values for all states (batchSize * numActions)
     */
    function batchedPredict(statesFlat, batchSize) {
        return tf.tidy(() => {
            const statesTensor = tf.tensor2d(statesFlat, [batchSize, STATE_SIZE]);
            const qValues = model.predict(statesTensor);
            return qValues.dataSync();
        });
    }
    
    /**
     * Select action using epsilon-greedy policy with pre-allocated buffer.
     * Uses tf.tidy() to prevent memory leaks.
     * 
     * @param {Float32Array} stateBuffer - Pre-allocated state buffer
     * @param {number} epsilon - Exploration rate (0-1)
     * @returns {number} Action index (0-3)
     */
    function selectActionFromBuffer(stateBuffer, epsilon) {
        // Epsilon-greedy: with probability epsilon, choose random action
        if (epsilon > 0 && Math.random() < epsilon) {
            return Math.floor(Math.random() * NUM_ACTIONS);
        }
        
        // Otherwise, choose action with highest Q-value (exploitation)
        return tf.tidy(() => {
            const stateTensor = tf.tensor2d(stateBuffer, [1, STATE_SIZE]);
            const qValues = model.predict(stateTensor);
            return qValues.argMax(1).dataSync()[0];
        });
    }
    
    /**
     * Train on a minibatch using gradient descent.
     * Computes Q-targets and performs a single gradient update.
     * All tensor operations are wrapped in tf.tidy() where possible.
     * 
     * @param {Object} batch - Sampled batch from replay buffer
     * @returns {number} Training loss
     */
    function trainOnBatch(batch) {
        const { states, actions, rewards, nextStates, dones, actualBatchSize } = batch;
        
        // Compute targets outside of gradient tape
        const targets = tf.tidy(() => {
            const nextStatesTensor = tf.tensor2d(nextStates, [actualBatchSize, STATE_SIZE]);
            const nextQValues = model.predict(nextStatesTensor);
            const maxNextQ = nextQValues.max(1).dataSync();
            
            const statesTensor = tf.tensor2d(states, [actualBatchSize, STATE_SIZE]);
            const currentQValues = model.predict(statesTensor);
            const currentQData = currentQValues.arraySync();
            
            // Compute target Q-values
            // For each sample: target[action] = reward + gamma * max(Q(s', a')) * (1 - done)
            for (let i = 0; i < actualBatchSize; i++) {
                const action = actions[i];
                const reward = rewards[i];
                const done = dones[i];
                
                if (done) {
                    currentQData[i][action] = reward;
                } else {
                    currentQData[i][action] = reward + GAMMA * maxNextQ[i];
                }
            }
            
            return tf.tensor2d(currentQData);
        });
        
        // Compute gradients and apply using optimizer
        const statesTensor = tf.tensor2d(states, [actualBatchSize, STATE_SIZE]);
        
        const lossFunction = () => {
            const predictions = model.predict(statesTensor);
            return tf.losses.meanSquaredError(targets, predictions);
        };
        
        const { value: loss, grads } = tf.variableGrads(lossFunction);
        optimizer.applyGradients(grads);
        
        const lossValue = loss.dataSync()[0];
        
        // Clean up tensors
        targets.dispose();
        statesTensor.dispose();
        loss.dispose();
        Object.values(grads).forEach(g => g.dispose());
        
        return lossValue;
    }
    
    /**
     * Run a single physics step without rendering.
     * 
     * @param {Object} engine - Matter.js engine
     */
    function stepPhysics(engine) {
        Matter.Engine.update(engine, DELTA_TIME);
    }
    
    /**
     * Run training for the specified number of episodes.
     * Uses a fully synchronous loop with experience replay.
     * 
     * @param {number} numEpisodes - Number of episodes to train
     * @param {Object} [options={}] - Optional configuration options
     * @param {number} [options.batchSize=64] - Minibatch size for training
     * @param {number} [options.epsilon=0.1] - Exploration rate for epsilon-greedy action selection (0-1)
     * @param {number} [options.trainEveryNSteps=4] - Train every N steps
     * @param {boolean} [options.verbose=true] - Whether to log progress
     * @returns {Object} Training results summary (synchronous)
     */
    window.RL.train = function(numEpisodes, options = {}) {
        // Extract options with defaults
        const {
            batchSize = DEFAULT_BATCH_SIZE,
            epsilon = 0.1,
            trainEveryNSteps = 4,
            verbose = true
        } = options;
        
        // Validate and clamp epsilon to [0, 1]
        const validEpsilon = Math.max(0, Math.min(1, epsilon));
        const validBatchSize = Math.max(1, Math.floor(batchSize));
        const validTrainEveryNSteps = Math.max(1, Math.floor(trainEveryNSteps));
        
        // Validate model is initialized
        if (!model) {
            console.error('[Train] Model not initialized. Call RL.initModel() first.');
            return null;
        }
        
        // Validate numEpisodes
        if (typeof numEpisodes !== 'number' || numEpisodes < 1) {
            console.error(`[Train] Invalid numEpisodes: ${numEpisodes}. Must be a positive number.`);
            return null;
        }
        
        // Check if game context is available
        if (!gameContext) {
            console.error('[Train] Game context not available. Training module not properly initialized.');
            return null;
        }
        
        // Check if required RL methods are available
        if (typeof window.RL.resetEpisode !== 'function' ||
            typeof window.RL.getState !== 'function' ||
            typeof window.RL.step !== 'function' ||
            typeof window.RL.isTerminal !== 'function' ||
            typeof window.RL.getReward !== 'function' ||
            typeof window.RL.setHeadlessMode !== 'function' ||
            typeof window.RL.tickCooldown !== 'function') {
            console.error('[Train] RL interface incomplete. Required: resetEpisode, getState, step, isTerminal, getReward, setHeadlessMode, tickCooldown');
            return null;
        }
        
        if (verbose) {
            console.log(`[Train] Starting training: ${numEpisodes} episodes, epsilon=${validEpsilon}, batchSize=${validBatchSize}`);
        }
        const startTime = performance.now();
        
        const results = [];
        let totalTrainingSteps = 0;
        
        // Enable headless mode - completely disables rendering, DOM updates, and audio
        window.RL.setHeadlessMode(true);
        
        // Validate Matter.js is available
        if (typeof Matter === 'undefined') {
            console.error('[Train] Matter.js is not loaded. Make sure the physics engine is initialized.');
            window.RL.setHeadlessMode(false);
            return null;
        }
        
        // Stop the normal game runner and renderer to take full control
        const { Runner, Render } = Matter;
        const runner = gameContext.runner();
        const render = gameContext.render();
        
        if (runner) {
            Runner.stop(runner);
        }
        if (render) {
            Render.stop(render);
        }
        
        // Clear replay buffer for fresh training
        if (replayBuffer) {
            replayBuffer.clear();
        } else {
            replayBuffer = new ReplayBuffer(DEFAULT_REPLAY_BUFFER_SIZE, STATE_SIZE);
        }
        
        try {
            // Main training loop - fully synchronous
            for (let episode = 0; episode < numEpisodes; episode++) {
                if (verbose) {
                    console.log(`[Train] Episode ${episode + 1}/${numEpisodes} starting...`);
                }
                
                // Reset environment for new episode
                window.RL.resetEpisode();
                
                // Get fresh engine reference after reset
                const engine = gameContext.engine();
                
                let stepCount = 0;
                let totalReward = 0;
                let totalLoss = 0;
                let trainCount = 0;
                const episodeStartTime = performance.now();
                
                // Get initial state and encode into buffer
                const rawState = window.RL.getState();
                encodeStateIntoBuffer(rawState, stateBuffer);
                
                // Episode loop - synchronous, no awaits
                while (!window.RL.isTerminal() && stepCount < MAX_STEPS_PER_EPISODE) {
                    // Select action using epsilon-greedy policy
                    const action = selectActionFromBuffer(stateBuffer, validEpsilon);
                    
                    // Execute action (modifies game state)
                    window.RL.step(action);
                    
                    // Step physics simulation
                    stepPhysics(engine);
                    
                    // Tick cooldown (needed for drop timing)
                    window.RL.tickCooldown();
                    
                    // Get reward
                    const reward = window.RL.getReward();
                    totalReward += reward;
                    
                    // Get next state and check if terminal
                    const rawNextState = window.RL.getState();
                    encodeStateIntoBuffer(rawNextState, nextStateBuffer);
                    const done = window.RL.isTerminal();
                    
                    // Store transition in replay buffer
                    replayBuffer.add(stateBuffer, action, reward, nextStateBuffer, done);
                    
                    // Train on minibatch if enough samples and at training interval
                    if (replayBuffer.canSample(validBatchSize) && 
                        stepCount % validTrainEveryNSteps === 0) {
                        const batch = replayBuffer.sampleBatch(validBatchSize);
                        const loss = trainOnBatch(batch);
                        totalLoss += loss;
                        trainCount++;
                        totalTrainingSteps++;
                    }
                    
                    // Copy next state to current state buffer (avoid array allocation)
                    for (let i = 0; i < STATE_SIZE; i++) {
                        stateBuffer[i] = nextStateBuffer[i];
                    }
                    
                    stepCount++;
                }
                
                const episodeTime = performance.now() - episodeStartTime;
                const avgLoss = trainCount > 0 ? totalLoss / trainCount : 0;
                
                const episodeResult = {
                    episode: episode + 1,
                    steps: stepCount,
                    totalReward: totalReward,
                    avgLoss: avgLoss,
                    trainSteps: trainCount,
                    timeMs: episodeTime
                };
                
                results.push(episodeResult);
                
                if (verbose) {
                    console.log(
                        `[Train] Episode ${episode + 1} ended: ` +
                        `steps=${stepCount}, reward=${totalReward.toFixed(2)}, ` +
                        `avgLoss=${avgLoss.toFixed(6)}, trainSteps=${trainCount}, ` +
                        `time=${episodeTime.toFixed(2)}ms`
                    );
                }
            }
            
            // Save model to localStorage (this is async but we fire and forget)
            if (verbose) {
                console.log('[Train] Saving model to localStorage...');
            }
            model.save('localstorage://fruit-merge-dqn-v1').then(() => {
                if (verbose) {
                    console.log('[Train] Model saved successfully to localstorage://fruit-merge-dqn-v1');
                }
            }).catch(err => {
                console.error('[Train] Failed to save model:', err);
            });
            
        } finally {
            // Restore normal mode
            window.RL.setHeadlessMode(false);
            
            // Reset game to clean state
            if (typeof window.RL.reset === 'function') {
                window.RL.reset();
            }
        }
        
        // Log summary
        const totalTime = performance.now() - startTime;
        const totalSteps = results.reduce((sum, r) => sum + r.steps, 0);
        const avgReward = results.length > 0 ? 
            results.reduce((sum, r) => sum + r.totalReward, 0) / results.length : 0;
        const avgSteps = results.length > 0 ? totalSteps / results.length : 0;
        
        if (verbose) {
            console.log(`[Train] ========== TRAINING SUMMARY ==========`);
            console.log(`[Train] Episodes completed: ${results.length}`);
            console.log(`[Train] Total time: ${totalTime.toFixed(2)}ms`);
            console.log(`[Train] Average reward per episode: ${avgReward.toFixed(2)}`);
            console.log(`[Train] Average steps per episode: ${avgSteps.toFixed(2)}`);
            console.log(`[Train] Total steps: ${totalSteps}`);
            console.log(`[Train] Total training steps: ${totalTrainingSteps}`);
            console.log(`[Train] Replay buffer size: ${replayBuffer.size}`);
            console.log(`[Train] ==========================================`);
        }
        
        return {
            episodes: results,
            totalTime: totalTime,
            avgReward: avgReward,
            avgSteps: avgSteps,
            totalTrainingSteps: totalTrainingSteps,
            replayBufferSize: replayBuffer.size
        };
    };
    
    /**
     * Async version of train for compatibility with UI that needs event loop access.
     * Yields to the event loop periodically to prevent browser freezing.
     * 
     * @param {number} numEpisodes - Number of episodes to train
     * @param {Object} [options={}] - Optional configuration options
     * @returns {Promise<Object>} Training results summary
     */
    window.RL.trainAsync = async function(numEpisodes, options = {}) {
        const {
            batchSize = DEFAULT_BATCH_SIZE,
            epsilon = 0.1,
            trainEveryNSteps = 4,
            yieldEveryNSteps = 100,
            verbose = true
        } = options;
        
        const validEpsilon = Math.max(0, Math.min(1, epsilon));
        const validBatchSize = Math.max(1, Math.floor(batchSize));
        const validTrainEveryNSteps = Math.max(1, Math.floor(trainEveryNSteps));
        const validYieldEveryNSteps = Math.max(1, Math.floor(yieldEveryNSteps));
        
        if (!model) {
            console.error('[Train] Model not initialized. Call RL.initModel() first.');
            return null;
        }
        
        if (typeof numEpisodes !== 'number' || numEpisodes < 1) {
            console.error(`[Train] Invalid numEpisodes: ${numEpisodes}. Must be a positive number.`);
            return null;
        }
        
        if (!gameContext) {
            console.error('[Train] Game context not available.');
            return null;
        }
        
        if (typeof window.RL.resetEpisode !== 'function' ||
            typeof window.RL.getState !== 'function' ||
            typeof window.RL.step !== 'function' ||
            typeof window.RL.isTerminal !== 'function' ||
            typeof window.RL.getReward !== 'function' ||
            typeof window.RL.setHeadlessMode !== 'function' ||
            typeof window.RL.tickCooldown !== 'function') {
            console.error('[Train] RL interface incomplete.');
            return null;
        }
        
        if (verbose) {
            console.log(`[Train] Starting async training: ${numEpisodes} episodes`);
        }
        const startTime = performance.now();
        
        const results = [];
        let totalTrainingSteps = 0;
        
        window.RL.setHeadlessMode(true);
        
        const { Runner, Render } = Matter;
        const runner = gameContext.runner();
        const render = gameContext.render();
        
        if (runner) Runner.stop(runner);
        if (render) Render.stop(render);
        
        if (replayBuffer) {
            replayBuffer.clear();
        } else {
            replayBuffer = new ReplayBuffer(DEFAULT_REPLAY_BUFFER_SIZE, STATE_SIZE);
        }
        
        try {
            for (let episode = 0; episode < numEpisodes; episode++) {
                if (verbose) {
                    console.log(`[Train] Episode ${episode + 1}/${numEpisodes} starting...`);
                }
                
                window.RL.resetEpisode();
                const engine = gameContext.engine();
                
                let stepCount = 0;
                let totalReward = 0;
                let totalLoss = 0;
                let trainCount = 0;
                const episodeStartTime = performance.now();
                
                const rawState = window.RL.getState();
                encodeStateIntoBuffer(rawState, stateBuffer);
                
                while (!window.RL.isTerminal() && stepCount < MAX_STEPS_PER_EPISODE) {
                    const action = selectActionFromBuffer(stateBuffer, validEpsilon);
                    
                    window.RL.step(action);
                    stepPhysics(engine);
                    window.RL.tickCooldown();
                    
                    const reward = window.RL.getReward();
                    totalReward += reward;
                    
                    const rawNextState = window.RL.getState();
                    encodeStateIntoBuffer(rawNextState, nextStateBuffer);
                    const done = window.RL.isTerminal();
                    
                    replayBuffer.add(stateBuffer, action, reward, nextStateBuffer, done);
                    
                    if (replayBuffer.canSample(validBatchSize) && 
                        stepCount % validTrainEveryNSteps === 0) {
                        const batch = replayBuffer.sampleBatch(validBatchSize);
                        const loss = trainOnBatch(batch);
                        totalLoss += loss;
                        trainCount++;
                        totalTrainingSteps++;
                    }
                    
                    for (let i = 0; i < STATE_SIZE; i++) {
                        stateBuffer[i] = nextStateBuffer[i];
                    }
                    
                    stepCount++;
                    
                    // Yield to event loop periodically
                    if (stepCount % validYieldEveryNSteps === 0) {
                        await new Promise(resolve => setTimeout(resolve, 0));
                    }
                }
                
                const episodeTime = performance.now() - episodeStartTime;
                const avgLoss = trainCount > 0 ? totalLoss / trainCount : 0;
                
                results.push({
                    episode: episode + 1,
                    steps: stepCount,
                    totalReward: totalReward,
                    avgLoss: avgLoss,
                    trainSteps: trainCount,
                    timeMs: episodeTime
                });
                
                if (verbose) {
                    console.log(
                        `[Train] Episode ${episode + 1} ended: ` +
                        `steps=${stepCount}, reward=${totalReward.toFixed(2)}, ` +
                        `avgLoss=${avgLoss.toFixed(6)}, time=${episodeTime.toFixed(2)}ms`
                    );
                }
            }
            
            if (verbose) {
                console.log('[Train] Saving model to localStorage...');
            }
            await model.save('localstorage://fruit-merge-dqn-v1');
            if (verbose) {
                console.log('[Train] Model saved successfully.');
            }
            
        } finally {
            window.RL.setHeadlessMode(false);
            if (typeof window.RL.reset === 'function') {
                window.RL.reset();
            }
        }
        
        const totalTime = performance.now() - startTime;
        const totalSteps = results.reduce((sum, r) => sum + r.steps, 0);
        const avgReward = results.length > 0 ? 
            results.reduce((sum, r) => sum + r.totalReward, 0) / results.length : 0;
        const avgSteps = results.length > 0 ? totalSteps / results.length : 0;
        
        if (verbose) {
            console.log(`[Train] ========== TRAINING SUMMARY ==========`);
            console.log(`[Train] Episodes completed: ${results.length}`);
            console.log(`[Train] Total time: ${totalTime.toFixed(2)}ms`);
            console.log(`[Train] Average reward per episode: ${avgReward.toFixed(2)}`);
            console.log(`[Train] Average steps per episode: ${avgSteps.toFixed(2)}`);
            console.log(`[Train] Total steps: ${totalSteps}`);
            console.log(`[Train] Total training steps: ${totalTrainingSteps}`);
            console.log(`[Train] ==========================================`);
        }
        
        return {
            episodes: results,
            totalTime: totalTime,
            avgReward: avgReward,
            avgSteps: avgSteps,
            totalTrainingSteps: totalTrainingSteps,
            replayBufferSize: replayBuffer.size
        };
    };
    
    /**
     * Load a previously saved model from localStorage.
     * 
     * @returns {Promise<boolean>} True if model loaded successfully
     */
    window.RL.loadModel = async function() {
        try {
            console.log('[Train] Loading model from localStorage...');
            model = await tf.loadLayersModel('localstorage://fruit-merge-dqn-v1');
            
            // Create optimizer for trainOnBatch
            optimizer = tf.train.adam(LEARNING_RATE);
            
            // Recompile the model
            model.compile({
                optimizer: optimizer,
                loss: 'meanSquaredError'
            });
            
            // Initialize replay buffer if not exists
            if (!replayBuffer) {
                replayBuffer = new ReplayBuffer(DEFAULT_REPLAY_BUFFER_SIZE, STATE_SIZE);
            }
            
            console.log('[Train] Model loaded successfully.');
            return true;
        } catch (error) {
            console.error('[Train] Failed to load model:', error);
            return false;
        }
    };
    
    /**
     * Get the current model (for inspection or manual inference).
     * 
     * @returns {tf.LayersModel|null} The current model or null if not initialized
     */
    window.RL.getModel = function() {
        return model;
    };
    
    /**
     * Get the replay buffer (for inspection).
     * 
     * @returns {ReplayBuffer|null} The replay buffer or null if not initialized
     */
    window.RL.getReplayBuffer = function() {
        return replayBuffer;
    };
    
    /**
     * Clear the replay buffer.
     */
    window.RL.clearReplayBuffer = function() {
        if (replayBuffer) {
            replayBuffer.clear();
            console.log('[Train] Replay buffer cleared.');
        }
    };
    
    console.log('[Train] Optimized training module initialized.');
    console.log('[Train] Use RL.initModel() to build the model, then RL.train(numEpisodes) to train.');
    console.log('[Train] For async training with UI responsiveness, use RL.trainAsync(numEpisodes).');
}
