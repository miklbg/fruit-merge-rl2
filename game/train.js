/**
 * train.js - Optimized TensorFlow.js Training Loop for Fruit Merge RL
 * 
 * High-performance DQN training implementation with:
 * - Pre-allocated state buffers (reduces GC pressure)
 * - Batched predictions for action selection and Q-target computation
 * - Fixed-size ring buffer replay with O(1) add and sample operations
 * - Target network for stable Q-learning (updated periodically)
 * - Epsilon-greedy exploration with dynamic decay
 * - Batched gradient updates using tf.variableGrads() and optimizer.applyGradients()
 * - tf.tidy() wrapping for automatic memory management
 * - Async training loop with periodic yields (prevents browser freeze)
 * - Completely headless training (no DOM/rendering)
 * 
 * Usage:
 *   RL.initModel();                        // Build and compile main + target models
 *   await RL.train(5);                     // Run 5 episodes of training
 *   await RL.train(5, { batchSize: 64 });  // Use batch size of 64
 *   await RL.train(5, { epsilon: 0.1 });   // Use 10% fixed exploration
 *   await RL.train(5, {                    // Use dynamic epsilon decay
 *     epsilonStart: 1.0,
 *     epsilonEnd: 0.01,
 *     epsilonDecay: 0.995
 *   });
 * 
 * API:
 *   RL.initModel()           - Build main and target models
 *   RL.train(n, opts)        - Train for n episodes
 *   RL.trainAsync(n, opts)   - Same as train, for UI compatibility
 *   RL.selectAction(s, eps)  - Select action using epsilon-greedy policy
 *   RL.updateTargetModel()   - Copy weights from main to target model
 *   RL.getReplayBuffer()     - Get the replay buffer instance
 *   RL.clearReplayBuffer()   - Clear the replay buffer
 *   RL.getModel()            - Get the main model
 *   RL.getTargetModel()      - Get the target model
 * 
 * @module train
 */

// Game context reference (set by initTraining)
let gameContext = null;

/**
 * Fixed-size ring buffer for experience replay.
 * Stores transitions (s, a, r, s', done) with O(1) add and sample operations.
 * Uses pre-allocated typed arrays to minimize GC pressure.
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
        this._size = 0;
        
        // Pre-allocate typed arrays for all data
        this.states = new Float32Array(capacity * stateSize);
        this.actions = new Uint8Array(capacity);
        this.rewards = new Float32Array(capacity);
        this.nextStates = new Float32Array(capacity * stateSize);
        this.dones = new Uint8Array(capacity);
        
        // Pre-allocate batch index array for sampling (reused across calls)
        this._batchIndices = new Uint32Array(capacity);
    }
    
    /**
     * Add a transition to the buffer.
     * O(1) insert using index = count % capacity.
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
        if (this._size < this.capacity) {
            this._size++;
        }
    }
    
    /**
     * Get the current number of transitions in the buffer.
     * @returns {number}
     */
    size() {
        return this._size;
    }
    
    /**
     * Sample a random minibatch of transitions.
     * Returns typed arrays for direct tensor creation.
     * Uses pre-allocated arrays where possible to minimize allocations.
     * 
     * Note: Uses sampling WITH replacement for O(batchSize) complexity.
     * Sampling without replacement would require O(batchSize^2) or additional
     * data structures. The impact on training quality is minimal for typical
     * buffer sizes (10k+) and batch sizes (64).
     * 
     * @param {number} batchSize - Number of transitions to sample
     * @returns {{states: Float32Array, actions: Uint8Array, rewards: Float32Array, nextStates: Float32Array, dones: Uint8Array, indices: Uint32Array, actualBatchSize: number}}
     */
    sampleBatch(batchSize) {
        const actualBatchSize = Math.min(batchSize, this._size);
        
        // Pre-allocated output arrays (these need to be created for tensor creation)
        const batchStates = new Float32Array(actualBatchSize * this.stateSize);
        const batchActions = new Uint8Array(actualBatchSize);
        const batchRewards = new Float32Array(actualBatchSize);
        const batchNextStates = new Float32Array(actualBatchSize * this.stateSize);
        const batchDones = new Uint8Array(actualBatchSize);
        
        // Random sampling with replacement (O(batchSize) complexity)
        for (let i = 0; i < actualBatchSize; i++) {
            const idx = Math.floor(Math.random() * this._size);
            this._batchIndices[i] = idx;
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
            indices: this._batchIndices.subarray(0, actualBatchSize),
            actualBatchSize
        };
    }
    
    /**
     * Check if buffer has enough samples for a batch.
     * @param {number} batchSize - Desired batch size
     * @returns {boolean}
     */
    canSample(batchSize) {
        return this._size >= batchSize;
    }
    
    /**
     * Clear the buffer.
     */
    clear() {
        this.position = 0;
        this._size = 0;
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
    const DEFAULT_GAMMA = 0.99;  // Discount factor for Q-learning (default)
    
    // Training configuration
    const DEFAULT_BATCH_SIZE = 64;
    const DEFAULT_REPLAY_BUFFER_SIZE = 10000;
    const DEFAULT_MIN_BUFFER_SIZE = 100; // Minimum samples before training starts
    const DEFAULT_TRAIN_EVERY_N_STEPS = 4;
    const DEFAULT_TARGET_UPDATE_EVERY = 1000; // Update target model every N training steps
    
    // Epsilon-greedy defaults
    const DEFAULT_EPSILON = 0.1;
    const DEFAULT_EPSILON_START = 1.0;
    const DEFAULT_EPSILON_END = 0.01;
    const DEFAULT_EPSILON_DECAY = 0.995;
    
    // Physics timestep (60 FPS equivalent)
    const DELTA_TIME = 1000 / 60;
    
    // Maximum steps per episode to prevent infinite loops
    const MAX_STEPS_PER_EPISODE = 10000;
    
    // Model references
    let model = null;
    let targetModel = null;  // Target network for stable Q-learning
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
     * Create a Q-network model with the standard architecture.
     * Used for both main model and target model.
     * 
     * Architecture:
     * - Input: 155-element state vector
     * - Dense: 32 units, ReLU
     * - Dense: 32 units, ReLU
     * - Output: 4 units (Q-values for each action)
     * 
     * @returns {tf.LayersModel} The created model
     */
    function createQNetwork() {
        const network = tf.sequential();
        
        // First hidden layer with input shape
        network.add(tf.layers.dense({
            units: HIDDEN_UNITS,
            activation: 'relu',
            inputShape: [STATE_SIZE],
            kernelInitializer: 'heNormal'
        }));
        
        // Second hidden layer
        network.add(tf.layers.dense({
            units: HIDDEN_UNITS,
            activation: 'relu',
            kernelInitializer: 'heNormal'
        }));
        
        // Output layer (Q-values for each action)
        network.add(tf.layers.dense({
            units: NUM_ACTIONS,
            activation: 'linear',
            kernelInitializer: 'glorotNormal'
        }));
        
        return network;
    }
    
    /**
     * Build and compile the Q-network model.
     * Also creates the target network with the same architecture.
     * 
     * Optimizer: Adam(0.001)
     * Loss: Mean Squared Error (MSE)
     */
    window.RL.initModel = function() {
        console.log('[Train] Building Q-network model...');
        
        // Create main model
        model = createQNetwork();
        
        // Create optimizer for gradient updates
        optimizer = tf.train.adam(LEARNING_RATE);
        
        // Compile main model
        model.compile({
            optimizer: optimizer,
            loss: 'meanSquaredError'
        });
        
        // Create target model with same architecture
        targetModel = createQNetwork();
        
        // Compile target model (not used for training, but needed for predict)
        targetModel.compile({
            optimizer: tf.train.adam(LEARNING_RATE),
            loss: 'meanSquaredError'
        });
        
        // Initialize target model with same weights as main model
        updateTargetModel();
        
        // Initialize replay buffer
        replayBuffer = new ReplayBuffer(DEFAULT_REPLAY_BUFFER_SIZE, STATE_SIZE);
        
        console.log('[Train] Model built and compiled successfully.');
        console.log('[Train] Target model initialized with same weights.');
        console.log('[Train] Model summary:');
        model.summary();
        
        return model;
    };
    
    /**
     * Copy weights from main model to target model.
     * This provides stable Q-value targets for training.
     */
    function updateTargetModel() {
        if (!model || !targetModel) {
            console.warn('[Train] Cannot update target model: models not initialized');
            return;
        }
        
        // Get weights from main model
        const mainWeights = model.getWeights();
        
        // Set weights on target model
        targetModel.setWeights(mainWeights);
        
        // Dispose of weight tensors to prevent memory leak
        mainWeights.forEach(w => w.dispose());
    }
    
    /**
     * Expose updateTargetModel on the RL namespace.
     */
    window.RL.updateTargetModel = function() {
        updateTargetModel();
        console.log('[Train] Target model updated with main model weights.');
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
     * Computes Q-targets using the TARGET MODEL and performs a single gradient update.
     * All tensor operations are wrapped in tf.tidy() where possible.
     * 
     * @param {Object} batch - Sampled batch from replay buffer
     * @param {number} gamma - Discount factor for Q-learning
     * @returns {number} Training loss
     */
    function trainOnBatch(batch, gamma) {
        const { states, actions, rewards, nextStates, dones, actualBatchSize } = batch;
        
        // Compute targets outside of gradient tape using TARGET MODEL
        const targets = tf.tidy(() => {
            const nextStatesTensor = tf.tensor2d(nextStates, [actualBatchSize, STATE_SIZE]);
            // Use TARGET MODEL for stable Q-value estimation
            const nextQValues = targetModel.predict(nextStatesTensor);
            const maxNextQ = nextQValues.max(1).dataSync();
            
            const statesTensor = tf.tensor2d(states, [actualBatchSize, STATE_SIZE]);
            const currentQValues = model.predict(statesTensor);
            const currentQData = currentQValues.arraySync();
            
            // Compute target Q-values
            // For each sample: target[action] = reward + gamma * max(Q_target(s', a')) * (1 - done)
            for (let i = 0; i < actualBatchSize; i++) {
                const action = actions[i];
                const reward = rewards[i];
                const done = dones[i];
                
                if (done) {
                    currentQData[i][action] = reward;
                } else {
                    currentQData[i][action] = reward + gamma * maxNextQ[i];
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
     * Uses experience replay with periodic yields to prevent browser freezing.
     * 
     * @param {number} numEpisodes - Number of episodes to train
     * @param {Object} [options={}] - Optional configuration options
     * @param {number} [options.gamma=0.99] - Discount factor for Q-learning
     * @param {number} [options.batchSize=64] - Minibatch size for training
     * @param {number} [options.trainEveryNSteps=4] - Train every N steps
     * @param {number} [options.minBufferSize=100] - Minimum buffer size before training starts
     * @param {number} [options.epsilon=0.1] - Fixed epsilon for exploration (ignored if epsilonStart is provided)
     * @param {number} [options.epsilonStart=1.0] - Starting epsilon for decay schedule
     * @param {number} [options.epsilonEnd=0.01] - Minimum epsilon after decay
     * @param {number} [options.epsilonDecay=0.995] - Multiplicative decay factor per step
     * @param {number} [options.targetUpdateEvery=1000] - Update target model every N training steps
     * @param {number} [options.yieldEveryNSteps=100] - Yield to event loop every N steps to prevent browser freeze
     * @param {boolean} [options.verbose=true] - Whether to log progress
     * @returns {Promise<Object>} Training results summary
     */
    window.RL.train = async function(numEpisodes, options = {}) {
        // Extract options with defaults
        const {
            gamma = DEFAULT_GAMMA,
            batchSize = DEFAULT_BATCH_SIZE,
            trainEveryNSteps = DEFAULT_TRAIN_EVERY_N_STEPS,
            minBufferSize = DEFAULT_MIN_BUFFER_SIZE,
            epsilon: fixedEpsilon,
            epsilonStart,
            epsilonEnd = DEFAULT_EPSILON_END,
            epsilonDecay = DEFAULT_EPSILON_DECAY,
            targetUpdateEvery = DEFAULT_TARGET_UPDATE_EVERY,
            yieldEveryNSteps = 100,
            verbose = true
        } = options;
        
        // Determine epsilon mode: dynamic decay or fixed
        const useDynamicEpsilon = epsilonStart !== undefined;
        let currentEpsilon = useDynamicEpsilon 
            ? Math.max(0, Math.min(1, epsilonStart))
            : Math.max(0, Math.min(1, fixedEpsilon !== undefined ? fixedEpsilon : DEFAULT_EPSILON));
        const validEpsilonEnd = Math.max(0, Math.min(1, epsilonEnd));
        const validEpsilonDecay = Math.max(0, Math.min(1, epsilonDecay));
        
        // Validate other parameters
        const validGamma = Math.max(0, Math.min(1, gamma));
        const validBatchSize = Math.max(1, Math.floor(batchSize));
        const validTrainEveryNSteps = Math.max(1, Math.floor(trainEveryNSteps));
        const validMinBufferSize = Math.max(1, Math.floor(minBufferSize));
        const validTargetUpdateEvery = Math.max(1, Math.floor(targetUpdateEvery));
        const validYieldEveryNSteps = Math.max(1, Math.floor(yieldEveryNSteps));
        
        // Validate model is initialized
        if (!model) {
            console.error('[Train] Model not initialized. Call RL.initModel() first.');
            return null;
        }
        
        // Validate target model is initialized
        if (!targetModel) {
            console.error('[Train] Target model not initialized. Call RL.initModel() first.');
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
            console.log(`[Train] Starting training: ${numEpisodes} episodes`);
            console.log(`[Train] Hyperparameters: gamma=${validGamma}, batchSize=${validBatchSize}, trainEveryNSteps=${validTrainEveryNSteps}`);
            console.log(`[Train] Epsilon: ${useDynamicEpsilon ? `dynamic (${currentEpsilon} -> ${validEpsilonEnd}, decay=${validEpsilonDecay})` : `fixed (${currentEpsilon})`}`);
            console.log(`[Train] minBufferSize=${validMinBufferSize}, targetUpdateEvery=${validTargetUpdateEvery}`);
        }
        const startTime = performance.now();
        
        const results = [];
        let totalTrainingSteps = 0;
        let totalStepsAllEpisodes = 0;
        
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
                const episodeStartEpsilon = currentEpsilon;
                
                // Get initial state and encode into buffer
                const rawState = window.RL.getState();
                encodeStateIntoBuffer(rawState, stateBuffer);
                
                // Track when buffer has enough samples for training (optimization to avoid repeated size() calls)
                let canTrain = replayBuffer.size() >= validMinBufferSize;
                
                // Episode loop with periodic yields to prevent browser freeze
                while (!window.RL.isTerminal() && stepCount < MAX_STEPS_PER_EPISODE) {
                    // Select action using epsilon-greedy policy
                    const action = selectActionFromBuffer(stateBuffer, currentEpsilon);
                    
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
                    
                    // Update canTrain flag if we just crossed the threshold
                    if (!canTrain && replayBuffer.size() >= validMinBufferSize) {
                        canTrain = true;
                    }
                    
                    // Train on minibatch if enough samples and at training interval
                    if (canTrain && stepCount % validTrainEveryNSteps === 0) {
                        const batch = replayBuffer.sampleBatch(validBatchSize);
                        const loss = trainOnBatch(batch, validGamma);
                        totalLoss += loss;
                        trainCount++;
                        totalTrainingSteps++;
                        
                        // Update target model periodically
                        if (totalTrainingSteps % validTargetUpdateEvery === 0) {
                            updateTargetModel();
                            if (verbose) {
                                console.log(`[Train] Target model updated at training step ${totalTrainingSteps}`);
                            }
                        }
                    }
                    
                    // Copy next state to current state buffer (avoid array allocation)
                    for (let i = 0; i < STATE_SIZE; i++) {
                        stateBuffer[i] = nextStateBuffer[i];
                    }
                    
                    stepCount++;
                    totalStepsAllEpisodes++;
                    
                    // Decay epsilon after each step (if using dynamic epsilon)
                    if (useDynamicEpsilon) {
                        currentEpsilon = Math.max(validEpsilonEnd, currentEpsilon * validEpsilonDecay);
                    }
                    
                    // Yield to event loop periodically to prevent browser freeze
                    if (stepCount % validYieldEveryNSteps === 0) {
                        await new Promise(resolve => setTimeout(resolve, 0));
                    }
                }
                
                const episodeTime = performance.now() - episodeStartTime;
                const avgLoss = trainCount > 0 ? totalLoss / trainCount : 0;
                
                const episodeResult = {
                    episode: episode + 1,
                    steps: stepCount,
                    reward: totalReward,
                    epsilon: episodeStartEpsilon,
                    avgLoss: avgLoss,
                    trainSteps: trainCount,
                    durationMs: episodeTime
                };
                
                results.push(episodeResult);
                
                if (verbose) {
                    console.log(
                        `[Train] Episode ${episode + 1} ended: ` +
                        `steps=${stepCount}, reward=${totalReward.toFixed(2)}, ` +
                        `epsilon=${episodeStartEpsilon.toFixed(4)}, avgLoss=${avgLoss.toFixed(6)}, ` +
                        `trainSteps=${trainCount}, duration=${episodeTime.toFixed(2)}ms`
                    );
                }
            }
            
            // Save model to localStorage
            if (verbose) {
                console.log('[Train] Saving model to localStorage...');
            }
            try {
                await model.save('localstorage://fruit-merge-dqn-v1');
                if (verbose) {
                    console.log('[Train] Model saved successfully to localstorage://fruit-merge-dqn-v1');
                }
            } catch (err) {
                console.error('[Train] Failed to save model:', err);
            }
            
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
            results.reduce((sum, r) => sum + r.reward, 0) / results.length : 0;
        const avgSteps = results.length > 0 ? totalSteps / results.length : 0;
        
        if (verbose) {
            console.log(`[Train] ========== TRAINING SUMMARY ==========`);
            console.log(`[Train] Episodes completed: ${results.length}`);
            console.log(`[Train] Total time: ${totalTime.toFixed(2)}ms`);
            console.log(`[Train] Average reward per episode: ${avgReward.toFixed(2)}`);
            console.log(`[Train] Average steps per episode: ${avgSteps.toFixed(2)}`);
            console.log(`[Train] Total steps: ${totalSteps}`);
            console.log(`[Train] Total training steps: ${totalTrainingSteps}`);
            console.log(`[Train] Final epsilon: ${currentEpsilon.toFixed(4)}`);
            console.log(`[Train] Replay buffer size: ${replayBuffer.size()}`);
            console.log(`[Train] ==========================================`);
        }
        
        return {
            episodes: results,
            totalTime: totalTime,
            avgReward: avgReward,
            avgSteps: avgSteps,
            totalTrainingSteps: totalTrainingSteps,
            finalEpsilon: currentEpsilon,
            replayBufferSize: replayBuffer.size()
        };
    };
    
    /**
     * Async version of train for compatibility with UI that needs event loop access.
     * Yields to the event loop periodically to prevent browser freezing.
     * Has the same parameters as RL.train().
     * 
     * @param {number} numEpisodes - Number of episodes to train
     * @param {Object} [options={}] - Optional configuration options (same as RL.train)
     * @returns {Promise<Object>} Training results summary
     */
    window.RL.trainAsync = async function(numEpisodes, options = {}) {
        // Extract options with defaults (same as train)
        const {
            gamma = DEFAULT_GAMMA,
            batchSize = DEFAULT_BATCH_SIZE,
            trainEveryNSteps = DEFAULT_TRAIN_EVERY_N_STEPS,
            minBufferSize = DEFAULT_MIN_BUFFER_SIZE,
            epsilon: fixedEpsilon,
            epsilonStart,
            epsilonEnd = DEFAULT_EPSILON_END,
            epsilonDecay = DEFAULT_EPSILON_DECAY,
            targetUpdateEvery = DEFAULT_TARGET_UPDATE_EVERY,
            yieldEveryNSteps = 100,
            verbose = true
        } = options;
        
        // Determine epsilon mode: dynamic decay or fixed
        const useDynamicEpsilon = epsilonStart !== undefined;
        let currentEpsilon = useDynamicEpsilon 
            ? Math.max(0, Math.min(1, epsilonStart))
            : Math.max(0, Math.min(1, fixedEpsilon !== undefined ? fixedEpsilon : DEFAULT_EPSILON));
        const validEpsilonEnd = Math.max(0, Math.min(1, epsilonEnd));
        const validEpsilonDecay = Math.max(0, Math.min(1, epsilonDecay));
        
        // Validate other parameters
        const validGamma = Math.max(0, Math.min(1, gamma));
        const validBatchSize = Math.max(1, Math.floor(batchSize));
        const validTrainEveryNSteps = Math.max(1, Math.floor(trainEveryNSteps));
        const validMinBufferSize = Math.max(1, Math.floor(minBufferSize));
        const validTargetUpdateEvery = Math.max(1, Math.floor(targetUpdateEvery));
        const validYieldEveryNSteps = Math.max(1, Math.floor(yieldEveryNSteps));
        
        if (!model) {
            console.error('[Train] Model not initialized. Call RL.initModel() first.');
            return null;
        }
        
        if (!targetModel) {
            console.error('[Train] Target model not initialized. Call RL.initModel() first.');
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
            console.log(`[Train] Hyperparameters: gamma=${validGamma}, batchSize=${validBatchSize}, trainEveryNSteps=${validTrainEveryNSteps}`);
            console.log(`[Train] Epsilon: ${useDynamicEpsilon ? `dynamic (${currentEpsilon} -> ${validEpsilonEnd}, decay=${validEpsilonDecay})` : `fixed (${currentEpsilon})`}`);
            console.log(`[Train] minBufferSize=${validMinBufferSize}, targetUpdateEvery=${validTargetUpdateEvery}`);
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
                const episodeStartEpsilon = currentEpsilon;
                
                const rawState = window.RL.getState();
                encodeStateIntoBuffer(rawState, stateBuffer);
                
                // Track when buffer has enough samples for training (optimization to avoid repeated size() calls)
                let canTrain = replayBuffer.size() >= validMinBufferSize;
                
                while (!window.RL.isTerminal() && stepCount < MAX_STEPS_PER_EPISODE) {
                    const action = selectActionFromBuffer(stateBuffer, currentEpsilon);
                    
                    window.RL.step(action);
                    stepPhysics(engine);
                    window.RL.tickCooldown();
                    
                    const reward = window.RL.getReward();
                    totalReward += reward;
                    
                    const rawNextState = window.RL.getState();
                    encodeStateIntoBuffer(rawNextState, nextStateBuffer);
                    const done = window.RL.isTerminal();
                    
                    replayBuffer.add(stateBuffer, action, reward, nextStateBuffer, done);
                    
                    // Update canTrain flag if we just crossed the threshold
                    if (!canTrain && replayBuffer.size() >= validMinBufferSize) {
                        canTrain = true;
                    }
                    
                    // Train on minibatch if enough samples and at training interval
                    if (canTrain && stepCount % validTrainEveryNSteps === 0) {
                        const batch = replayBuffer.sampleBatch(validBatchSize);
                        const loss = trainOnBatch(batch, validGamma);
                        totalLoss += loss;
                        trainCount++;
                        totalTrainingSteps++;
                        
                        // Update target model periodically
                        if (totalTrainingSteps % validTargetUpdateEvery === 0) {
                            updateTargetModel();
                            if (verbose) {
                                console.log(`[Train] Target model updated at training step ${totalTrainingSteps}`);
                            }
                        }
                    }
                    
                    for (let i = 0; i < STATE_SIZE; i++) {
                        stateBuffer[i] = nextStateBuffer[i];
                    }
                    
                    stepCount++;
                    
                    // Decay epsilon after each step (if using dynamic epsilon)
                    if (useDynamicEpsilon) {
                        currentEpsilon = Math.max(validEpsilonEnd, currentEpsilon * validEpsilonDecay);
                    }
                    
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
                    reward: totalReward,
                    epsilon: episodeStartEpsilon,
                    avgLoss: avgLoss,
                    trainSteps: trainCount,
                    durationMs: episodeTime
                });
                
                if (verbose) {
                    console.log(
                        `[Train] Episode ${episode + 1} ended: ` +
                        `steps=${stepCount}, reward=${totalReward.toFixed(2)}, ` +
                        `epsilon=${episodeStartEpsilon.toFixed(4)}, avgLoss=${avgLoss.toFixed(6)}, ` +
                        `trainSteps=${trainCount}, duration=${episodeTime.toFixed(2)}ms`
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
            results.reduce((sum, r) => sum + r.reward, 0) / results.length : 0;
        const avgSteps = results.length > 0 ? totalSteps / results.length : 0;
        
        if (verbose) {
            console.log(`[Train] ========== TRAINING SUMMARY ==========`);
            console.log(`[Train] Episodes completed: ${results.length}`);
            console.log(`[Train] Total time: ${totalTime.toFixed(2)}ms`);
            console.log(`[Train] Average reward per episode: ${avgReward.toFixed(2)}`);
            console.log(`[Train] Average steps per episode: ${avgSteps.toFixed(2)}`);
            console.log(`[Train] Total steps: ${totalSteps}`);
            console.log(`[Train] Total training steps: ${totalTrainingSteps}`);
            console.log(`[Train] Final epsilon: ${currentEpsilon.toFixed(4)}`);
            console.log(`[Train] Replay buffer size: ${replayBuffer.size()}`);
            console.log(`[Train] ==========================================`);
        }
        
        return {
            episodes: results,
            totalTime: totalTime,
            avgReward: avgReward,
            avgSteps: avgSteps,
            totalTrainingSteps: totalTrainingSteps,
            finalEpsilon: currentEpsilon,
            replayBufferSize: replayBuffer.size()
        };
    };
    
    /**
     * Load a previously saved model from localStorage.
     * Also creates and initializes the target model.
     * 
     * @returns {Promise<boolean>} True if model loaded successfully
     */
    window.RL.loadModel = async function() {
        try {
            console.log('[Train] Loading model from localStorage...');
            model = await tf.loadLayersModel('localstorage://fruit-merge-dqn-v1');
            
            // Create optimizer for gradient updates
            optimizer = tf.train.adam(LEARNING_RATE);
            
            // Recompile the model
            model.compile({
                optimizer: optimizer,
                loss: 'meanSquaredError'
            });
            
            // Create and initialize target model
            targetModel = createQNetwork();
            targetModel.compile({
                optimizer: tf.train.adam(LEARNING_RATE),
                loss: 'meanSquaredError'
            });
            updateTargetModel();
            
            // Initialize replay buffer if not exists
            if (!replayBuffer) {
                replayBuffer = new ReplayBuffer(DEFAULT_REPLAY_BUFFER_SIZE, STATE_SIZE);
            }
            
            console.log('[Train] Model loaded successfully.');
            console.log('[Train] Target model initialized with loaded weights.');
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
     * Get the target model (for inspection).
     * 
     * @returns {tf.LayersModel|null} The target model or null if not initialized
     */
    window.RL.getTargetModel = function() {
        return targetModel;
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
    
    /**
     * Select action using epsilon-greedy policy.
     * This is the public API for action selection.
     * 
     * @param {number[]|Float32Array} state - State vector (155 elements)
     * @param {number} [epsilon=0] - Exploration rate (0-1). With probability epsilon, returns random action.
     * @returns {number} Action index (0-3)
     */
    window.RL.selectAction = function(state, epsilon = 0) {
        if (!model) {
            console.error('[Train] Model not initialized. Call RL.initModel() first.');
            return Math.floor(Math.random() * NUM_ACTIONS);
        }
        
        // Copy state to buffer
        encodeStateIntoBuffer(state, stateBuffer);
        return selectActionFromBuffer(stateBuffer, epsilon);
    };
    
    console.log('[Train] Optimized training module initialized.');
    console.log('[Train] Use RL.initModel() to build the model, then await RL.train(numEpisodes) to train.');
    console.log('[Train] Both RL.train() and RL.trainAsync() yield to event loop to prevent browser freeze.');
    console.log('[Train] New features: target network, epsilon decay, minBufferSize, targetUpdateEvery.');
}
