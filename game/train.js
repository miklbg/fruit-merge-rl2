/**
 * train.js - Optimized TensorFlow.js Training Loop for Fruit Merge RL
 * 
 * High-performance DQN training implementation with:
 * - Pre-allocated state buffers (reduces GC pressure)
 * - Batched predictions for action selection and Q-target computation
 * - Fixed-size ring buffer replay with O(1) add and sample operations
 * - Target network for stable Q-learning (soft updates with τ parameter)
 * - Epsilon-greedy exploration with dynamic decay
 * - Batched gradient updates using tf.variableGrads() and optimizer.applyGradients()
 * - tf.tidy() wrapping for automatic memory management
 * - Async training loop with periodic yields (prevents browser freeze)
 * - Completely headless training (no DOM/rendering)
 * 
 * Stability improvements:
 * - Gradient clipping to prevent exploding gradients
 * - Huber loss for robust training (less sensitive to outliers)
 * - Soft target network updates (Polyak averaging) for smoother learning
 * - Reward clipping to bound reward magnitudes
 * - TD error clamping for stable priority updates
 * - Q-value clipping to prevent numerical instability
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
 *   RL.updateTargetModel()   - Copy weights from main to target model (soft update)
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
/**
 * Rank-based prioritized replay buffer constants.
 * α (alpha) controls how much prioritization is used (0 = uniform, 1 = full prioritization)
 * β (beta) controls importance-sampling correction (0 = no correction, 1 = full correction)
 */
const PRIORITY_ALPHA = 0.6;  // Reduced from 0.7 for more uniform sampling (stability)
const PRIORITY_BETA = 0.4;   // Reduced from 0.5 for less aggressive correction initially

/**
 * Stability constants for training.
 */
const MAX_TD_ERROR = 100.0;       // Maximum TD error for priority clamping
const MIN_TD_ERROR = 1e-6;        // Minimum TD error to prevent zero priority
const MAX_Q_VALUE = 1000.0;       // Maximum Q-value for clipping
const GRADIENT_CLIP_NORM = 10.0;  // Maximum gradient norm for clipping
const TAU = 0.005;                // Soft update coefficient for target network (Polyak averaging)
const HUBER_DELTA = 1.0;          // Delta parameter for Huber loss
const REWARD_CLIP_MIN = -10.0;    // Minimum reward after normalization
const REWARD_CLIP_MAX = 10.0;     // Maximum reward after normalization

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
        
        // TD errors for prioritized replay (absolute TD error per transition)
        // Initialize with high priority so new transitions get sampled
        this.tdErrors = new Float32Array(capacity);
        for (let i = 0; i < capacity; i++) {
            this.tdErrors[i] = 1.0; // Default high priority for new transitions
        }
        
        // Pre-allocate batch index array for sampling (reused across calls)
        this._batchIndices = new Uint32Array(capacity);
        
        // Pre-allocate arrays for sorted indices (used in prioritized sampling)
        this._sortedIndices = new Uint32Array(capacity);
        this._sortBuffer = new Float32Array(capacity);
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
        
        // New transitions get max priority (will be updated after first training)
        // Find the max TD error in the buffer for initialization
        let maxError = 1.0;
        for (let i = 0; i < this._size; i++) {
            if (this.tdErrors[i] > maxError) {
                maxError = this.tdErrors[i];
            }
        }
        this.tdErrors[this.position] = maxError;
        
        this.position = (this.position + 1) % this.capacity;
        if (this._size < this.capacity) {
            this._size++;
        }
    }
    
    /**
     * Update TD errors for specific indices after training.
     * Clamps TD errors to prevent extreme priority values.
     * @param {Uint32Array|number[]} indices - Buffer indices to update
     * @param {Float32Array|number[]} newErrors - New absolute TD errors
     */
    updateTDErrors(indices, newErrors) {
        const len = Math.min(indices.length, newErrors.length);
        for (let i = 0; i < len; i++) {
            const idx = indices[i];
            if (idx < this._size) {
                // Clamp TD error to prevent extreme priority values (stability improvement)
                const clampedError = Math.min(Math.max(Math.abs(newErrors[i]), MIN_TD_ERROR), MAX_TD_ERROR);
                this.tdErrors[idx] = clampedError;
            }
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
     * Sample a random minibatch of transitions (uniform sampling).
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
     * Sample a minibatch using rank-based prioritized replay.
     * Transitions with higher TD errors are more likely to be sampled.
     * Returns importance sampling weights for bias correction.
     * 
     * Sampling probability: P(i) = 1 / rank(i)^α where rank is based on TD error
     * Importance weight: w(i) = (N * P(i))^(-β) / max(w)
     * 
     * @param {number} batchSize - Number of transitions to sample
     * @returns {{states: Float32Array, actions: Uint8Array, rewards: Float32Array, nextStates: Float32Array, dones: Uint8Array, indices: Uint32Array, weights: Float32Array, actualBatchSize: number, meanTDError: number}}
     */
    getPrioritizedBatch(batchSize) {
        const actualBatchSize = Math.min(batchSize, this._size);
        
        // Step 1: Create indices and sort by TD error (descending)
        // Copy TD errors and indices for sorting
        for (let i = 0; i < this._size; i++) {
            this._sortedIndices[i] = i;
            this._sortBuffer[i] = this.tdErrors[i];
        }
        
        // Sort indices by TD error (descending order - highest error first)
        const indices = Array.from(this._sortedIndices.subarray(0, this._size));
        const errors = this._sortBuffer;
        indices.sort((a, b) => errors[b] - errors[a]);
        
        // Step 2: Compute rank-based probabilities
        // P(i) = 1 / rank(i)^α, where rank starts at 1
        const probabilities = new Float32Array(this._size);
        let sumProb = 0;
        for (let rank = 1; rank <= this._size; rank++) {
            const prob = 1.0 / Math.pow(rank, PRIORITY_ALPHA);
            probabilities[rank - 1] = prob;
            sumProb += prob;
        }
        
        // Normalize probabilities
        for (let i = 0; i < this._size; i++) {
            probabilities[i] /= sumProb;
        }
        
        // Step 3: Sample according to probabilities
        const batchStates = new Float32Array(actualBatchSize * this.stateSize);
        const batchActions = new Uint8Array(actualBatchSize);
        const batchRewards = new Float32Array(actualBatchSize);
        const batchNextStates = new Float32Array(actualBatchSize * this.stateSize);
        const batchDones = new Uint8Array(actualBatchSize);
        const weights = new Float32Array(actualBatchSize);
        const sampledIndices = new Uint32Array(actualBatchSize);
        
        // Build cumulative distribution for sampling
        const cumulative = new Float32Array(this._size);
        cumulative[0] = probabilities[0];
        for (let i = 1; i < this._size; i++) {
            cumulative[i] = cumulative[i - 1] + probabilities[i];
        }
        
        // Track TD errors for logging
        let totalTDError = 0;
        
        // Sample using inverse transform sampling
        for (let i = 0; i < actualBatchSize; i++) {
            const r = Math.random();
            
            // Binary search to find the index
            let low = 0;
            let high = this._size - 1;
            while (low < high) {
                const mid = Math.floor((low + high) / 2);
                if (cumulative[mid] < r) {
                    low = mid + 1;
                } else {
                    high = mid;
                }
            }
            
            // low is now the rank-1 index in the sorted order
            const rank = low + 1;
            const bufferIdx = indices[low]; // Get actual buffer index
            
            sampledIndices[i] = bufferIdx;
            this._batchIndices[i] = bufferIdx;
            
            const srcOffset = bufferIdx * this.stateSize;
            const dstOffset = i * this.stateSize;
            
            // Copy state data
            for (let j = 0; j < this.stateSize; j++) {
                batchStates[dstOffset + j] = this.states[srcOffset + j];
                batchNextStates[dstOffset + j] = this.nextStates[srcOffset + j];
            }
            
            batchActions[i] = this.actions[bufferIdx];
            batchRewards[i] = this.rewards[bufferIdx];
            batchDones[i] = this.dones[bufferIdx];
            
            // Compute importance sampling weight
            // w(i) = (N * P(i))^(-β)
            const prob = probabilities[low];
            weights[i] = Math.pow(this._size * prob, -PRIORITY_BETA);
            totalTDError += this.tdErrors[bufferIdx];
        }
        
        // Normalize weights by max weight for stability
        let maxWeight = 0;
        for (let i = 0; i < actualBatchSize; i++) {
            if (weights[i] > maxWeight) {
                maxWeight = weights[i];
            }
        }
        if (maxWeight > 0) {
            for (let i = 0; i < actualBatchSize; i++) {
                weights[i] /= maxWeight;
            }
        }
        
        const meanTDError = totalTDError / actualBatchSize;
        
        return {
            states: batchStates,
            actions: batchActions,
            rewards: batchRewards,
            nextStates: batchNextStates,
            dones: batchDones,
            indices: sampledIndices,
            weights: weights,
            actualBatchSize,
            meanTDError
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
        // Reset TD errors
        for (let i = 0; i < this.capacity; i++) {
            this.tdErrors[i] = 1.0;
        }
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
    const HIDDEN_UNITS = 64; // Hidden layer units (increased for better representation)
    const LEARNING_RATE = 0.0005; // Adam optimizer learning rate (reduced for stability)
    const DEFAULT_GAMMA = 0.99;  // Discount factor for Q-learning (default)
    
    // Training configuration
    const DEFAULT_BATCH_SIZE = 64;
    const DEFAULT_REPLAY_BUFFER_SIZE = 10000;
    const DEFAULT_MIN_BUFFER_SIZE = 500; // Increased minimum samples before training starts (stability)
    const DEFAULT_TRAIN_EVERY_N_STEPS = 4;
    const DEFAULT_TARGET_UPDATE_EVERY = 1; // Update target model every training step (soft updates)
    const USE_SOFT_UPDATE = true; // Use soft updates (Polyak averaging) instead of hard updates
    
    // Epsilon-greedy defaults
    const DEFAULT_EPSILON = 0.1;
    const DEFAULT_EPSILON_START = 1.0;
    const DEFAULT_EPSILON_END = 0.01;
    const DEFAULT_EPSILON_DECAY = 0.995;
    
    // Physics timestep (60 FPS equivalent)
    const DELTA_TIME = 1000 / 60;
    
    // Maximum steps per episode to prevent infinite loops
    const MAX_STEPS_PER_EPISODE = 10000;
    
    // Reward shaping constants (scaled down for stability)
    const REWARD_MERGE = 1.0;           // +1 for every fruit merge (scaled down)
    const REWARD_LARGE_FRUIT = 5.0;     // +5 for creating large/rare fruit (level >= 5)
    const REWARD_STEP_PENALTY = -0.01;  // -0.01 penalty per step to encourage faster play
    const REWARD_FRUIT_DROP = 1.0;      // +1 for each fruit dropped
    const REWARD_GAME_OVER = -10.0;     // -10 on game over (scaled down)
    const LARGE_FRUIT_THRESHOLD = 6;    // Fruit level 6 or higher is considered "large"
    
    // Model references
    let model = null;
    let targetModel = null;  // Target network for stable Q-learning
    let optimizer = null;
    
    // Track previous score for reward shaping
    let previousScore = 0;
    let previousMaxFruitLevel = -1;
    
    /**
     * Compute shaped reward based on game events.
     * Never returns zero - always applies at least the step penalty.
     * 
     * Reward components:
     * - +1 for every fruit merge (detected via score increase)
     * - +5 for creating a large/rare fruit (level >= 5)
     * - +1 for each fruit dropped
     * - -0.01 step penalty to encourage faster play
     * - -10 on game over
     * 
     * @param {number} scoreDelta - Change in score since last step
     * @param {number} currentMaxFruit - Current maximum fruit level on board
     * @param {boolean} isGameOver - Whether the game just ended
     * @param {boolean} wasDrop - Whether the action was a fruit drop (action 3)
     * @returns {{shapedReward: number, components: {merge: number, largeFruit: number, fruitDrop: number, stepPenalty: number, gameOver: number}}}
     */
    function computeShapedReward(scoreDelta, currentMaxFruit, isGameOver, wasDrop) {
        const components = {
            merge: 0,
            largeFruit: 0,
            fruitDrop: 0,
            stepPenalty: REWARD_STEP_PENALTY, // Always apply step penalty
            gameOver: 0
        };
        
        // Detect merge: if score increased, a merge occurred
        if (scoreDelta > 0) {
            components.merge = REWARD_MERGE;
        }
        
        // Detect large fruit creation: if max fruit level increased to >= threshold
        if (currentMaxFruit >= LARGE_FRUIT_THRESHOLD && currentMaxFruit > previousMaxFruitLevel) {
            components.largeFruit = REWARD_LARGE_FRUIT;
        }
        
        // Fruit drop reward: +1 for each fruit dropped
        if (wasDrop) {
            components.fruitDrop = REWARD_FRUIT_DROP;
        }
        
        // Game over penalty
        if (isGameOver) {
            components.gameOver = REWARD_GAME_OVER;
        }
        
        // Update tracking variables
        previousMaxFruitLevel = currentMaxFruit;
        
        // Total shaped reward (never zero - at minimum we have step penalty)
        const shapedReward = components.merge + components.largeFruit + components.fruitDrop + components.stepPenalty + components.gameOver;
        
        return { shapedReward, components };
    }
    
    /**
     * Reset reward shaping state for a new episode.
     */
    function resetRewardShaping() {
        previousScore = 0;
        previousMaxFruitLevel = -1;
    }
    
    /**
     * Get the maximum fruit level currently on the board from state.
     * State format: [currentX, currentY, currentFruit, nextFruit, booster, ...boardFruits]
     * Board fruits are at positions 5+, with 3 values each (x, y, level)
     * 
     * @param {Float32Array|number[]} state - The state array
     * @returns {number} Maximum fruit level on board (0-9 normalized to 0-1, need to convert back)
     */
    function getMaxFruitLevelFromState(state) {
        let maxLevel = -1;
        const MAX_FRUIT_LEVEL = 9; // Watermelon is level 9
        
        // Board fruits start at index 5, each has 3 values (x, y, normalizedLevel)
        for (let i = 5; i + 2 < state.length; i += 3) {
            const normalizedLevel = state[i + 2];
            if (normalizedLevel > 0) {
                // Convert normalized level (0-1) back to actual level (0-9)
                const level = Math.round(normalizedLevel * MAX_FRUIT_LEVEL);
                if (level > maxLevel) {
                    maxLevel = level;
                }
            }
        }
        
        return maxLevel;
    }
    
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
        
        // Initialize target model with same weights as main model (hard update for initialization)
        updateTargetModel(true);
        
        // Initialize replay buffer
        replayBuffer = new ReplayBuffer(DEFAULT_REPLAY_BUFFER_SIZE, STATE_SIZE);
        
        console.log('[Train] Model built and compiled successfully.');
        console.log('[Train] Target model initialized with same weights (hard update).');
        console.log('[Train] Stability features enabled: Huber loss, gradient clipping, soft target updates.');
        console.log('[Train] Model summary:');
        model.summary();
        
        return model;
    };
    
    /**
     * Update target model weights.
     * Supports both soft updates (Polyak averaging) and hard updates.
     * Soft updates: θ_target = τ * θ_main + (1 - τ) * θ_target
     * Hard updates: θ_target = θ_main
     * 
     * @param {boolean} hardUpdate - If true, perform hard update instead of soft update
     */
    function updateTargetModel(hardUpdate = false) {
        if (!model || !targetModel) {
            console.warn('[Train] Cannot update target model: models not initialized');
            return;
        }
        
        if (USE_SOFT_UPDATE && !hardUpdate) {
            // Soft update (Polyak averaging) for smoother, more stable learning
            // θ_target = τ * θ_main + (1 - τ) * θ_target
            tf.tidy(() => {
                const mainWeights = model.getWeights();
                const targetWeights = targetModel.getWeights();
                
                const newWeights = mainWeights.map((mainW, i) => {
                    const targetW = targetWeights[i];
                    // τ * θ_main + (1 - τ) * θ_target
                    return mainW.mul(TAU).add(targetW.mul(1 - TAU));
                });
                
                targetModel.setWeights(newWeights);
                
                // Dispose new weight tensors (setWeights copies the data internally)
                newWeights.forEach(w => w.dispose());
            });
        } else {
            // Hard update: directly copy weights
            const mainWeights = model.getWeights();
            const clonedWeights = mainWeights.map(w => w.clone());
            
            targetModel.setWeights(clonedWeights);
            
            // Dispose of the cloned tensors (setWeights copies the data internally)
            clonedWeights.forEach(w => w.dispose());
        }
    }
    
    /**
     * Expose updateTargetModel on the RL namespace.
     * @param {boolean} hardUpdate - If true, perform hard update instead of soft update
     */
    window.RL.updateTargetModel = function(hardUpdate = false) {
        updateTargetModel(hardUpdate);
        console.log(`[Train] Target model updated with ${hardUpdate || !USE_SOFT_UPDATE ? 'hard' : 'soft'} update (τ=${TAU}).`);
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
     * Compute Huber loss (smooth L1 loss) between predictions and targets.
     * More robust to outliers than MSE, providing stability in RL training.
     * 
     * Huber(x) = 0.5 * x^2           if |x| <= delta
     *          = delta * (|x| - 0.5 * delta)  otherwise
     * 
     * @param {tf.Tensor} predictions - Model predictions
     * @param {tf.Tensor} targets - Target values
     * @param {number} delta - Threshold for switching between quadratic and linear
     * @returns {tf.Tensor} Huber loss value
     */
    function huberLoss(predictions, targets, delta = HUBER_DELTA) {
        return tf.tidy(() => {
            const error = tf.sub(predictions, targets);
            const absError = tf.abs(error);
            const quadratic = tf.minimum(absError, delta);
            const linear = tf.sub(absError, quadratic);
            // 0.5 * quadratic^2 + delta * linear
            return tf.mean(tf.add(tf.mul(0.5, tf.square(quadratic)), tf.mul(delta, linear)));
        });
    }
    
    /**
     * Clip gradients by global norm to prevent exploding gradients.
     * 
     * @param {Object} grads - Object mapping variable names to gradients
     * @param {number} maxNorm - Maximum gradient norm
     * @returns {Object} Clipped gradients
     */
    function clipGradientsByNorm(grads, maxNorm) {
        return tf.tidy(() => {
            // Compute global norm
            const gradValues = Object.values(grads);
            const squaredNorms = gradValues.map(g => tf.sum(tf.square(g)));
            const globalNormSquared = tf.addN(squaredNorms);
            const globalNorm = tf.sqrt(globalNormSquared);
            
            // Compute clip coefficient: min(1, maxNorm / globalNorm)
            const clipCoef = tf.minimum(1.0, tf.div(maxNorm, tf.add(globalNorm, 1e-6)));
            
            // Clip each gradient
            const clippedGrads = {};
            for (const [name, grad] of Object.entries(grads)) {
                clippedGrads[name] = tf.mul(grad, clipCoef);
            }
            
            return clippedGrads;
        });
    }
    
    /**
     * Train on a minibatch using gradient descent with Double DQN.
     * Uses ONLINE model for action selection and TARGET model for action evaluation.
     * All Q-target computations run inside tf.tidy() for proper memory management.
     * 
     * Stability improvements:
     * - Huber loss instead of MSE for robustness to outliers
     * - Gradient clipping to prevent exploding gradients
     * - Q-value clipping to prevent numerical instability
     * - TD error clamping for stable priority updates
     * 
     * Double DQN target calculation:
     *   a* = argmax_a Q_online(nextState)[a]
     *   target = reward + gamma * Q_target(nextState)[a*] * (1 - done)
     * 
     * @param {Object} batch - Sampled batch from replay buffer
     * @param {number} gamma - Discount factor for Q-learning
     * @param {boolean} verbose - Whether to log Double-DQN updates
     * @returns {{loss: number, tdErrors: Float32Array}} Training loss and TD errors for priority update
     */
    function trainOnBatch(batch, gamma, verbose = false) {
        const { states, actions, rewards, nextStates, dones, weights, actualBatchSize } = batch;
        
        // Track TD errors for prioritized replay
        const tdErrors = new Float32Array(actualBatchSize);
        
        // Compute targets and TD errors using Double DQN inside tf.tidy()
        const { targets, currentQForActions } = tf.tidy(() => {
            const nextStatesTensor = tf.tensor2d(nextStates, [actualBatchSize, STATE_SIZE]);
            
            // DOUBLE DQN: Use ONLINE model to select actions
            const onlineNextQValues = model.predict(nextStatesTensor);
            const bestActions = onlineNextQValues.argMax(1).dataSync();
            
            // DOUBLE DQN: Use TARGET model to evaluate the selected actions
            // Clip target Q-values to prevent extreme values
            const targetNextQValues = targetModel.predict(nextStatesTensor);
            const targetNextQClipped = tf.clipByValue(targetNextQValues, -MAX_Q_VALUE, MAX_Q_VALUE);
            const targetNextQData = targetNextQClipped.arraySync();
            
            if (verbose) {
                console.log('[Train] Double-DQN: Using online model for action selection, target model for evaluation');
                console.log('[Train] Stability: Huber loss, gradient clipping, Q-value clipping enabled');
            }
            
            const statesTensor = tf.tensor2d(states, [actualBatchSize, STATE_SIZE]);
            const currentQValues = model.predict(statesTensor);
            const currentQData = currentQValues.arraySync();
            
            // Store current Q values for the taken actions (for TD error calculation)
            const qForActions = new Float32Array(actualBatchSize);
            for (let i = 0; i < actualBatchSize; i++) {
                qForActions[i] = currentQData[i][actions[i]];
            }
            
            // Compute target Q-values using Double DQN formula
            // target = reward + gamma * Q_target(s', argmax_a Q_online(s', a)) * (1 - done)
            for (let i = 0; i < actualBatchSize; i++) {
                const action = actions[i];
                // Clip reward for stability
                const reward = Math.min(Math.max(rewards[i], REWARD_CLIP_MIN), REWARD_CLIP_MAX);
                const done = dones[i];
                
                if (done) {
                    currentQData[i][action] = reward;
                } else {
                    // Double DQN: use online model's best action to index target model's Q-values
                    const bestAction = bestActions[i];
                    const targetQValue = targetNextQData[i][bestAction];
                    // Clip the computed target to prevent extreme values
                    currentQData[i][action] = Math.min(Math.max(reward + gamma * targetQValue, -MAX_Q_VALUE), MAX_Q_VALUE);
                }
            }
            
            return {
                targets: tf.tensor2d(currentQData),
                currentQForActions: qForActions
            };
        });
        
        // Compute gradients and apply using optimizer with gradient clipping
        const statesTensor = tf.tensor2d(states, [actualBatchSize, STATE_SIZE]);
        
        // Create weights tensor if we have importance sampling weights
        let weightsTensor = null;
        if (weights) {
            weightsTensor = tf.tensor1d(weights);
        }
        
        // Define loss function using Huber loss for stability
        const lossFunction = () => {
            const predictions = model.predict(statesTensor);
            
            if (weightsTensor) {
                // Weighted Huber loss for prioritized experience replay
                const error = tf.sub(predictions, targets);
                const absError = tf.abs(error);
                const quadratic = tf.minimum(absError, HUBER_DELTA);
                const linear = tf.sub(absError, quadratic);
                // Per-element Huber loss
                const elementLoss = tf.add(tf.mul(0.5, tf.square(quadratic)), tf.mul(HUBER_DELTA, linear));
                // Reduce to per-sample loss (mean over actions)
                const perSampleLoss = elementLoss.mean(1);
                // Weight by importance sampling weights
                const weightedLoss = perSampleLoss.mul(weightsTensor);
                return weightedLoss.mean();
            } else {
                // Unweighted Huber loss
                return huberLoss(predictions, targets, HUBER_DELTA);
            }
        };
        
        // Compute gradients
        const { value: loss, grads } = tf.variableGrads(lossFunction);
        
        // Clip gradients by global norm for stability
        const clippedGrads = clipGradientsByNorm(grads, GRADIENT_CLIP_NORM);
        
        // Apply clipped gradients
        optimizer.applyGradients(clippedGrads);
        
        const lossValue = loss.dataSync()[0];
        
        // Compute TD errors for priority update with clamping
        // TD error = |target - Q(s, a)|
        const targetData = targets.arraySync();
        for (let i = 0; i < actualBatchSize; i++) {
            const action = actions[i];
            const targetValue = targetData[i][action];
            const currentValue = currentQForActions[i];
            // Clamp TD error for stability
            tdErrors[i] = Math.min(Math.max(Math.abs(targetValue - currentValue), MIN_TD_ERROR), MAX_TD_ERROR);
        }
        
        // Clean up tensors
        targets.dispose();
        statesTensor.dispose();
        loss.dispose();
        Object.values(grads).forEach(g => g.dispose());
        Object.values(clippedGrads).forEach(g => g.dispose());
        if (weightsTensor) {
            weightsTensor.dispose();
        }
        
        return { loss: lossValue, tdErrors };
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
     * @param {number} [options.epsilonDecay=0.995] - Multiplicative decay factor per episode
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
                
                // Reset reward shaping state
                resetRewardShaping();
                
                // Get fresh engine reference after reset
                const engine = gameContext.engine();
                
                let stepCount = 0;
                let totalReward = 0;
                let totalLoss = 0;
                let trainCount = 0;
                let totalMeanTDError = 0; // Track mean TD error for logging
                const episodeStartTime = performance.now();
                const episodeStartEpsilon = currentEpsilon;
                
                // Track shaped reward components for logging
                let totalMergeReward = 0;
                let totalLargeFruitReward = 0;
                let totalFruitDropReward = 0;
                let totalStepPenalty = 0;
                let totalGameOverPenalty = 0;
                
                // Get initial state and encode into buffer
                const rawState = window.RL.getState();
                encodeStateIntoBuffer(rawState, stateBuffer);
                previousScore = gameContext.getScore();
                
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
                    
                    // Get current score and compute shaped reward
                    const currentScore = gameContext.getScore();
                    const scoreDelta = currentScore - previousScore;
                    previousScore = currentScore;
                    
                    // Get next state and check if terminal
                    const rawNextState = window.RL.getState();
                    encodeStateIntoBuffer(rawNextState, nextStateBuffer);
                    const done = window.RL.isTerminal();
                    
                    // Get max fruit level from state for reward shaping
                    const currentMaxFruit = getMaxFruitLevelFromState(nextStateBuffer);
                    
                    // Compute shaped reward (never zero) - action 3 is drop
                    const wasDrop = action === 3;
                    const { shapedReward, components } = computeShapedReward(scoreDelta, currentMaxFruit, done, wasDrop);
                    totalReward += shapedReward;
                    
                    // Track reward components for logging
                    totalMergeReward += components.merge;
                    totalLargeFruitReward += components.largeFruit;
                    totalFruitDropReward += components.fruitDrop;
                    totalStepPenalty += components.stepPenalty;
                    totalGameOverPenalty += components.gameOver;
                    
                    // Store transition in replay buffer with shaped reward
                    replayBuffer.add(stateBuffer, action, shapedReward, nextStateBuffer, done);
                    
                    // Update canTrain flag if we just crossed the threshold
                    if (!canTrain && replayBuffer.size() >= validMinBufferSize) {
                        canTrain = true;
                    }
                    
                    // Train on minibatch if enough samples and at training interval
                    if (canTrain && stepCount % validTrainEveryNSteps === 0) {
                        // Use prioritized replay sampling
                        const batch = replayBuffer.getPrioritizedBatch(validBatchSize);
                        const { loss, tdErrors } = trainOnBatch(batch, validGamma, verbose && trainCount === 0);
                        
                        // Update TD errors for sampled transitions
                        replayBuffer.updateTDErrors(batch.indices, tdErrors);
                        
                        totalLoss += loss;
                        totalMeanTDError += batch.meanTDError;
                        trainCount++;
                        totalTrainingSteps++;
                        
                        // Update target model periodically
                        if (totalTrainingSteps % validTargetUpdateEvery === 0) {
                            updateTargetModel();
                            if (verbose) {
                                //console.log(`[Train] Double-DQN target model updated at training step ${totalTrainingSteps}`);
                            }
                        }
                    }
                    
                    // Copy next state to current state buffer (avoid array allocation)
                    for (let i = 0; i < STATE_SIZE; i++) {
                        stateBuffer[i] = nextStateBuffer[i];
                    }
                    
                    stepCount++;
                    totalStepsAllEpisodes++;
                    
                    // Yield to event loop periodically to prevent browser freeze
                    if (stepCount % validYieldEveryNSteps === 0) {
                        await new Promise(resolve => setTimeout(resolve, 0));
                    }
                }
                
                // Decay epsilon once per episode (if using dynamic epsilon and buffer has enough samples)
                if (useDynamicEpsilon && replayBuffer.size() >= validMinBufferSize) {
                    currentEpsilon = Math.max(validEpsilonEnd, currentEpsilon * validEpsilonDecay);
                }
                
                const episodeTime = performance.now() - episodeStartTime;
                const avgLoss = trainCount > 0 ? totalLoss / trainCount : 0;
                const avgMeanTDError = trainCount > 0 ? totalMeanTDError / trainCount : 0;
                
                const episodeResult = {
                    episode: episode + 1,
                    steps: stepCount,
                    reward: totalReward,
                    epsilon: episodeStartEpsilon,
                    avgLoss: avgLoss,
                    trainSteps: trainCount,
                    durationMs: episodeTime,
                    // Reward shaping components
                    rewardComponents: {
                        merge: totalMergeReward,
                        largeFruit: totalLargeFruitReward,
                        fruitDrop: totalFruitDropReward,
                        stepPenalty: totalStepPenalty,
                        gameOver: totalGameOverPenalty
                    },
                    avgMeanTDError: avgMeanTDError
                };
                
                results.push(episodeResult);
                
                if (verbose) {
                    console.log(
                        `[Train] Episode ${episode + 1} ended: ` +
                        `steps=${stepCount}, reward=${totalReward.toFixed(2)}, ` +
                        `epsilon=${episodeStartEpsilon.toFixed(4)}, avgLoss=${avgLoss.toFixed(6)}, ` +
                        `trainSteps=${trainCount}, duration=${episodeTime.toFixed(2)}ms, ` +
                        `nextEpsilon=${currentEpsilon.toFixed(4)}`
                    );
                    console.log(
                        `[Train] Reward components: ` +
                        `merge=${totalMergeReward.toFixed(0)}, largeFruit=${totalLargeFruitReward.toFixed(0)}, ` +
                        `fruitDrop=${totalFruitDropReward.toFixed(0)}, stepPenalty=${totalStepPenalty.toFixed(2)}, gameOver=${totalGameOverPenalty.toFixed(0)}`
                    );
                    if (trainCount > 0) {
                        console.log(`[Train] Mean TD error: ${avgMeanTDError.toFixed(4)}`);
                    }
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
                
                // Reset reward shaping state
                resetRewardShaping();
                
                const engine = gameContext.engine();
                
                let stepCount = 0;
                let totalReward = 0;
                let totalLoss = 0;
                let trainCount = 0;
                let totalMeanTDError = 0; // Track mean TD error for logging
                const episodeStartTime = performance.now();
                const episodeStartEpsilon = currentEpsilon;
                
                // Track shaped reward components for logging
                let totalMergeReward = 0;
                let totalLargeFruitReward = 0;
                let totalFruitDropReward = 0;
                let totalStepPenalty = 0;
                let totalGameOverPenalty = 0;
                
                const rawState = window.RL.getState();
                encodeStateIntoBuffer(rawState, stateBuffer);
                previousScore = gameContext.getScore();
                
                // Track when buffer has enough samples for training (optimization to avoid repeated size() calls)
                let canTrain = replayBuffer.size() >= validMinBufferSize;
                
                while (!window.RL.isTerminal() && stepCount < MAX_STEPS_PER_EPISODE) {
                    const action = selectActionFromBuffer(stateBuffer, currentEpsilon);
                    
                    window.RL.step(action);
                    stepPhysics(engine);
                    window.RL.tickCooldown();
                    
                    // Get current score and compute shaped reward
                    const currentScore = gameContext.getScore();
                    const scoreDelta = currentScore - previousScore;
                    previousScore = currentScore;
                    
                    // Get next state and check if terminal
                    const rawNextState = window.RL.getState();
                    encodeStateIntoBuffer(rawNextState, nextStateBuffer);
                    const done = window.RL.isTerminal();
                    
                    // Get max fruit level from state for reward shaping
                    const currentMaxFruit = getMaxFruitLevelFromState(nextStateBuffer);
                    
                    // Compute shaped reward (never zero) - action 3 is drop
                    const wasDrop = action === 3;
                    const { shapedReward, components } = computeShapedReward(scoreDelta, currentMaxFruit, done, wasDrop);
                    totalReward += shapedReward;
                    
                    // Track reward components for logging
                    totalMergeReward += components.merge;
                    totalLargeFruitReward += components.largeFruit;
                    totalFruitDropReward += components.fruitDrop;
                    totalStepPenalty += components.stepPenalty;
                    totalGameOverPenalty += components.gameOver;
                    
                    // Store transition with shaped reward
                    replayBuffer.add(stateBuffer, action, shapedReward, nextStateBuffer, done);
                    
                    // Update canTrain flag if we just crossed the threshold
                    if (!canTrain && replayBuffer.size() >= validMinBufferSize) {
                        canTrain = true;
                    }
                    
                    // Train on minibatch if enough samples and at training interval
                    if (canTrain && stepCount % validTrainEveryNSteps === 0) {
                        // Use prioritized replay sampling
                        const batch = replayBuffer.getPrioritizedBatch(validBatchSize);
                        const { loss, tdErrors } = trainOnBatch(batch, validGamma, verbose && trainCount === 0);
                        
                        // Update TD errors for sampled transitions
                        replayBuffer.updateTDErrors(batch.indices, tdErrors);
                        
                        totalLoss += loss;
                        totalMeanTDError += batch.meanTDError;
                        trainCount++;
                        totalTrainingSteps++;
                        
                        // Update target model periodically
                        if (totalTrainingSteps % validTargetUpdateEvery === 0) {
                            updateTargetModel();
                            if (verbose) {
                                console.log(`[Train] Double-DQN target model updated at training step ${totalTrainingSteps}`);
                            }
                        }
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
                
                // Decay epsilon once per episode (if using dynamic epsilon and buffer has enough samples)
                if (useDynamicEpsilon && replayBuffer.size() >= validMinBufferSize) {
                    currentEpsilon = Math.max(validEpsilonEnd, currentEpsilon * validEpsilonDecay);
                }
                
                const episodeTime = performance.now() - episodeStartTime;
                const avgLoss = trainCount > 0 ? totalLoss / trainCount : 0;
                const avgMeanTDError = trainCount > 0 ? totalMeanTDError / trainCount : 0;
                
                results.push({
                    episode: episode + 1,
                    steps: stepCount,
                    reward: totalReward,
                    epsilon: episodeStartEpsilon,
                    avgLoss: avgLoss,
                    trainSteps: trainCount,
                    durationMs: episodeTime,
                    // Reward shaping components
                    rewardComponents: {
                        merge: totalMergeReward,
                        largeFruit: totalLargeFruitReward,
                        fruitDrop: totalFruitDropReward,
                        stepPenalty: totalStepPenalty,
                        gameOver: totalGameOverPenalty
                    },
                    avgMeanTDError: avgMeanTDError
                });
                
                if (verbose) {
                    console.log(
                        `[Train] Episode ${episode + 1} ended: ` +
                        `steps=${stepCount}, reward=${totalReward.toFixed(2)}, ` +
                        `epsilon=${episodeStartEpsilon.toFixed(4)}, avgLoss=${avgLoss.toFixed(6)}, ` +
                        `trainSteps=${trainCount}, duration=${episodeTime.toFixed(2)}ms, ` +
                        `nextEpsilon=${currentEpsilon.toFixed(4)}`
                    );
                    console.log(
                        `[Train] Reward components: ` +
                        `merge=${totalMergeReward.toFixed(0)}, largeFruit=${totalLargeFruitReward.toFixed(0)}, ` +
                        `fruitDrop=${totalFruitDropReward.toFixed(0)}, stepPenalty=${totalStepPenalty.toFixed(2)}, gameOver=${totalGameOverPenalty.toFixed(0)}`
                    );
                    if (trainCount > 0) {
                        console.log(`[Train] Mean TD error: ${avgMeanTDError.toFixed(4)}`);
                    }
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
            // Use hard update when loading to ensure exact weight copy
            updateTargetModel(true);
            
            // Initialize replay buffer if not exists
            if (!replayBuffer) {
                replayBuffer = new ReplayBuffer(DEFAULT_REPLAY_BUFFER_SIZE, STATE_SIZE);
            }
            
            console.log('[Train] Model loaded successfully.');
            console.log('[Train] Target model initialized with loaded weights (hard update).');
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
    console.log('[Train] Features: Double DQN, rank-based prioritized replay (α=' + PRIORITY_ALPHA + ', β=' + PRIORITY_BETA + '), reward shaping.');
    console.log('[Train] Stability: Huber loss (δ=' + HUBER_DELTA + '), gradient clipping (max=' + GRADIENT_CLIP_NORM + '), soft target updates (τ=' + TAU + ').');
    console.log('[Train] Reward shaping: +' + REWARD_MERGE + ' merge, +' + REWARD_LARGE_FRUIT + ' large fruit, +' + REWARD_FRUIT_DROP + ' fruit drop, ' + REWARD_STEP_PENALTY + ' step penalty, ' + REWARD_GAME_OVER + ' game over.');
}
