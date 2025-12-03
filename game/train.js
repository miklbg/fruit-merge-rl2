/**
 * train.js - Optimized TensorFlow.js Training Loop for Fruit Merge RL
 * 
 * High-performance DQN training implementation with:
 * - Pre-allocated state buffers (reduces GC pressure)
 * - Batched predictions for action selection and Q-target computation
 * - Fixed-size ring buffer replay with O(1) add and sample operations
 * - Target network for stable Q-learning (soft updates with τ parameter)
 * - NoisyNet exploration with factorized Gaussian noise (replaces epsilon-greedy)
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
 *   // Note: epsilon parameters are ignored - NoisyNet provides exploration
 * 
 * API:
 *   RL.initModel()           - Build main and target models with NoisyNet layers
 *   RL.train(n, opts)        - Train for n episodes
 *   RL.trainAsync(n, opts)   - Same as train, for UI compatibility
 *   RL.selectAction(s)       - Select action using noisy Q-values (epsilon ignored)
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
const PRIORITY_ALPHA = 0.7;  // Prioritization strength for prioritized experience replay
const PRIORITY_BETA_START = 0.4;   // Starting beta for importance sampling (was constant 0.4)
const PRIORITY_BETA_END = 1.0;     // Final beta - full correction for later training
const PRIORITY_BETA_EPISODES = 200; // Number of episodes to anneal beta from start to end

/**
 * Stability constants for training.
 */
const MAX_TD_ERROR = 100.0;       // Maximum TD error for priority clamping
const MIN_TD_ERROR = 1e-6;        // Minimum TD error to prevent zero priority
const MAX_Q_VALUE = 1000.0;       // Maximum Q-value for clipping
const GRADIENT_CLIP_NORM = 10.0;  // Maximum gradient norm for clipping
const TAU = 0.001;                // Soft update coefficient for target network (changed to 0.001)
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
     * @param {number} beta - Importance sampling exponent (0 = no correction, 1 = full correction)
     * @returns {{states: Float32Array, actions: Uint8Array, rewards: Float32Array, nextStates: Float32Array, dones: Uint8Array, indices: Uint32Array, weights: Float32Array, actualBatchSize: number, meanTDError: number}}
     */
    getPrioritizedBatch(batchSize, beta = 0.4) {
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
            weights[i] = Math.pow(this._size * prob, -beta);
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
    const BASE_STATE_SIZE = 155;  // 155-element base state vector (legacy)
    const FRAME_STACK_SIZE = 3;   // Number of frames to stack
    const STATE_SIZE = BASE_STATE_SIZE * FRAME_STACK_SIZE;  // 155 * 3 = 465-element stacked state vector
    const NUM_ACTIONS = 10;  // 10 discrete actions (column 0-9)
    
    // Spatial grid configuration for CNN with embedding
    const GRID_WIDTH = 10;   // Grid width for spatial representation
    const GRID_HEIGHT = 15;  // Grid height for spatial representation
    const MAX_ACTUAL_FRUIT_LEVEL = 9; // Maximum fruit level value (0-9, watermelon is 9)
    const MAX_FRUIT_LEVEL = 10; // Maximum fruit level (0-9) + empty (10)
    const EMBEDDING_DIM = 16; // Embedding dimension for fruit types
    
    // Additional input features (concatenated after CNN)
    const ADDITIONAL_FEATURES = 4; // [currentX, currentY, currentFruitType, nextFruitType]
    
    // CNN architecture configuration - NEW DEEPER ARCHITECTURE
    const CNN_FILTERS_1 = 32;   // First conv layer filters
    const CNN_FILTERS_2 = 64;   // Second conv layer filters
    const CNN_FILTERS_3 = 64;   // Third conv layer filters
    const CNN_FILTERS_4 = 128;  // Fourth conv layer filters
    const CNN_KERNEL_SIZE = 3;  // Kernel size for conv layers
    const CNN_STRIDE = 1;       // Stride for conv layers
    const CNN_DENSE_UNITS = 256; // Dense layer after CNN
    const NOISY_DENSE_UNITS = 256; // NoisyDense layer units
    
    // Original dense architecture (not used with CNN)
    const HIDDEN_UNITS_1 = 512; // First hidden layer units
    const HIDDEN_UNITS_2 = 512; // Second hidden layer units
    const HIDDEN_UNITS_3 = 256; // Third hidden layer units
    const HIDDEN_UNITS_4 = 128; // Fourth hidden layer units
    const LEARNING_RATE_INITIAL = 0.0001; // Initial Adam optimizer learning rate (1e-4)
    const LEARNING_RATE_100 = 0.00005; // Learning rate after 100 episodes (5e-5)
    const LEARNING_RATE_200 = 0.00003; // Learning rate after 200 episodes (3e-5)
    const DEFAULT_GAMMA = 0.99;  // Discount factor for Q-learning (default)
    
    // Training configuration
    const DEFAULT_BATCH_SIZE = 128;
    const DEFAULT_REPLAY_BUFFER_SIZE = 300000; // Increased to 300k (200k-500k range)
    const DEFAULT_MIN_BUFFER_SIZE = 20000; // Warmup steps = 20k before training starts
    const DEFAULT_TRAIN_EVERY_N_STEPS = 2;
    const DEFAULT_TARGET_UPDATE_EVERY = 1000; // Hard update every 1000 steps
    const USE_SOFT_UPDATE = false; // Use hard updates every 1000 steps (changed from soft)
    const N_STEP_RETURNS = 3; // N-step returns for multi-step DQN
    
    // Epsilon-greedy defaults (added on top of NoisyNet)
    const DEFAULT_EPSILON = 0.01;
    const DEFAULT_EPSILON_START = 0.4;  // Start with epsilon = 0.4
    const DEFAULT_EPSILON_END = 0.01;    // Decay to 0.01
    const DEFAULT_EPSILON_DECAY = 0.995;
    
    // Physics timestep (60 FPS equivalent)
    const DELTA_TIME = 1000 / 60;
    
    // Maximum steps per episode to prevent infinite loops
    const MAX_STEPS_PER_EPISODE = 10000;
    
    // Reward shaping constants - NEW REWARD STRUCTURE
    const REWARD_VALID_PLACEMENT = 0.05;     // +0.05 for any valid fruit placement
    const REWARD_TOUCHING_SAME_TYPE = 0.10;  // +0.10 if fruit touches another fruit of same type
    const REWARD_HEIGHT_DECREASE = 0.05;     // +0.05 if tower height decreases
    const REWARD_HEIGHT_INCREASE = -0.05;    // -0.05 if tower height increases
    const REWARD_MERGE = 0.1;                // Keep existing merge reward
    const REWARD_GAME_OVER = -2.0;           // -2 for game over (overflow top)
    const LARGE_FRUIT_THRESHOLD = 6;         // Fruit level 6 or higher is considered "large"
    
    // Track previous tower height for height-based rewards
    let previousTowerHeight = 0;
    
    // Model references
    let model = null;
    let targetModel = null;  // Target network for stable Q-learning
    let optimizer = null;
    let episodeCount = 0; // Track total episodes for learning rate decay
    let currentLearningRate = LEARNING_RATE_INITIAL; // Track current learning rate
    
    // Track previous score for reward shaping
    let previousScore = 0;
    let previousMaxFruitLevel = -1;
    
    /**
     * Compute shaped reward based on game events - NEW REWARD STRUCTURE.
     * Returns rewards on every step, not only merges.
     * 
     * Reward components:
     * - +0.05 for any valid fruit placement
     * - +0.10 if the fruit touches another fruit of the same type
     * - +0.05 if tower height decreases
     * - -0.05 if tower height increases
     * - +mergeReward (keep existing) for merges
     * - -2 for game over (overflow top)
     * 
     * @param {number} scoreDelta - Change in score since last step
     * @param {number} currentTowerHeight - Current tower height (highest fruit Y position)
     * @param {boolean} isGameOver - Whether the game just ended
     * @param {boolean} wasValidPlacement - Whether a fruit was placed (column action taken)
     * @param {boolean} touchesSameType - Whether the placed fruit touches same type
     * @returns {{shapedReward: number, components: {validPlacement: number, touchingSameType: number, heightDecrease: number, heightIncrease: number, merge: number, gameOver: number}}}
     */
    function computeShapedReward(scoreDelta, currentTowerHeight, isGameOver, wasValidPlacement, touchesSameType) {
        const components = {
            validPlacement: 0,
            touchingSameType: 0,
            heightDecrease: 0,
            heightIncrease: 0,
            merge: 0,
            gameOver: 0
        };
        
        // Valid fruit placement reward
        if (wasValidPlacement) {
            components.validPlacement = REWARD_VALID_PLACEMENT;
        }
        
        // Touching same type reward
        if (touchesSameType) {
            components.touchingSameType = REWARD_TOUCHING_SAME_TYPE;
        }
        
        // Tower height change rewards
        const heightDelta = currentTowerHeight - previousTowerHeight;
        if (heightDelta < -0.01) { // Height decreased (tower got shorter)
            components.heightDecrease = REWARD_HEIGHT_DECREASE;
        } else if (heightDelta > 0.01) { // Height increased (tower got taller)
            components.heightIncrease = REWARD_HEIGHT_INCREASE;
        }
        
        // Detect merge: if score increased, a merge occurred
        if (scoreDelta > 0) {
            components.merge = REWARD_MERGE;
        }
        
        // Game over penalty
        if (isGameOver) {
            components.gameOver = REWARD_GAME_OVER;
        }
        
        // Update tracking variables
        previousTowerHeight = currentTowerHeight;
        
        // Total shaped reward
        const shapedReward = components.validPlacement + components.touchingSameType + 
                            components.heightDecrease + components.heightIncrease + 
                            components.merge + components.gameOver;
        
        return { shapedReward, components };
    }
    
    /**
     * Reset reward shaping state for a new episode.
     */
    function resetRewardShaping() {
        previousScore = 0;
        previousTowerHeight = 0;
    }
    
    /**
     * Get the tower height (highest fruit Y position) from state.
     * State format: [currentX, currentY, currentFruit, nextFruit, booster, ...boardFruits]
     * Board fruits are at positions 5+, with 3 values each (x, y, level)
     * 
     * @param {Float32Array|number[]} state - The state array
     * @returns {number} Tower height (0-1, highest Y position of fruits)
     */
    function getTowerHeightFromState(state) {
        let maxY = 0; // Lowest point (0 is top, 1 is bottom)
        
        // Board fruits start at index 5, each has 3 values (x, y, normalizedLevel)
        for (let i = 5; i + 2 < state.length; i += 3) {
            const normalizedY = state[i + 1];
            const normalizedLevel = state[i + 2];
            if (normalizedLevel > 0 && normalizedY > maxY) {
                maxY = normalizedY;
            }
        }
        
        // Return 1 - maxY because higher Y = lower on screen, we want tower height
        return 1 - maxY;
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
        const MAX_ACTUAL_FRUIT_LEVEL = 9; // Watermelon is level 9
        
        // Board fruits start at index 5, each has 3 values (x, y, normalizedLevel)
        for (let i = 5; i + 2 < state.length; i += 3) {
            const normalizedLevel = state[i + 2];
            if (normalizedLevel > 0) {
                // Convert normalized level (0-1) back to actual level (0-9)
                const level = Math.round(normalizedLevel * MAX_ACTUAL_FRUIT_LEVEL);
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
    
    // Pre-allocated temporary buffer for extracting single frame from stacked state
    const tempSingleFrameBuffer = new Float32Array(BASE_STATE_SIZE);
    
    // Frame stacking: keep a history of the last FRAME_STACK_SIZE frames
    // Each frame is BASE_STATE_SIZE elements
    const frameHistory = [];
    for (let i = 0; i < FRAME_STACK_SIZE; i++) {
        frameHistory.push(new Float32Array(BASE_STATE_SIZE));
    }
    let frameHistoryIndex = 0; // Current position in circular buffer
    
    /**
     * Initialize frame history with the given frame (typically at episode start).
     * Fills all frame slots with the same initial frame.
     * @param {number[]|Float32Array} frame - Initial frame to fill history with
     */
    function initFrameHistory(frame) {
        for (let i = 0; i < FRAME_STACK_SIZE; i++) {
            const buffer = frameHistory[i];
            const len = Math.min(frame.length, BASE_STATE_SIZE);
            for (let j = 0; j < len; j++) {
                buffer[j] = frame[j];
            }
            // Zero-pad if source is shorter
            for (let j = len; j < BASE_STATE_SIZE; j++) {
                buffer[j] = 0;
            }
        }
        frameHistoryIndex = 0;
    }
    
    /**
     * Add a new frame to the frame history (circular buffer).
     * @param {number[]|Float32Array} frame - New frame to add
     */
    function addFrameToHistory(frame) {
        const buffer = frameHistory[frameHistoryIndex];
        const len = Math.min(frame.length, BASE_STATE_SIZE);
        for (let i = 0; i < len; i++) {
            buffer[i] = frame[i];
        }
        // Zero-pad if source is shorter
        for (let i = len; i < BASE_STATE_SIZE; i++) {
            buffer[i] = 0;
        }
        frameHistoryIndex = (frameHistoryIndex + 1) % FRAME_STACK_SIZE;
    }
    
    /**
     * Stack the frame history into a single stacked state buffer.
     * Frames are stacked in temporal order (oldest to newest).
     * @param {Float32Array} stackedBuffer - Output buffer to write stacked frames to
     */
    function stackFrames(stackedBuffer) {
        // Stack frames in temporal order: oldest to newest
        // Current index points to the slot that will be overwritten next,
        // so the oldest frame is at frameHistoryIndex
        for (let i = 0; i < FRAME_STACK_SIZE; i++) {
            const frameIdx = (frameHistoryIndex + i) % FRAME_STACK_SIZE;
            const frame = frameHistory[frameIdx];
            const offset = i * BASE_STATE_SIZE;
            for (let j = 0; j < BASE_STATE_SIZE; j++) {
                stackedBuffer[offset + j] = frame[j];
            }
        }
    }
    
    // Replay buffer instance
    let replayBuffer = null;
    
    // N-step return buffer: stores (state, action, reward) for the last N steps
    // Used to compute n-step returns
    let nStepBuffer = [];
    
    /**
     * Compute n-step return from the n-step buffer.
     * For terminal transitions, just sum discounted rewards.
     * For non-terminal, the bootstrapped Q-value will be added during training via the target model.
     * R_t^n = r_t + γ*r_{t+1} + ... + γ^{n-1}*r_{t+n-1} (+ γ^n * Q(s_{t+n}, a*) if not done)
     * 
     * Note: The Q-value bootstrapping is handled by the training function using the next state,
     * so this function only computes the cumulative discounted reward portion.
     * 
     * @param {number} gamma - Discount factor
     * @returns {number} N-step cumulative discounted reward
     */
    function computeNStepReturn(gamma) {
        let nStepReturn = 0;
        let discount = 1;
        for (let i = 0; i < nStepBuffer.length; i++) {
            nStepReturn += discount * nStepBuffer[i].reward;
            discount *= gamma;
        }
        return nStepReturn;
    }
    
    /**
     * Clear the n-step buffer (at episode boundaries).
     */
    function clearNStepBuffer() {
        nStepBuffer = [];
    }
    
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
     * Convert a single flat state frame (155 elements) to integer board matrix + additional features.
     * 
     * NEW STATE REPRESENTATION:
     * The flat state format is:
     * - [0]: Current fruit X (0-1)
     * - [1]: Current fruit Y (0-1)
     * - [2]: Current fruit level (0-1)
     * - [3]: Next fruit level (0-1)
     * - [4]: Booster available (0 or 1)
     * - [5-154]: Board fruits, 50 fruits * 3 values (x, y, level)
     * 
     * Output format:
     * - boardMatrix: GRID_HEIGHT x GRID_WIDTH integer matrix with fruitID (0..maxFruit)
     *   - 0 = empty cell
     *   - 1-10 = fruit levels 0-9 (add 1 to distinguish from empty)
     * - additionalFeatures: [currentX, currentY, currentFruitType, nextFruitType]
     * 
     * @param {Float32Array|number[]} flatState - 155-element flat state
     * @returns {{boardMatrix: Int32Array, additionalFeatures: Float32Array}}
     */
    function convertFlatStateToIntegerBoard(flatState) {
        // Create integer board matrix (GRID_HEIGHT x GRID_WIDTH)
        const boardMatrix = new Int32Array(GRID_HEIGHT * GRID_WIDTH);
        boardMatrix.fill(0); // 0 = empty
        
        // Extract metadata from flat state
        const currentX = flatState[0];
        const currentY = flatState[1];
        const currentLevel = flatState[2]; // normalized 0-1
        const nextLevel = flatState[3]; // normalized 0-1
        
        // Convert normalized levels to fruit IDs (0-10, where 0=empty, 1-10=fruit levels 0-9)
        const currentFruitID = currentLevel > 0 ? Math.round(currentLevel * MAX_ACTUAL_FRUIT_LEVEL) + 1 : 0;
        const nextFruitID = nextLevel > 0 ? Math.round(nextLevel * MAX_ACTUAL_FRUIT_LEVEL) + 1 : 0;
        
        // Helper function to convert normalized position to grid indices
        function posToGridIndex(normX, normY) {
            const gridX = Math.floor(normX * GRID_WIDTH);
            const gridY = Math.floor(normY * GRID_HEIGHT);
            // Clamp to valid range
            const clampedX = Math.max(0, Math.min(GRID_WIDTH - 1, gridX));
            const clampedY = Math.max(0, Math.min(GRID_HEIGHT - 1, gridY));
            return { x: clampedX, y: clampedY };
        }
        
        // Fill board matrix with board fruits
        // Process all board fruits (starting at index 5)
        for (let i = 5; i < flatState.length; i += 3) {
            const fruitX = flatState[i];
            const fruitY = flatState[i + 1];
            const fruitLevel = flatState[i + 2]; // normalized 0-1
            
            // Skip if no fruit at this slot (level is 0)
            if (fruitLevel === 0) continue;
            
            const { x: gridX, y: gridY } = posToGridIndex(fruitX, fruitY);
            const idx = gridY * GRID_WIDTH + gridX;
            
            // Convert normalized level to fruit ID (0-10: 0=empty, 1-10=fruit levels 0-9)
            const fruitID = fruitLevel > 0 ? Math.round(fruitLevel * MAX_ACTUAL_FRUIT_LEVEL) + 1 : 0;
            
            // Take max fruit ID if multiple fruits in same cell (shouldn't happen often)
            boardMatrix[idx] = Math.max(boardMatrix[idx], fruitID);
        }
        
        // Create additional features vector
        const additionalFeatures = new Float32Array(ADDITIONAL_FEATURES);
        additionalFeatures[0] = currentX; // normalized X position
        additionalFeatures[1] = currentY; // normalized Y position
        // Normalize fruit IDs: 0 → 0, 1-10 → 0.1-1.0
        additionalFeatures[2] = currentFruitID > 0 ? currentFruitID / MAX_FRUIT_LEVEL : 0;
        additionalFeatures[3] = nextFruitID > 0 ? nextFruitID / MAX_FRUIT_LEVEL : 0;
        
        return { boardMatrix, additionalFeatures };
    }
    
    /**
     * Convert stacked flat state (3 frames * 155 elements = 465) to stacked integer board + features.
     * 
     * NEW STATE REPRESENTATION:
     * Output format:
     * - boardMatrices: FRAME_STACK_SIZE x GRID_HEIGHT x GRID_WIDTH integer matrices
     * - additionalFeatures: Concatenated features from all frames (FRAME_STACK_SIZE * ADDITIONAL_FEATURES)
     * 
     * @param {Float32Array} stackedFlatState - 465-element stacked flat state
     * @returns {{boardMatrices: Int32Array[], additionalFeatures: Float32Array}}
     */
    function convertStackedStateToIntegerBoards(stackedFlatState) {
        const boardMatrices = [];
        const allAdditionalFeatures = new Float32Array(FRAME_STACK_SIZE * ADDITIONAL_FEATURES);
        
        // Convert each frame
        for (let frameIdx = 0; frameIdx < FRAME_STACK_SIZE; frameIdx++) {
            // Extract single frame from stacked state
            const frameOffset = frameIdx * BASE_STATE_SIZE;
            const singleFrame = stackedFlatState.subarray(frameOffset, frameOffset + BASE_STATE_SIZE);
            
            // Convert to integer board + features
            const { boardMatrix, additionalFeatures } = convertFlatStateToIntegerBoard(singleFrame);
            
            boardMatrices.push(boardMatrix);
            
            // Copy additional features to the combined array
            const featureOffset = frameIdx * ADDITIONAL_FEATURES;
            for (let i = 0; i < ADDITIONAL_FEATURES; i++) {
                allAdditionalFeatures[featureOffset + i] = additionalFeatures[i];
            }
        }
        
        return { boardMatrices, additionalFeatures: allAdditionalFeatures };
    }
    
    /**
     * Convert a batch of flat states to model inputs for GPU-accelerated processing.
     * NEW: Returns board integer tensors for embedding + additional features tensor.
     * 
     * @param {Float32Array} flatStates - Flat states array (batchSize * STATE_SIZE)
     * @param {number} batchSize - Number of states in the batch
     * @returns {{boardTensor: tf.Tensor3D, featuresTensor: tf.Tensor2D}} 
     *          boardTensor shape: [batchSize, GRID_HEIGHT, GRID_WIDTH]
     *          featuresTensor shape: [batchSize, FRAME_STACK_SIZE * ADDITIONAL_FEATURES]
     */
    function convertBatchToModelInputs(flatStates, batchSize) {
        // Pre-allocate arrays
        const boardSize = GRID_HEIGHT * GRID_WIDTH;
        const batchBoardData = new Int32Array(batchSize * boardSize);
        const featuresSize = FRAME_STACK_SIZE * ADDITIONAL_FEATURES;
        const batchFeaturesData = new Float32Array(batchSize * featuresSize);
        
        // Convert each state in the batch
        for (let b = 0; b < batchSize; b++) {
            const stateOffset = b * STATE_SIZE;
            
            // Extract single stacked state
            const singleState = flatStates.subarray(stateOffset, stateOffset + STATE_SIZE);
            
            // Convert to integer boards + features
            const { boardMatrices, additionalFeatures } = convertStackedStateToIntegerBoards(singleState);
            
            // Use only the most recent frame's board (last frame in stack)
            // This simplifies the model architecture
            const mostRecentBoard = boardMatrices[FRAME_STACK_SIZE - 1];
            const boardOffset = b * boardSize;
            for (let i = 0; i < boardSize; i++) {
                batchBoardData[boardOffset + i] = mostRecentBoard[i];
            }
            
            // Copy all additional features
            const featuresOffset = b * featuresSize;
            for (let i = 0; i < featuresSize; i++) {
                batchFeaturesData[featuresOffset + i] = additionalFeatures[i];
            }
        }
        
        // Create tensors
        const boardTensor = tf.tensor3d(
            batchBoardData,
            [batchSize, GRID_HEIGHT, GRID_WIDTH],
            'int32'
        );
        
        const featuresTensor = tf.tensor2d(
            batchFeaturesData,
            [batchSize, featuresSize]
        );
        
        return { boardTensor, featuresTensor };
    }
    
    /**
     * Custom NoisyDense layer using factorized Gaussian noise.
     * Implements the NoisyNet formula: y = (μ_w + σ_w ⊙ ε_w) x + (μ_b + σ_b ⊙ ε_b)
     * 
     * Noise is reset at each forward pass for exploration.
     * 
     * @class NoisyDense
     */
    class NoisyDense extends tf.layers.Layer {
        constructor(config) {
            super(config);
            this.units = config.units;
            this.activation = config.activation || 'linear';
            this.useBias = config.useBias !== undefined ? config.useBias : true;
            // Configurable noise initialization - standard value from NoisyNet paper
            this.sigmaInit = config.sigmaInit || 0.017;
            
            // Factorized noise parameters
            // For input dimension p and output dimension q:
            // We use factorized noise: ε_w = f(ε_i) ⊗ f(ε_j)
            // This reduces noise parameters from p*q to p+q
            this.inputDim = null; // Set in build()
            
            // Trainable parameters
            this.muWeight = null;
            this.sigmaWeight = null;
            this.muBias = null;
            this.sigmaBias = null;
            
            // Noise samples (not trainable, regenerated each forward pass)
            this.epsilonInput = null;
            this.epsilonOutput = null;
            
            // Pre-create activation layer if needed (optimization)
            this.activationLayer = null;
            if (this.activation !== 'linear') {
                this.activationLayer = tf.layers.activation({activation: this.activation});
            }
        }
        
        build(inputShape) {
            this.inputDim = inputShape[inputShape.length - 1];
            
            // Initialize μ_w and σ_w for weights
            // Using uniform initialization: μ ~ U(-1/√fan_in, 1/√fan_in)
            // This is a common simplified initialization (also used in original NoisyNet paper)
            // Note: This differs from Xavier/Glorot which uses √(6/(fan_in + fan_out))
            const muRange = 1.0 / Math.sqrt(this.inputDim);
            
            this.muWeight = this.addWeight(
                'mu_weight',
                [this.inputDim, this.units],
                'float32',
                tf.initializers.randomUniform({minval: -muRange, maxval: muRange})
            );
            
            this.sigmaWeight = this.addWeight(
                'sigma_weight',
                [this.inputDim, this.units],
                'float32',
                tf.initializers.constant({value: this.sigmaInit})
            );
            
            if (this.useBias) {
                this.muBias = this.addWeight(
                    'mu_bias',
                    [this.units],
                    'float32',
                    tf.initializers.randomUniform({minval: -muRange, maxval: muRange})
                );
                
                this.sigmaBias = this.addWeight(
                    'sigma_bias',
                    [this.units],
                    'float32',
                    tf.initializers.constant({value: this.sigmaInit})
                );
            }
            
            super.build(inputShape);
        }
        
        /**
         * Generate factorized Gaussian noise using f(x) = sgn(x) * sqrt(|x|)
         * This is the factorization function from the NoisyNet paper.
         * The noise tensor is kept (not tidied) so it can be stored as instance variable.
         */
        factorizedNoise(size) {
            // Create noise and transform it - wrapped in tidy() for automatic cleanup
            return tf.tidy(() => {
                const noise = tf.randomNormal([size]);
                // f(x) = sgn(x) * sqrt(|x|)
                const sign = tf.sign(noise);
                const abs = tf.abs(noise);
                const sqrt = tf.sqrt(abs);
                const result = tf.mul(sign, sqrt);
                // tf.keep() prevents the result from being disposed by tidy()
                return tf.keep(result);
            });
        }
        
        /**
         * Reset noise samples (called at each forward pass).
         * Note: Resetting noise at each forward pass is intentional per NoisyNet paper.
         * This provides state-dependent exploration.
         */
        resetNoise() {
            // Dispose previous noise tensors if they exist
            if (this.epsilonInput) {
                this.epsilonInput.dispose();
            }
            if (this.epsilonOutput) {
                this.epsilonOutput.dispose();
            }
            
            this.epsilonInput = this.factorizedNoise(this.inputDim);
            this.epsilonOutput = this.factorizedNoise(this.units);
        }
        
        call(inputs, kwargs) {
            // Reset noise OUTSIDE of tf.tidy() to avoid memory management issues
            // The noise tensors are stored as instance variables and managed manually
            // Use try-finally to ensure proper cleanup even if an error occurs
            try {
                this.resetNoise();
                
                return tf.tidy(() => {
                    const input = inputs instanceof Array ? inputs[0] : inputs;
                    
                    // Compute factorized noise for weights: ε_w = ε_input ⊗ ε_output
                    // This gives us a [inputDim, units] noise matrix
                    const epsilonWeight = tf.outerProduct(this.epsilonInput, this.epsilonOutput);
                    
                    // Compute noisy weights: w = μ_w + σ_w ⊙ ε_w
                    const noisyWeight = tf.add(
                        this.muWeight.read(),
                        tf.mul(this.sigmaWeight.read(), epsilonWeight)
                    );
                    
                    // Apply linear transformation: y = x * w
                    let output = tf.matMul(input, noisyWeight);
                    
                    if (this.useBias) {
                        // Compute noisy bias: b = μ_b + σ_b ⊙ ε_output
                        const noisyBias = tf.add(
                            this.muBias.read(),
                            tf.mul(this.sigmaBias.read(), this.epsilonOutput)
                        );
                        
                        // Add bias: y = y + b
                        output = tf.add(output, noisyBias);
                    }
                    
                    // Apply activation using pre-created layer (optimization)
                    if (this.activationLayer) {
                        output = this.activationLayer.apply(output);
                    }
                    
                    return output;
                });
            } catch (error) {
                // If an error occurs, ensure noise tensors are cleaned up to prevent leaks
                if (this.epsilonInput) {
                    this.epsilonInput.dispose();
                    this.epsilonInput = null;
                }
                if (this.epsilonOutput) {
                    this.epsilonOutput.dispose();
                    this.epsilonOutput = null;
                }
                throw error;
            }
        }
        
        computeOutputShape(inputShape) {
            return [inputShape[0], this.units];
        }
        
        getConfig() {
            const config = {
                units: this.units,
                activation: this.activation,
                useBias: this.useBias,
                sigmaInit: this.sigmaInit
            };
            const baseConfig = super.getConfig();
            return Object.assign({}, baseConfig, config);
        }
        
        dispose() {
            // Clean up noise tensors
            if (this.epsilonInput) {
                this.epsilonInput.dispose();
                this.epsilonInput = null;
            }
            if (this.epsilonOutput) {
                this.epsilonOutput.dispose();
                this.epsilonOutput = null;
            }
            return super.dispose();
        }
        
        static get className() {
            return 'NoisyDense';
        }
    }
    
    // Register the custom layer (with error handling for re-registration)
    try {
        tf.serialization.registerClass(NoisyDense);
    } catch (e) {
        // Layer already registered - this can happen during hot reload or multiple script loads
        console.warn('[Train] NoisyDense layer already registered:', e.message);
    }
    
    /**
     * Create a Q-network model with NEW DEEPER CNN + Embedding + NoisyNet architecture.
     * Used for both main model and target model.
     * 
     * NEW ARCHITECTURE:
     * - Board Input: Integer matrix (GRID_HEIGHT x GRID_WIDTH) with fruit IDs
     * - Embedding layer: (MAX_FRUIT_LEVEL+1) x EMBEDDING_DIM, output shape [15, 10, 16]
     * - Conv2D(32, 3, relu)
     * - Conv2D(64, 3, relu)
     * - Conv2D(64, 3, relu)
     * - Conv2D(128, 3, relu)
     * - Flatten
     * - Concatenate with additional features [currentX, currentY, currentFruitType, nextFruitType]
     * - Dense(256, relu)
     * - NoisyDense(256, relu)
     * - Output layer: NUM_ACTIONS (10 column actions)
     * 
     * @returns {tf.LayersModel} The created model
     */
    function createQNetwork() {
        // Board input - integer matrix for embedding
        const boardInput = tf.input({shape: [GRID_HEIGHT, GRID_WIDTH], dtype: 'int32', name: 'board_input'});
        
        // Additional features input
        const featuresInput = tf.input({
            shape: [FRAME_STACK_SIZE * ADDITIONAL_FEATURES], 
            name: 'features_input'
        });
        
        // Embedding layer: converts integer fruit IDs to dense vectors
        // inputDim = MAX_FRUIT_LEVEL + 1 (0=empty, 1-10=fruit levels 0-9)
        let embedded = tf.layers.embedding({
            inputDim: MAX_FRUIT_LEVEL + 1,
            outputDim: EMBEDDING_DIM,
            name: 'fruit_embedding'
        }).apply(boardInput);
        
        // Reshape embedded output to [batch, GRID_HEIGHT, GRID_WIDTH, EMBEDDING_DIM]
        embedded = tf.layers.reshape({
            targetShape: [GRID_HEIGHT, GRID_WIDTH, EMBEDDING_DIM],
            name: 'reshape_embedded'
        }).apply(embedded);
        
        // First convolutional layer
        let conv = tf.layers.conv2d({
            filters: CNN_FILTERS_1,
            kernelSize: CNN_KERNEL_SIZE,
            strides: CNN_STRIDE,
            activation: 'relu',
            kernelInitializer: 'heNormal',
            name: 'conv2d_1'
        }).apply(embedded);
        
        // Second convolutional layer
        conv = tf.layers.conv2d({
            filters: CNN_FILTERS_2,
            kernelSize: CNN_KERNEL_SIZE,
            strides: CNN_STRIDE,
            activation: 'relu',
            kernelInitializer: 'heNormal',
            name: 'conv2d_2'
        }).apply(conv);
        
        // Third convolutional layer (NEW)
        conv = tf.layers.conv2d({
            filters: CNN_FILTERS_3,
            kernelSize: CNN_KERNEL_SIZE,
            strides: CNN_STRIDE,
            activation: 'relu',
            kernelInitializer: 'heNormal',
            name: 'conv2d_3'
        }).apply(conv);
        
        // Fourth convolutional layer (NEW)
        conv = tf.layers.conv2d({
            filters: CNN_FILTERS_4,
            kernelSize: CNN_KERNEL_SIZE,
            strides: CNN_STRIDE,
            activation: 'relu',
            kernelInitializer: 'heNormal',
            name: 'conv2d_4'
        }).apply(conv);
        
        // Flatten the convolutional output
        let flattened = tf.layers.flatten({
            name: 'flatten'
        }).apply(conv);
        
        // Concatenate with additional features
        let concatenated = tf.layers.concatenate({
            name: 'concatenate_features'
        }).apply([flattened, featuresInput]);
        
        // Dense layer after concatenation
        let hidden = tf.layers.dense({
            units: CNN_DENSE_UNITS,
            activation: 'relu',
            kernelInitializer: 'heNormal',
            name: 'dense_post_concat'
        }).apply(concatenated);
        
        // NoisyDense layer for exploration
        hidden = new NoisyDense({
            units: NOISY_DENSE_UNITS,
            activation: 'relu',
            name: 'noisy_dense'
        }).apply(hidden);
        
        // Output layer: Q-values for each action (10 columns)
        const output = tf.layers.dense({
            units: NUM_ACTIONS,
            activation: 'linear',
            kernelInitializer: 'heNormal',
            name: 'q_values'
        }).apply(hidden);
        
        // Create the model with two inputs
        const network = tf.model({
            inputs: [boardInput, featuresInput],
            outputs: output,
            name: 'cnn_embedding_noisynet_dqn'
        });
        
        return network;
    }
    
    /**
     * Build and compile the Q-network model.
     * Also creates the target network with the same architecture.
     * 
     * Optimizer: Adam with learning rate decay schedule
     * Loss: Mean Squared Error (MSE)
     */
    window.RL.initModel = function() {
        console.log('[Train] Building Q-network model...');
        
        // Create main model
        model = createQNetwork();
        
        // Create optimizer for gradient updates with initial learning rate
        optimizer = tf.train.adam(LEARNING_RATE_INITIAL);
        
        // Reset episode count and current learning rate
        episodeCount = 0;
        currentLearningRate = LEARNING_RATE_INITIAL;
        
        // Compile main model
        model.compile({
            optimizer: optimizer,
            loss: 'meanSquaredError'
        });
        
        // Create target model with same architecture
        targetModel = createQNetwork();
        
        // Compile target model (not used for training, but needed for predict)
        targetModel.compile({
            optimizer: tf.train.adam(LEARNING_RATE_INITIAL),
            loss: 'meanSquaredError'
        });
        
        // Initialize target model with same weights as main model (hard update for initialization)
        updateTargetModel(true);
        
        // Initialize replay buffer
        replayBuffer = new ReplayBuffer(DEFAULT_REPLAY_BUFFER_SIZE, STATE_SIZE);
        
        console.log('[Train] Model built and compiled successfully.');
        console.log('[Train] NEW ARCHITECTURE: Embedding + Deeper CNN + NoisyNet (GPU-accelerated)');
        console.log('[Train]   - Board Input: Integer matrix [15, 10] with fruit IDs (0-10)');
        console.log('[Train]   - Embedding: (11 x 16) -> Output shape [15, 10, 16]');
        console.log('[Train]   - Conv2D(32, 3x3, ReLU)');
        console.log('[Train]   - Conv2D(64, 3x3, ReLU)');
        console.log('[Train]   - Conv2D(64, 3x3, ReLU)');
        console.log('[Train]   - Conv2D(128, 3x3, ReLU)');
        console.log('[Train]   - Flatten + Concatenate with [currentX, currentY, currentFruitType, nextFruitType]');
        console.log('[Train]   - Dense(256, ReLU)');
        console.log('[Train]   - NoisyDense(256, ReLU)');
        console.log('[Train]   - Output: 10 Q-values (column actions 0-9)');
        console.log('[Train] Actions: Column-based (0-9), drop fruit at center of selected column');
        console.log('[Train] Exploration: NoisyNet + epsilon-greedy (ε: 0.4 → 0.01)');
        console.log('[Train] Replay buffer: ' + DEFAULT_REPLAY_BUFFER_SIZE + ' transitions, warmup: ' + DEFAULT_MIN_BUFFER_SIZE);
        console.log('[Train] Target network: Hard update every ' + DEFAULT_TARGET_UPDATE_EVERY + ' steps');
        console.log('[Train] Batch size: ' + DEFAULT_BATCH_SIZE);
        console.log('[Train] Learning rate decay: ' + LEARNING_RATE_INITIAL + ' → ' + LEARNING_RATE_100 + ' (100 eps) → ' + LEARNING_RATE_200 + ' (200 eps)');
        console.log('[Train] Target model initialized with same weights (hard update).');
        console.log('[Train] Stability features enabled: Huber loss, gradient clipping, TAU=' + TAU);
        console.log('[Train] NEW Reward shaping: +0.05 valid placement, +0.10 touching same, ±0.05 height, +merge, -2 game over');
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
     * NEW: Converts flat states to board integers + features for embedding-based model.
     * 
     * @param {Float32Array} statesFlat - Flattened states array (batchSize * stateSize)
     * @param {number} batchSize - Number of states in the batch
     * @returns {Float32Array} Q-values for all states (batchSize * numActions)
     */
    function batchedPredict(statesFlat, batchSize) {
        return tf.tidy(() => {
            const { boardTensor, featuresTensor } = convertBatchToModelInputs(statesFlat, batchSize);
            const qValues = model.predict([boardTensor, featuresTensor]);
            return qValues.dataSync();
        });
    }
    
    /**
     * Select action using Q-values with epsilon-greedy exploration.
     * NEW: Uses epsilon-greedy on top of NoisyNet for better exploration.
     * Converts flat state to board integers + features for embedding-based model.
     * 
     * @param {Float32Array} stateBuffer - Pre-allocated state buffer
     * @param {number} epsilon - Exploration rate (0-1), use epsilon-greedy on top of NoisyNet
     * @returns {number} Action index (0-9 for column actions)
     */
    function selectActionFromBuffer(stateBuffer, epsilon) {
        // Epsilon-greedy exploration on top of NoisyNet
        if (Math.random() < epsilon) {
            return Math.floor(Math.random() * NUM_ACTIONS);
        }
        
        // Choose action with highest noisy Q-value
        return tf.tidy(() => {
            const { boardTensor, featuresTensor } = convertBatchToModelInputs(stateBuffer, 1);
            const qValues = model.predict([boardTensor, featuresTensor]);
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
            // Convert next states to model inputs (board + features)
            const { boardTensor: nextBoardTensor, featuresTensor: nextFeaturesTensor } = 
                convertBatchToModelInputs(nextStates, actualBatchSize);
            
            // DOUBLE DQN: Use ONLINE model to select actions
            const onlineNextQValues = model.predict([nextBoardTensor, nextFeaturesTensor]);
            const bestActions = onlineNextQValues.argMax(1).dataSync();
            
            // DOUBLE DQN: Use TARGET model to evaluate the selected actions
            // Clip target Q-values to prevent extreme values
            const targetNextQValues = targetModel.predict([nextBoardTensor, nextFeaturesTensor]);
            const targetNextQClipped = tf.clipByValue(targetNextQValues, -MAX_Q_VALUE, MAX_Q_VALUE);
            const targetNextQData = targetNextQClipped.arraySync();
            
            if (verbose) {
                //console.log('[Train] Double-DQN: Using online model for action selection, target model for evaluation');
                //console.log('[Train] Stability: Huber loss, gradient clipping, Q-value clipping enabled');
            }
            
            // Convert current states to model inputs
            const { boardTensor: statesBoardTensor, featuresTensor: statesFeaturesTensor } = 
                convertBatchToModelInputs(states, actualBatchSize);
            const currentQValues = model.predict([statesBoardTensor, statesFeaturesTensor]);
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
        const { boardTensor: statesBoardTensor, featuresTensor: statesFeaturesTensor } = 
            convertBatchToModelInputs(states, actualBatchSize);
        
        // Create weights tensor if we have importance sampling weights
        let weightsTensor = null;
        if (weights) {
            weightsTensor = tf.tensor1d(weights);
        }
        
        // Define loss function using Huber loss for stability
        const lossFunction = () => {
            const predictions = model.predict([statesBoardTensor, statesFeaturesTensor]);
            
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
        statesBoardTensor.dispose();
        statesFeaturesTensor.dispose();
        loss.dispose();
        Object.values(grads).forEach(g => g.dispose());
        Object.values(clippedGrads).forEach(g => g.dispose());
        if (weightsTensor) {
            weightsTensor.dispose();
        }
        
        return { loss: lossValue, tdErrors };
    }
    
    /**
     * Update the optimizer learning rate based on episode count.
     * Implements stepped learning rate decay:
     * - Episodes 0-99: 1e-4 (LEARNING_RATE_INITIAL)
     * - Episodes 100-199: 5e-5 (LEARNING_RATE_100)
     * - Episodes 200+: 3e-5 (LEARNING_RATE_200)
     */
    function updateLearningRate() {
        // Early return if model or optimizer not initialized
        if (!model || !optimizer) {
            return;
        }
        
        let newLearningRate;
        if (episodeCount >= 200) {
            newLearningRate = LEARNING_RATE_200;
        } else if (episodeCount >= 100) {
            newLearningRate = LEARNING_RATE_100;
        } else {
            newLearningRate = LEARNING_RATE_INITIAL;
        }
        
        // Create new optimizer with updated learning rate if needed
        // TensorFlow.js optimizers have read-only learning rate, so we need to create a new instance
        if (currentLearningRate !== newLearningRate) {
            // Dispose old optimizer
            optimizer.dispose();
            // Create new optimizer with new learning rate
            optimizer = tf.train.adam(newLearningRate);
            // Update tracked learning rate
            currentLearningRate = newLearningRate;
            // Recompile model with new optimizer
            model.compile({
                optimizer: optimizer,
                loss: 'meanSquaredError'
            });
            console.log(`[Train] Learning rate updated to ${newLearningRate} at episode ${episodeCount}`);
        }
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
     * @param {number} [options.batchSize=128] - Minibatch size for training
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
            console.log(`[Train] Prioritized replay beta annealing: ${PRIORITY_BETA_START} -> ${PRIORITY_BETA_END} over ${PRIORITY_BETA_EPISODES} episodes`);
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
                    //console.log(`[Train] Episode ${episode + 1}/${numEpisodes} starting...`);
                }
                
                // Update learning rate based on episode count
                updateLearningRate();
                
                // Compute annealed beta for prioritized replay
                // Beta increases from PRIORITY_BETA_START to PRIORITY_BETA_END over PRIORITY_BETA_EPISODES
                const betaProgress = Math.min(1.0, episodeCount / PRIORITY_BETA_EPISODES);
                const currentBeta = PRIORITY_BETA_START + betaProgress * (PRIORITY_BETA_END - PRIORITY_BETA_START);
                
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
                
                // Track shaped reward components for logging (NEW reward structure)
                let totalValidPlacementReward = 0;
                let totalTouchingSameTypeReward = 0;
                let totalHeightDecreaseReward = 0;
                let totalHeightIncreaseReward = 0;
                let totalMergeReward = 0;
                let totalGameOverPenalty = 0;
                
                // Get initial state and initialize frame history with it
                const rawState = window.RL.getState();
                initFrameHistory(rawState);
                stackFrames(stateBuffer);
                previousScore = gameContext.getScore();
                
                // Clear n-step buffer for new episode
                clearNStepBuffer();
                
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
                    addFrameToHistory(rawNextState);
                    stackFrames(nextStateBuffer);
                    const done = window.RL.isTerminal();
                    
                    // For tower height and max fruit level, extract from raw state
                    // Reuse pre-allocated buffer for the single frame
                    const len = Math.min(rawNextState.length, BASE_STATE_SIZE);
                    for (let i = 0; i < len; i++) {
                        tempSingleFrameBuffer[i] = rawNextState[i];
                    }
                    const currentTowerHeight = getTowerHeightFromState(tempSingleFrameBuffer);
                    
                    // Compute shaped reward with NEW reward structure
                    // TODO: Implement proper wasValidPlacement and touchesSameType detection
                    // For now, assume all column actions are valid placements
                    const wasValidPlacement = (action >= 0 && action < NUM_ACTIONS);
                    const touchesSameType = false; // TODO: Detect from physics collisions
                    const { shapedReward, components } = computeShapedReward(
                        scoreDelta, currentTowerHeight, done, wasValidPlacement, touchesSameType
                    );
                    totalReward += shapedReward;
                    
                    // Track reward components for logging
                    totalValidPlacementReward += components.validPlacement;
                    totalTouchingSameTypeReward += components.touchingSameType;
                    totalHeightDecreaseReward += components.heightDecrease;
                    totalHeightIncreaseReward += components.heightIncrease;
                    totalMergeReward += components.merge;
                    totalGameOverPenalty += components.gameOver;
                    
                    // Add transition to n-step buffer
                    nStepBuffer.push({
                        state: new Float32Array(stateBuffer),
                        action: action,
                        reward: shapedReward,
                        nextState: new Float32Array(nextStateBuffer),
                        done: done
                    });
                    
                    // If we have N_STEP_RETURNS transitions or episode ended, add to replay buffer
                    if (nStepBuffer.length >= N_STEP_RETURNS || done) {
                        const nStepReturn = computeNStepReturn(validGamma);
                        const firstTransition = nStepBuffer[0];
                        const lastTransition = nStepBuffer[nStepBuffer.length - 1];
                        
                        // Store transition in replay buffer with n-step return
                        // Use the state from N steps ago, action from N steps ago, n-step return,
                        // next state from the last transition, and done flag from the last transition
                        replayBuffer.add(
                            firstTransition.state, 
                            firstTransition.action, 
                            nStepReturn, 
                            lastTransition.nextState, 
                            lastTransition.done
                        );
                        
                        // Remove the oldest transition from n-step buffer (sliding window)
                        nStepBuffer.shift();
                    }
                    
                    // If episode ended, flush remaining transitions in n-step buffer
                    if (done) {
                        while (nStepBuffer.length > 0) {
                            const nStepReturn = computeNStepReturn(validGamma);
                            const firstTransition = nStepBuffer[0];
                            const lastTransition = nStepBuffer[nStepBuffer.length - 1];
                            
                            // Each remaining transition uses its proper next state and done flag
                            replayBuffer.add(
                                firstTransition.state, 
                                firstTransition.action, 
                                nStepReturn, 
                                lastTransition.nextState, 
                                lastTransition.done
                            );
                            nStepBuffer.shift();
                        }
                    }
                    
                    // Update canTrain flag if we just crossed the threshold
                    if (!canTrain && replayBuffer.size() >= validMinBufferSize) {
                        canTrain = true;
                    }
                    
                    // Train on minibatch if enough samples and at training interval
                    if (canTrain && stepCount % validTrainEveryNSteps === 0) {
                        // Use prioritized replay sampling
                        const batch = replayBuffer.getPrioritizedBatch(validBatchSize, currentBeta);
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
                
                // Increment episode count for learning rate decay
                episodeCount++;
                
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
                    // Reward shaping components (NEW)
                    rewardComponents: {
                        validPlacement: totalValidPlacementReward,
                        touchingSameType: totalTouchingSameTypeReward,
                        heightDecrease: totalHeightDecreaseReward,
                        heightIncrease: totalHeightIncreaseReward,
                        merge: totalMergeReward,
                        gameOver: totalGameOverPenalty,
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
                        `[Train] NEW Reward components: ` +
                        `validPlace=${totalValidPlacementReward.toFixed(2)}, touchSame=${totalTouchingSameTypeReward.toFixed(2)}, ` +
                        `heightDown=${totalHeightDecreaseReward.toFixed(2)}, heightUp=${totalHeightIncreaseReward.toFixed(2)}, ` +
                        `merge=${totalMergeReward.toFixed(2)}, gameOver=${totalGameOverPenalty.toFixed(0)}`
                    );
                    if (trainCount > 0) {
                        console.log(`[Train] Mean TD error: ${avgMeanTDError.toFixed(4)}`);
                    }
                }
                
                // Save model and metadata after each episode
                try {
                    await model.save('localstorage://fruit-merge-dqn-v1');
                    await saveTrainingMetadata();
                    if (verbose && (episode + 1) % 5 === 0) {
                        console.log(`[Train] Progress saved at episode ${episode + 1}`);
                    }
                } catch (err) {
                    console.error('[Train] Failed to save training progress:', err);
                }
            }
            
            // Final save to localStorage
            if (verbose) {
                console.log('[Train] Saving final model to localStorage...');
            }
            try {
                await model.save('localstorage://fruit-merge-dqn-v1');
                await saveTrainingMetadata();
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
            console.log(`[Train] Prioritized replay beta annealing: ${PRIORITY_BETA_START} -> ${PRIORITY_BETA_END} over ${PRIORITY_BETA_EPISODES} episodes`);
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
                
                // Update learning rate based on episode count
                updateLearningRate();
                
                // Compute annealed beta for prioritized replay
                // Beta increases from PRIORITY_BETA_START to PRIORITY_BETA_END over PRIORITY_BETA_EPISODES
                const betaProgress = Math.min(1.0, episodeCount / PRIORITY_BETA_EPISODES);
                const currentBeta = PRIORITY_BETA_START + betaProgress * (PRIORITY_BETA_END - PRIORITY_BETA_START);
                
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
                
                // Track shaped reward components for logging (NEW reward structure)
                let totalValidPlacementReward = 0;
                let totalTouchingSameTypeReward = 0;
                let totalHeightDecreaseReward = 0;
                let totalHeightIncreaseReward = 0;
                let totalMergeReward = 0;
                let totalGameOverPenalty = 0;
                
                const rawState = window.RL.getState();
                initFrameHistory(rawState);
                stackFrames(stateBuffer);
                previousScore = gameContext.getScore();
                
                // Clear n-step buffer for new episode
                clearNStepBuffer();
                
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
                    addFrameToHistory(rawNextState);
                    stackFrames(nextStateBuffer);
                    const done = window.RL.isTerminal();
                    
                    // For tower height extraction from raw state
                    // Reuse pre-allocated buffer for the single frame
                    const len = Math.min(rawNextState.length, BASE_STATE_SIZE);
                    for (let i = 0; i < len; i++) {
                        tempSingleFrameBuffer[i] = rawNextState[i];
                    }
                    const currentTowerHeight = getTowerHeightFromState(tempSingleFrameBuffer);
                    
                    // Compute shaped reward with NEW reward structure
                    // TODO: Implement proper wasValidPlacement and touchesSameType detection
                    const wasValidPlacement = (action >= 0 && action < NUM_ACTIONS);
                    const touchesSameType = false; // TODO: Detect from physics collisions
                    const { shapedReward, components } = computeShapedReward(
                        scoreDelta, currentTowerHeight, done, wasValidPlacement, touchesSameType
                    );
                    totalReward += shapedReward;
                    
                    // Track reward components for logging
                    totalValidPlacementReward += components.validPlacement;
                    totalTouchingSameTypeReward += components.touchingSameType;
                    totalHeightDecreaseReward += components.heightDecrease;
                    totalHeightIncreaseReward += components.heightIncrease;
                    totalMergeReward += components.merge;
                    totalGameOverPenalty += components.gameOver;
                    
                    // Add transition to n-step buffer
                    nStepBuffer.push({
                        state: new Float32Array(stateBuffer),
                        action: action,
                        reward: shapedReward,
                        nextState: new Float32Array(nextStateBuffer),
                        done: done
                    });
                    
                    // If we have N_STEP_RETURNS transitions or episode ended, add to replay buffer
                    if (nStepBuffer.length >= N_STEP_RETURNS || done) {
                        const nStepReturn = computeNStepReturn(validGamma);
                        const firstTransition = nStepBuffer[0];
                        const lastTransition = nStepBuffer[nStepBuffer.length - 1];
                        
                        // Store transition in replay buffer with n-step return
                        // Use the state from N steps ago, action from N steps ago, n-step return,
                        // next state from the last transition, and done flag from the last transition
                        replayBuffer.add(
                            firstTransition.state, 
                            firstTransition.action, 
                            nStepReturn, 
                            lastTransition.nextState, 
                            lastTransition.done
                        );
                        
                        // Remove the oldest transition from n-step buffer (sliding window)
                        nStepBuffer.shift();
                    }
                    
                    // If episode ended, flush remaining transitions in n-step buffer
                    if (done) {
                        while (nStepBuffer.length > 0) {
                            const nStepReturn = computeNStepReturn(validGamma);
                            const firstTransition = nStepBuffer[0];
                            const lastTransition = nStepBuffer[nStepBuffer.length - 1];
                            
                            // Each remaining transition uses its proper next state and done flag
                            replayBuffer.add(
                                firstTransition.state, 
                                firstTransition.action, 
                                nStepReturn, 
                                lastTransition.nextState, 
                                lastTransition.done
                            );
                            nStepBuffer.shift();
                        }
                    }
                    
                    // Update canTrain flag if we just crossed the threshold
                    if (!canTrain && replayBuffer.size() >= validMinBufferSize) {
                        canTrain = true;
                    }
                    
                    // Train on minibatch if enough samples and at training interval
                    if (canTrain && stepCount % validTrainEveryNSteps === 0) {
                        // Use prioritized replay sampling
                        const batch = replayBuffer.getPrioritizedBatch(validBatchSize, currentBeta);
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
                
                // Increment episode count for learning rate decay
                episodeCount++;
                
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
                    // Reward shaping components (NEW)
                    rewardComponents: {
                        validPlacement: totalValidPlacementReward,
                        touchingSameType: totalTouchingSameTypeReward,
                        heightDecrease: totalHeightDecreaseReward,
                        heightIncrease: totalHeightIncreaseReward,
                        merge: totalMergeReward,
                        gameOver: totalGameOverPenalty,
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
                        `[Train] NEW Reward components: ` +
                        `validPlace=${totalValidPlacementReward.toFixed(2)}, touchSame=${totalTouchingSameTypeReward.toFixed(2)}, ` +
                        `heightDown=${totalHeightDecreaseReward.toFixed(2)}, heightUp=${totalHeightIncreaseReward.toFixed(2)}, ` +
                        `merge=${totalMergeReward.toFixed(2)}, gameOver=${totalGameOverPenalty.toFixed(0)}`
                    );
                    if (trainCount > 0) {
                        console.log(`[Train] Mean TD error: ${avgMeanTDError.toFixed(4)}`);
                    }
                }
                
                // Save model and metadata after each episode
                try {
                    await model.save('localstorage://fruit-merge-dqn-v1');
                    await saveTrainingMetadata();
                    if (verbose && (episode + 1) % 5 === 0) {
                        console.log(`[Train] Progress saved at episode ${episode + 1}`);
                    }
                } catch (err) {
                    console.error('[Train] Failed to save training progress:', err);
                }
            }
            
            if (verbose) {
                console.log('[Train] Saving final model to localStorage...');
            }
            await model.save('localstorage://fruit-merge-dqn-v1');
            await saveTrainingMetadata();
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
     * Save training metadata (episode count, epsilon, etc.) to localStorage.
     */
    async function saveTrainingMetadata() {
        try {
            const metadata = {
                episodeCount: episodeCount,
                currentLearningRate: currentLearningRate,
                timestamp: Date.now()
            };
            localStorage.setItem('fruit-merge-dqn-metadata', JSON.stringify(metadata));
        } catch (error) {
            console.error('[Train] Failed to save training metadata:', error);
        }
    }
    
    /**
     * Load training metadata from localStorage.
     * @returns {Object|null} Metadata object or null if not found
     */
    function loadTrainingMetadata() {
        try {
            const metadataStr = localStorage.getItem('fruit-merge-dqn-metadata');
            if (metadataStr) {
                return JSON.parse(metadataStr);
            }
        } catch (error) {
            console.error('[Train] Failed to load training metadata:', error);
        }
        return null;
    }
    
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
            
            // Load training metadata to continue from saved state
            const metadata = loadTrainingMetadata();
            if (metadata) {
                episodeCount = metadata.episodeCount || 0;
                currentLearningRate = metadata.currentLearningRate || LEARNING_RATE_INITIAL;
                console.log(`[Train] Loaded training metadata: episodeCount=${episodeCount}, learningRate=${currentLearningRate}`);
            } else {
                // No metadata found, start fresh
                episodeCount = 0;
                currentLearningRate = LEARNING_RATE_INITIAL;
                console.log('[Train] No metadata found, starting from episode 0');
            }
            
            // Create optimizer with loaded learning rate
            optimizer = tf.train.adam(currentLearningRate);
            
            // Recompile the model
            model.compile({
                optimizer: optimizer,
                loss: 'meanSquaredError'
            });
            
            // Create and initialize target model
            targetModel = createQNetwork();
            targetModel.compile({
                optimizer: tf.train.adam(currentLearningRate),
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
     * Select action using NoisyNet Q-values.
     * This is the public API for action selection.
     * Note: For frame stacking, the caller must maintain frame history and provide stacked state.
     * Note: epsilon parameter is ignored - NoisyNet provides exploration through parametric noise.
     * 
     * @param {number[]|Float32Array} state - Stacked state vector (465 elements = 155 * 3 frames)
     * @param {number} [epsilon=0] - Ignored (kept for API compatibility)
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
    console.log('[Train] NEW Model architecture: Embedding + Deeper CNN + NoisyNet.');
    console.log('[Train]   - Board input [15x10] -> Embedding(11, 16) -> [15x10x16]');
    console.log('[Train]   - Conv2D(32,3) -> Conv2D(64,3) -> Conv2D(64,3) -> Conv2D(128,3)');
    console.log('[Train]   - Flatten + Concatenate [currentX, currentY, currentFruit, nextFruit]');
    console.log('[Train]   - Dense(256) -> NoisyDense(256) -> Output(10 column actions)');
    console.log('[Train] Exploration: NoisyNet + epsilon-greedy (ε: 0.4 → 0.01).');
    console.log('[Train] State representation: Frame stacking with ' + FRAME_STACK_SIZE + ' frames (' + BASE_STATE_SIZE + ' * ' + FRAME_STACK_SIZE + ' = ' + STATE_SIZE + ' elements).');
    console.log('[Train] N-step returns: n=' + N_STEP_RETURNS + ' for multi-step DQN.');
    console.log('[Train] Batch size: ' + DEFAULT_BATCH_SIZE + ', Learning rate decay: 1e-4 → 5e-5 (100 eps) → 3e-5 (200 eps).');
    console.log('[Train] Replay buffer: ' + DEFAULT_REPLAY_BUFFER_SIZE + ', warmup: ' + DEFAULT_MIN_BUFFER_SIZE + ', training frequency: every ' + DEFAULT_TRAIN_EVERY_N_STEPS + ' steps.');
    console.log('[Train] Features: Double DQN, rank-based prioritized replay (α=' + PRIORITY_ALPHA + ', β annealed ' + PRIORITY_BETA_START + '→' + PRIORITY_BETA_END + '), NEW reward shaping.');
    console.log('[Train] Stability: Huber loss (δ=' + HUBER_DELTA + '), gradient clipping (max=' + GRADIENT_CLIP_NORM + '), target updates every ' + DEFAULT_TARGET_UPDATE_EVERY + ' steps (τ=' + TAU + ').');
    console.log('[Train] NEW Reward shaping: +' + REWARD_VALID_PLACEMENT + ' valid placement, +' + REWARD_TOUCHING_SAME_TYPE + ' touching same type, ±' + REWARD_HEIGHT_DECREASE + ' height change, +' + REWARD_MERGE + ' merge, ' + REWARD_GAME_OVER + ' game over.');
}
