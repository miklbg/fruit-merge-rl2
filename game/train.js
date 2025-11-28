/**
 * Minimal TensorFlow.js Training Loop for Fruit Merge RL
 * 
 * A simple DQN-style training module that uses the existing RL environment.
 * Features:
 * - Epsilon-greedy action selection
 * - Small replay buffer (capacity 1000)
 * - Single gradient step per environment step
 * - No target network (minimal implementation)
 * 
 * @module train
 */

// ============================================================================
// Configuration
// ============================================================================

const CONFIG = {
    // Model architecture
    hiddenUnits1: 64,
    hiddenUnits2: 64,
    numActions: 4,
    learningRate: 0.0005,
    
    // Replay buffer
    bufferCapacity: 1000,
    batchSize: 32,
    
    // Epsilon-greedy
    epsilonStart: 1.0,
    epsilonEnd: 0.1,
    epsilonDecay: 0.995,
    
    // Training
    gamma: 0.99,  // Discount factor
    
    // Model save key
    modelSaveKey: 'localstorage://fruit-merge-dqn-v1'
};

// ============================================================================
// State
// ============================================================================

let model = null;
let stateSize = null;
let epsilon = CONFIG.epsilonStart;

// Replay buffer: array of {s, a, r, s2, done}
const replayBuffer = [];

// ============================================================================
// Model Functions
// ============================================================================

/**
 * Initialize and compile the TensorFlow.js model.
 * Architecture: Input -> Dense(64, relu) -> Dense(64, relu) -> Dense(4, linear)
 * 
 * @returns {tf.Sequential} The compiled model
 */
function initModel() {
    // Get state size from the RL environment
    if (typeof window.RL === 'undefined' || typeof window.RL.getState !== 'function') {
        throw new Error('RL environment not available. Make sure the game is loaded.');
    }
    
    const state = window.RL.getState();
    stateSize = state.length;
    
    console.log(`[Train] Initializing model with stateSize=${stateSize}, actions=${CONFIG.numActions}`);
    
    // Create the model
    model = tf.sequential();
    
    // Input layer + first hidden layer
    model.add(tf.layers.dense({
        inputShape: [stateSize],
        units: CONFIG.hiddenUnits1,
        activation: 'relu',
        kernelInitializer: 'heNormal'
    }));
    
    // Second hidden layer
    model.add(tf.layers.dense({
        units: CONFIG.hiddenUnits2,
        activation: 'relu',
        kernelInitializer: 'heNormal'
    }));
    
    // Output layer (Q-values for each action)
    model.add(tf.layers.dense({
        units: CONFIG.numActions,
        activation: 'linear',
        kernelInitializer: 'heNormal'
    }));
    
    // Compile with Adam optimizer and MSE loss
    model.compile({
        optimizer: tf.train.adam(CONFIG.learningRate),
        loss: 'meanSquaredError'
    });
    
    console.log('[Train] Model initialized and compiled');
    model.summary();
    
    return model;
}

/**
 * Select an action using epsilon-greedy policy.
 * 
 * @param {number[]} state - Current state array
 * @returns {number} Action index (0-3)
 */
function selectAction(state) {
    // Epsilon-greedy: explore with probability epsilon
    if (Math.random() < epsilon) {
        return Math.floor(Math.random() * CONFIG.numActions);
    }
    
    // Exploit: select action with highest Q-value
    const stateTensor = tf.tensor2d([state]);
    const qValues = model.predict(stateTensor);
    const action = qValues.argMax(1).dataSync()[0];
    
    // Clean up tensors
    stateTensor.dispose();
    qValues.dispose();
    
    return action;
}

// ============================================================================
// Replay Buffer Functions
// ============================================================================

/**
 * Store an experience tuple in the replay buffer.
 * Uses a circular buffer that overwrites oldest experiences when full.
 * 
 * @param {number[]} s - Current state
 * @param {number} a - Action taken
 * @param {number} r - Reward received
 * @param {number[]} s2 - Next state
 * @param {boolean} done - Whether episode ended
 */
function storeExperience(s, a, r, s2, done) {
    const experience = { s, a, r, s2, done };
    
    if (replayBuffer.length >= CONFIG.bufferCapacity) {
        // Remove oldest experience
        replayBuffer.shift();
    }
    
    replayBuffer.push(experience);
}

/**
 * Sample a random batch of experiences from the replay buffer.
 * 
 * @param {number} batchSize - Number of experiences to sample
 * @returns {Object[]} Array of experience objects
 */
function sampleBatch(batchSize) {
    const batch = [];
    const bufferSize = replayBuffer.length;
    
    for (let i = 0; i < batchSize; i++) {
        const idx = Math.floor(Math.random() * bufferSize);
        batch.push(replayBuffer[idx]);
    }
    
    return batch;
}

// ============================================================================
// Training Functions
// ============================================================================

/**
 * Train the model on a single batch sampled from the replay buffer.
 * Uses simple DQN update: Q(s,a) = r + gamma * max_a' Q(s', a')
 * 
 * @returns {number|null} Training loss, or null if buffer too small
 */
async function trainOnBatch() {
    // Don't train if buffer doesn't have enough samples
    if (replayBuffer.length < CONFIG.batchSize) {
        return null;
    }
    
    // Sample a batch
    const batch = sampleBatch(CONFIG.batchSize);
    
    // Prepare batch data
    const states = batch.map(exp => exp.s);
    const nextStates = batch.map(exp => exp.s2);
    const actions = batch.map(exp => exp.a);
    const rewards = batch.map(exp => exp.r);
    const dones = batch.map(exp => exp.done);
    
    // Convert to tensors
    const statesTensor = tf.tensor2d(states);
    const nextStatesTensor = tf.tensor2d(nextStates);
    
    // Get current Q-values for all actions
    const currentQs = model.predict(statesTensor);
    
    // Get next Q-values for computing targets
    const nextQs = model.predict(nextStatesTensor);
    const maxNextQs = nextQs.max(1).dataSync();
    
    // Compute target Q-values
    const currentQsArray = currentQs.arraySync();
    for (let i = 0; i < CONFIG.batchSize; i++) {
        const targetQ = dones[i] 
            ? rewards[i]  // Terminal state: no future reward
            : rewards[i] + CONFIG.gamma * maxNextQs[i];
        
        // Update only the Q-value for the action taken
        currentQsArray[i][actions[i]] = targetQ;
    }
    
    // Create target tensor
    const targetsTensor = tf.tensor2d(currentQsArray);
    
    // Train the model
    const result = await model.fit(statesTensor, targetsTensor, {
        epochs: 1,
        verbose: 0
    });
    
    const loss = result.history.loss[0];
    
    // Clean up tensors
    statesTensor.dispose();
    nextStatesTensor.dispose();
    currentQs.dispose();
    nextQs.dispose();
    targetsTensor.dispose();
    
    return loss;
}

/**
 * Run training for multiple episodes.
 * Each episode runs until terminal state, collecting experiences and training.
 * 
 * @param {number} numEpisodes - Number of episodes to train
 * @returns {Promise<Object>} Training statistics
 */
async function trainEpisodes(numEpisodes) {
    // Ensure model is initialized
    if (!model) {
        initModel();
    }
    
    console.log(`[Train] Starting training for ${numEpisodes} episodes`);
    console.log(`[Train] Initial epsilon: ${epsilon.toFixed(4)}`);
    
    const startTime = performance.now();
    const episodeRewards = [];
    const episodeLengths = [];
    let totalSteps = 0;
    let totalLoss = 0;
    let lossCount = 0;
    
    for (let episode = 0; episode < numEpisodes; episode++) {
        // Reset the episode
        window.RL.resetEpisode();
        
        let state = window.RL.getState();
        let episodeReward = 0;
        let stepCount = 0;
        
        // Run episode until terminal
        while (!window.RL.isTerminal()) {
            // Select action using epsilon-greedy
            const action = selectAction(state);
            
            // Execute action and get reward
            // Note: RL.step() returns just the reward number, not { reward, done }
            const reward = window.RL.step(action);
            const done = window.RL.isTerminal();
            
            // Get next state
            const nextState = window.RL.getState();
            
            // Store experience
            storeExperience(state, action, reward, nextState, done);
            
            // Train on a batch
            const loss = await trainOnBatch();
            if (loss !== null) {
                totalLoss += loss;
                lossCount++;
            }
            
            // Update state
            state = nextState;
            episodeReward += reward;
            stepCount++;
            totalSteps++;
            
            // Yield to prevent blocking
            if (stepCount % 100 === 0) {
                await new Promise(resolve => setTimeout(resolve, 0));
            }
            
            // Safety limit to prevent infinite loops
            if (stepCount >= 10000) {
                console.warn(`[Train] Episode ${episode} exceeded 10000 steps, breaking`);
                break;
            }
        }
        
        // Decay epsilon
        epsilon = Math.max(CONFIG.epsilonEnd, epsilon * CONFIG.epsilonDecay);
        
        // Record episode stats
        episodeRewards.push(episodeReward);
        episodeLengths.push(stepCount);
        
        // Log progress every 5 episodes
        if ((episode + 1) % 5 === 0 || episode === numEpisodes - 1) {
            const avgReward = episodeRewards.slice(-5).reduce((a, b) => a + b, 0) / Math.min(5, episodeRewards.length);
            const avgLoss = lossCount > 0 ? totalLoss / lossCount : 0;
            console.log(
                `[Train] Episode ${episode + 1}/${numEpisodes}: ` +
                `reward=${episodeReward.toFixed(2)}, avgReward(5)=${avgReward.toFixed(2)}, ` +
                `steps=${stepCount}, epsilon=${epsilon.toFixed(4)}, avgLoss=${avgLoss.toFixed(6)}`
            );
        }
    }
    
    const endTime = performance.now();
    const totalTime = (endTime - startTime) / 1000;
    
    // Compute final statistics
    const avgReward = episodeRewards.reduce((a, b) => a + b, 0) / episodeRewards.length;
    const maxReward = Math.max(...episodeRewards);
    const avgSteps = episodeLengths.reduce((a, b) => a + b, 0) / episodeLengths.length;
    const avgLoss = lossCount > 0 ? totalLoss / lossCount : 0;
    
    console.log(`[Train] ========== TRAINING COMPLETE ==========`);
    console.log(`[Train] Episodes: ${numEpisodes}`);
    console.log(`[Train] Total steps: ${totalSteps}`);
    console.log(`[Train] Total time: ${totalTime.toFixed(2)}s`);
    console.log(`[Train] Average reward: ${avgReward.toFixed(2)}`);
    console.log(`[Train] Max reward: ${maxReward.toFixed(2)}`);
    console.log(`[Train] Average steps: ${avgSteps.toFixed(2)}`);
    console.log(`[Train] Final epsilon: ${epsilon.toFixed(4)}`);
    console.log(`[Train] Average loss: ${avgLoss.toFixed(6)}`);
    console.log(`[Train] Buffer size: ${replayBuffer.length}`);
    console.log(`[Train] ==========================================`);
    
    // Save the model
    try {
        await model.save(CONFIG.modelSaveKey);
        console.log(`[Train] Model saved to ${CONFIG.modelSaveKey}`);
    } catch (error) {
        console.error('[Train] Failed to save model:', error);
    }
    
    return {
        numEpisodes,
        totalSteps,
        totalTime,
        avgReward,
        maxReward,
        avgSteps,
        finalEpsilon: epsilon,
        avgLoss,
        bufferSize: replayBuffer.length,
        episodeRewards,
        episodeLengths
    };
}

// ============================================================================
// Public API
// ============================================================================

/**
 * Load a previously saved model from localStorage.
 * 
 * @returns {Promise<boolean>} True if model loaded successfully
 */
async function loadModel() {
    try {
        model = await tf.loadLayersModel(CONFIG.modelSaveKey);
        
        // Recompile the model (needed after loading)
        model.compile({
            optimizer: tf.train.adam(CONFIG.learningRate),
            loss: 'meanSquaredError'
        });
        
        // Get state size from loaded model
        stateSize = model.input.shape[1];
        
        console.log(`[Train] Model loaded from ${CONFIG.modelSaveKey}`);
        model.summary();
        
        return true;
    } catch (error) {
        console.warn('[Train] Could not load model:', error.message);
        return false;
    }
}

/**
 * Reset training state (epsilon, replay buffer).
 */
function resetTraining() {
    epsilon = CONFIG.epsilonStart;
    replayBuffer.length = 0;
    console.log('[Train] Training state reset');
}

/**
 * Get the current model (for inference).
 * 
 * @returns {tf.Sequential|null} The model or null if not initialized
 */
function getModel() {
    return model;
}

/**
 * Get the current epsilon value.
 * 
 * @returns {number} Current epsilon
 */
function getEpsilon() {
    return epsilon;
}

/**
 * Set epsilon value manually (for testing).
 * 
 * @param {number} value - New epsilon value
 */
function setEpsilon(value) {
    epsilon = Math.max(0, Math.min(1, value));
}

// ============================================================================
// Exports and Window Integration
// ============================================================================

// Export functions for ES module usage
export {
    initModel,
    selectAction,
    storeExperience,
    trainOnBatch,
    trainEpisodes,
    loadModel,
    resetTraining,
    getModel,
    getEpsilon,
    setEpsilon,
    CONFIG
};

/**
 * Initialize the training module and expose window.RL.train
 * This should be called after the game and TensorFlow.js are loaded.
 */
export function initTraining() {
    // Ensure RL interface exists
    window.RL = window.RL || {};
    
    // Expose the main training function
    window.RL.train = async function(numEpisodes = 50) {
        return await trainEpisodes(numEpisodes);
    };
    
    // Expose additional utilities
    window.RL.initModel = initModel;
    window.RL.loadModel = loadModel;
    window.RL.resetTraining = resetTraining;
    window.RL.getTrainingModel = getModel;
    window.RL.selectAction = selectAction;
    
    console.log('[Train] Training module initialized. Use window.RL.train(numEpisodes) to start training.');
}
