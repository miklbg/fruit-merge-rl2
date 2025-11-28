/**
 * train.js - Minimal TensorFlow.js Training Loop for Fruit Merge RL
 * 
 * Implements a simple DQN training loop using the FastSim environment.
 * This is a minimal test loop to verify the RL environment works correctly.
 * 
 * Usage:
 *   RL.initModel();        // Build and compile model
 *   RL.train(5);           // Run 5 episodes of training
 * 
 * @module train
 */

// Game context reference (set by initTraining)
let gameContext = null;

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
    
    // Physics timestep (60 FPS equivalent)
    const DELTA_TIME = 1000 / 60;
    
    // Maximum steps per episode to prevent infinite loops
    const MAX_STEPS_PER_EPISODE = 10000;
    
    // Model reference
    let model = null;
    
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
        
        // Compile with Adam optimizer and MSE loss
        model.compile({
            optimizer: tf.train.adam(LEARNING_RATE),
            loss: 'meanSquaredError'
        });
        
        console.log('[Train] Model built and compiled successfully.');
        console.log('[Train] Model summary:');
        model.summary();
        
        return model;
    };
    
    /**
     * Get Q-values for a given state.
     * 
     * @param {number[]} state - State array of length 155
     * @returns {tf.Tensor} Q-values tensor of shape [1, 4]
     */
    function getQValues(state) {
        return tf.tidy(() => {
            const stateTensor = tf.tensor2d([state], [1, STATE_SIZE]);
            return model.predict(stateTensor);
        });
    }
    
    /**
     * Select action using pure exploitation (argmax of Q-values).
     * 
     * @param {number[]} state - State array of length 155
     * @returns {number} Action index (0-3)
     */
    function selectAction(state) {
        const qValues = getQValues(state);
        const action = qValues.argMax(1).dataSync()[0];
        qValues.dispose();
        return action;
    }
    
    /**
     * Perform a single Q-learning update.
     * 
     * Q(s,a) ← r + γ * max_a' Q(s', a')
     * 
     * @param {number[]} state - Current state
     * @param {number} action - Action taken
     * @param {number} reward - Reward received
     * @param {number[]} nextState - Next state
     * @param {boolean} done - Whether episode is terminal
     * @returns {Promise<number>} Training loss
     */
    async function updateQ(state, action, reward, nextState, done) {
        // Compute target Q-value
        let targetQ;
        
        if (done) {
            // Terminal state: target = reward only
            targetQ = reward;
        } else {
            // Non-terminal: target = r + γ * max_a' Q(s', a')
            const nextQValues = getQValues(nextState);
            const maxNextQ = nextQValues.max().dataSync()[0];
            nextQValues.dispose();
            targetQ = reward + GAMMA * maxNextQ;
        }
        
        // Get current Q-values and update the target for the taken action
        const currentQValues = getQValues(state);
        const qValuesArray = currentQValues.dataSync().slice();
        currentQValues.dispose();
        
        // Update only the Q-value for the taken action
        qValuesArray[action] = targetQ;
        
        // Perform gradient update
        const stateTensor = tf.tensor2d([state], [1, STATE_SIZE]);
        const targetTensor = tf.tensor2d([qValuesArray], [1, NUM_ACTIONS]);
        
        const result = await model.fit(stateTensor, targetTensor, {
            epochs: 1,
            verbose: 0
        });
        
        const loss = result.history.loss[0];
        
        // Clean up tensors
        stateTensor.dispose();
        targetTensor.dispose();
        
        return loss;
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
     * Uses FastSim for headless simulation.
     * 
     * @param {number} numEpisodes - Number of episodes to train
     * @returns {Promise<Object>} Training results summary
     */
    window.RL.train = async function(numEpisodes) {
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
        
        console.log(`[Train] Starting training: ${numEpisodes} episodes`);
        const startTime = performance.now();
        
        const results = [];
        
        // Enable headless mode
        window.RL.setHeadlessMode(true);
        
        // Validate Matter.js is available
        if (typeof Matter === 'undefined') {
            console.error('[Train] Matter.js is not loaded. Make sure the physics engine is initialized.');
            window.RL.setHeadlessMode(false);
            return null;
        }
        
        // Stop the normal game runner to take control of physics
        const { Runner, Render } = Matter;
        const runner = gameContext.runner();
        const render = gameContext.render();
        
        if (runner) {
            Runner.stop(runner);
        }
        if (render) {
            Render.stop(render);
        }
        
        try {
            for (let episode = 0; episode < numEpisodes; episode++) {
                console.log(`[Train] Episode ${episode + 1}/${numEpisodes} starting...`);
                
                // Reset environment
                window.RL.resetEpisode();
                
                // Get fresh engine reference after reset
                const engine = gameContext.engine();
                
                let stepCount = 0;
                let totalReward = 0;
                let totalLoss = 0;
                const episodeStartTime = performance.now();
                
                // Get initial state
                let state = window.RL.getState();
                
                // Episode loop
                while (!window.RL.isTerminal() && stepCount < MAX_STEPS_PER_EPISODE) {
                    // Select action (pure exploitation - argmax Q-values)
                    const action = selectAction(state);
                    
                    // Log Q-values and render a single frame periodically for debugging
                    if (stepCount % 500 === 0) {
                        const qValues = getQValues(state);
                        const qArray = qValues.dataSync();
                        console.log(`[Train] Episode ${episode + 1}, Step ${stepCount}: Q-values = [${qArray.map(q => q.toFixed(4)).join(', ')}]`);
                        qValues.dispose();
                        
                        // Render a single frame for visual debugging
                        if (render) {
                            window.RL.setHeadlessMode(false);
                            Render.world(render);
                            window.RL.setHeadlessMode(true);
                        }
                    }
                    
                    // Execute action
                    window.RL.step(action);
                    
                    // Step physics simulation
                    stepPhysics(engine);
                    
                    // Tick cooldown (this is needed for drop timing)
                    window.RL.tickCooldown();
                    
                    // Get reward
                    const reward = window.RL.getReward();
                    totalReward += reward;
                    
                    // Get next state
                    const nextState = window.RL.getState();
                    const done = window.RL.isTerminal();
                    
                    // Perform Q-learning update
                    const loss = await updateQ(state, action, reward, nextState, done);
                    totalLoss += loss;
                    
                    // Update state
                    state = nextState;
                    stepCount++;
                    
                    // Yield to event loop periodically to prevent blocking
                    // Use a small timeout (1ms) for better efficiency
                    if (stepCount % 100 === 0) {
                        await new Promise(resolve => setTimeout(resolve, 1));
                    }
                }
                
                const episodeTime = performance.now() - episodeStartTime;
                const avgLoss = stepCount > 0 ? totalLoss / stepCount : 0;
                
                const episodeResult = {
                    episode: episode + 1,
                    steps: stepCount,
                    totalReward: totalReward,
                    avgLoss: avgLoss,
                    timeMs: episodeTime
                };
                
                results.push(episodeResult);
                
                console.log(
                    `[Train] Episode ${episode + 1} ended: ` +
                    `steps=${stepCount}, reward=${totalReward.toFixed(2)}, ` +
                    `avgLoss=${avgLoss.toFixed(6)}, time=${episodeTime.toFixed(2)}ms`
                );
            }
            
            // Save model to localStorage
            console.log('[Train] Saving model to localStorage...');
            await model.save('localstorage://fruit-merge-dqn-v1');
            console.log('[Train] Model saved successfully to localstorage://fruit-merge-dqn-v1');
            
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
        const avgReward = results.reduce((sum, r) => sum + r.totalReward, 0) / results.length;
        const avgSteps = totalSteps / results.length;
        
        console.log(`[Train] ========== TRAINING SUMMARY ==========`);
        console.log(`[Train] Episodes completed: ${results.length}`);
        console.log(`[Train] Total time: ${totalTime.toFixed(2)}ms`);
        console.log(`[Train] Average reward per episode: ${avgReward.toFixed(2)}`);
        console.log(`[Train] Average steps per episode: ${avgSteps.toFixed(2)}`);
        console.log(`[Train] Total steps: ${totalSteps}`);
        console.log(`[Train] ==========================================`);
        
        return {
            episodes: results,
            totalTime: totalTime,
            avgReward: avgReward,
            avgSteps: avgSteps
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
            
            // Recompile the model
            model.compile({
                optimizer: tf.train.adam(LEARNING_RATE),
                loss: 'meanSquaredError'
            });
            
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
    
    console.log('[Train] Training module initialized.');
    console.log('[Train] Use RL.initModel() to build the model, then RL.train(numEpisodes) to train.');
}
