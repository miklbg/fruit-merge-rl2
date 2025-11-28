/**
 * FastSim - Headless Simulation Module for Fruit Merge RL
 * 
 * Provides fast-forward simulation capabilities for running many episodes
 * quickly without rendering or UI overhead. Uses synchronous game loops
 * for maximum speed.
 * 
 * @module fastSim
 */

/**
 * @typedef {Object} SimulationResult
 * @property {number} episode - Episode index (0-based)
 * @property {number} steps - Number of steps taken in the episode
 * @property {number} finalScore - Final score at end of episode
 */

/**
 * Creates the FastSim controller for headless simulation.
 * 
 * This function should be called after the game is fully initialized
 * and the RL API is available on window.RL.
 * 
 * @param {Object} gameContext - Game context containing references to game objects
 * @param {Function} gameContext.engine - Function that returns Matter.js engine instance
 * @param {Function} gameContext.runner - Function that returns Matter.js runner instance
 * @param {Function} gameContext.render - Function that returns Matter.js render instance
 * @param {Function} gameContext.getScore - Function to get current score
 * @param {Function} gameContext.getIsGameOver - Function to check if game is over
 * @returns {Object} FastSim controller with run() and stop() methods
 */
export function createFastSimController(gameContext) {
    const { Engine, Runner, Render } = Matter;
    
    // Internal state
    let isRunning = false;
    let shouldStop = false;
    
    // Number of valid actions (0-3: left, right, center, drop)
    const NUM_ACTIONS = 4;
    
    // Physics timestep (60 FPS equivalent)
    const DELTA_TIME = 1000 / 60;
    
    // Maximum steps per episode to prevent infinite loops
    const MAX_STEPS_PER_EPISODE = 10000;
    
    /**
     * Select a random action (dummy policy)
     * @returns {number} Action index (0-3)
     */
    function selectRandomAction() {
        return Math.floor(Math.random() * NUM_ACTIONS);
    }
    
    /**
     * Run a single physics step without rendering
     * @param {Object} engine - Matter.js engine
     */
    function stepPhysics(engine) {
        Engine.update(engine, DELTA_TIME);
    }
    
    /**
     * Run multiple episodes in headless mode
     * 
     * @param {number} numEpisodes - Number of episodes to run
     * @param {Object} [options={}] - Optional configuration options
     * @param {boolean} [options.renderOnGameOver=false] - When true, renders the fruits when game over occurs for each episode
     * @returns {Promise<SimulationResult[]>} Array of results, one per episode
     */
    async function run(numEpisodes, options = {}) {
        const { renderOnGameOver = false } = options;
        // Validate inputs
        if (typeof numEpisodes !== 'number' || numEpisodes < 1) {
            throw new Error(`Invalid numEpisodes: ${numEpisodes}. Must be a positive number.`);
        }
        
        // Check if RL interface is available
        if (typeof window.RL === 'undefined') {
            throw new Error('RL interface not available. Make sure the game is initialized.');
        }
        
        // Check for required RL methods (using resetEpisode for headless mode)
        if (typeof window.RL.resetEpisode !== 'function' ||
            typeof window.RL.getState !== 'function' ||
            typeof window.RL.step !== 'function' ||
            typeof window.RL.isTerminal !== 'function' ||
            typeof window.RL.getReward !== 'function' ||
            typeof window.RL.setHeadlessMode !== 'function' ||
            typeof window.RL.tickCooldown !== 'function') {
            throw new Error('RL interface incomplete. Required: resetEpisode, getState, step, isTerminal, getReward, setHeadlessMode, tickCooldown');
        }
        
        // If already running, stop the current simulation and wait for it to finish
        if (isRunning) {
            console.log('[FastSim] Simulation already running. Stopping current simulation...');
            stop();
            // Wait for the current run to finish with a timeout to prevent indefinite waiting
            const maxWaitTime = 5000; // 5 seconds max wait
            const pollInterval = 50; // 50ms polling interval
            let waitedTime = 0;
            await new Promise(resolve => {
                const checkInterval = setInterval(() => {
                    waitedTime += pollInterval;
                    if (!isRunning || waitedTime >= maxWaitTime) {
                        clearInterval(checkInterval);
                        if (waitedTime >= maxWaitTime && isRunning) {
                            console.warn('[FastSim] Timeout waiting for previous simulation to stop. Proceeding anyway.');
                            isRunning = false;
                        }
                        resolve();
                    }
                }, pollInterval);
            });
        }
        
        isRunning = true;
        shouldStop = false;
        
        const results = [];
        const startTime = performance.now();
        
        console.log(`[FastSim] Starting simulation: ${numEpisodes} episodes`);
        
        // Enable headless mode to disable rendering and audio
        window.RL.setHeadlessMode(true);
        
        // Stop the normal game runner and render to take control of physics updates
        const runner = gameContext.runner();
        const render = gameContext.render();
        
        if (runner) {
            Runner.stop(runner);
        }
        if (render) {
            Render.stop(render);
        }
        
        try {
            for (let episodeIdx = 0; episodeIdx < numEpisodes; episodeIdx++) {
                if (shouldStop) {
                    console.log(`[FastSim] Stopped early at episode ${episodeIdx}`);
                    break;
                }
                
                // Reset for new episode using lightweight resetEpisode
                try {
                    window.RL.resetEpisode();
                } catch (resetError) {
                    console.error(`[FastSim] Error resetting episode ${episodeIdx}:`, resetError);
                    console.error(resetError.stack);
                    throw resetError;
                }
                
                // Get engine reference (should be stable in headless mode)
                const currentEngine = gameContext.engine();
                
                let stepCount = 0;
                const episodeStartTime = performance.now();
                
                console.log(`[FastSim] Episode ${episodeIdx} started`);
                
                // Episode loop
                while (!window.RL.isTerminal() && stepCount < MAX_STEPS_PER_EPISODE) {
                    if (shouldStop) {
                        break;
                    }
                    
                    try {
                        // Get current state (validates state is working)
                        const state = window.RL.getState();
                        
                        // Validate state
                        if (!Array.isArray(state)) {
                            throw new Error(`Invalid state: expected array, got ${typeof state}`);
                        }
                        
                        // Check for NaN in state
                        for (let i = 0; i < state.length; i++) {
                            if (Number.isNaN(state[i])) {
                                throw new Error(`NaN detected in state at index ${i}`);
                            }
                        }
                        
                        // Select and execute action (random policy)
                        const action = selectRandomAction();
                        window.RL.step(action);
                        
                        // Advance physics simulation
                        stepPhysics(currentEngine);
                        
                        // Tick the step-based cooldown counter
                        window.RL.tickCooldown();
                        
                        // Get reward (optional, for future use)
                        window.RL.getReward();
                        
                        stepCount++;
                        
                        // Yield to event loop periodically to prevent blocking
                        if (stepCount % 500 === 0) {
                            await new Promise(resolve => setTimeout(resolve, 0));
                        }
                        
                    } catch (stepError) {
                        console.error(`[FastSim] Error at episode ${episodeIdx}, step ${stepCount}:`, stepError);
                        console.error(stepError.stack);
                        throw stepError;
                    }
                }
                
                // Render the game state when game over occurs if option is enabled
                if (renderOnGameOver && window.RL.isTerminal()) {
                    const render = gameContext.render();
                    if (render) {
                        // Temporarily enable rendering for a single frame
                        window.RL.setHeadlessMode(false);
                        Render.world(render);
                        window.RL.setHeadlessMode(true);
                        console.log(`[FastSim] Episode ${episodeIdx}: Rendered fruits at game over state`);
                    }
                }
                
                // Record episode result
                const finalScore = gameContext.getScore();
                const episodeTime = performance.now() - episodeStartTime;
                
                const result = {
                    episode: episodeIdx,
                    steps: stepCount,
                    finalScore: finalScore
                };
                
                results.push(result);
                
                console.log(
                    `[FastSim] Episode ${episodeIdx} ended: ` +
                    `steps=${stepCount}, score=${finalScore}, time=${episodeTime.toFixed(2)}ms`
                );
            }
            
        } finally {
            // Restore normal mode
            isRunning = false;
            
            // Disable headless mode
            window.RL.setHeadlessMode(false);
            
            // Reset the game to leave it in a clean state with rendering restored
            // RL.reset() calls handleRestart() which calls initGame()
            // initGame() restarts Render.run() and Runner.run() automatically
            try {
                if (typeof window.RL.reset === 'function') {
                    window.RL.reset();
                }
            } catch (e) {
                console.warn('[FastSim] Failed to reset game after simulation:', e);
            }
        }
        
        // Calculate and log summary statistics
        const endTime = performance.now();
        const totalTime = endTime - startTime;
        
        if (results.length > 0) {
            const scores = results.map(r => r.finalScore);
            const steps = results.map(r => r.steps);
            
            const avgScore = scores.reduce((a, b) => a + b, 0) / scores.length;
            const maxScore = Math.max(...scores);
            const minScore = Math.min(...scores);
            const avgSteps = steps.reduce((a, b) => a + b, 0) / steps.length;
            const timePerEpisode = totalTime / results.length;
            
            console.log(`[FastSim] ========== SIMULATION SUMMARY ==========`);
            console.log(`[FastSim] Episodes completed: ${results.length}/${numEpisodes}`);
            console.log(`[FastSim] Total time: ${totalTime.toFixed(2)}ms`);
            console.log(`[FastSim] Time per episode: ${timePerEpisode.toFixed(2)}ms`);
            console.log(`[FastSim] Average score: ${avgScore.toFixed(2)}`);
            console.log(`[FastSim] Max score: ${maxScore}`);
            console.log(`[FastSim] Min score: ${minScore}`);
            console.log(`[FastSim] Average steps: ${avgSteps.toFixed(2)}`);
            console.log(`[FastSim] ==========================================`);
        }
        
        return results;
    }
    
    /**
     * Stop any running simulation immediately
     */
    function stop() {
        if (isRunning) {
            console.log('[FastSim] Stop requested');
            shouldStop = true;
        }
    }
    
    /**
     * Check if simulation is currently running
     * @returns {boolean} True if running
     */
    function isSimulationRunning() {
        return isRunning;
    }
    
    return {
        run,
        stop,
        isRunning: isSimulationRunning
    };
}

/**
 * Initialize FastSim and expose it on window.FastSim
 * This should be called after the game is fully loaded.
 * 
 * @param {Object} gameContext - Game context with engine, runner, render references
 */
export function initFastSim(gameContext) {
    const controller = createFastSimController(gameContext);
    
    window.FastSim = {
        /**
         * Run headless simulation for the specified number of episodes
         * @param {number} numEpisodes - Number of episodes to run
         * @param {Object} [options={}] - Optional configuration options
         * @param {boolean} [options.renderOnGameOver=false] - When true, renders the fruits when game over occurs for each episode
         * @returns {Promise<SimulationResult[]>} Array of results
         */
        run: controller.run,
        
        /**
         * Stop any running simulation immediately
         */
        stop: controller.stop
    };
    
    console.log('[FastSim] Initialized. Use FastSim.run(numEpisodes) or FastSim.run(numEpisodes, { renderOnGameOver: true }) to start simulation.');
    
    return controller;
}
