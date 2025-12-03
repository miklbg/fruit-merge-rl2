/**
 * FastSim - Optimized Headless Simulation Module for Fruit Merge RL
 * 
 * Provides fast-forward simulation capabilities for running many episodes
 * quickly without rendering or UI overhead. Uses synchronous game loops
 * for maximum speed.
 * 
 * Performance optimizations:
 * - No temporary object creation in hot loops
 * - No array spreading or JSON operations
 * - Minimal function calls in inner loops
 * - Pre-cached references to avoid property lookups
 * 
 * Physics stability:
 * - Deterministic physics with fixed seed (Matter.Common._seed = 12345)
 * - Fixed timestep (no real-time randomness)
 * - Reduced restitution for stability
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
    // Make physics deterministic - CRITICAL FOR REPRODUCIBLE TRAINING
    // Note: This seed value should match PHYSICS_SEED in index.html
    const PHYSICS_SEED = 12345;
    if (typeof Matter !== 'undefined' && Matter.Common) {
        Matter.Common._seed = PHYSICS_SEED;
        console.log('[FastSim] Physics seed set to ' + PHYSICS_SEED + ' for deterministic behavior');
    }
    
    // Cache Matter.js references to avoid repeated property lookups
    const MatterEngine = Matter.Engine;
    const MatterRunner = Matter.Runner;
    const MatterRender = Matter.Render;
    
    // Internal state
    let isRunning = false;
    let shouldStop = false;
    
    // Number of valid actions (0-9: column actions)
    const NUM_ACTIONS = 10;
    
    // Physics timestep (60 FPS equivalent) - FIXED for determinism
    const DELTA_TIME = 1000 / 60;
    
    // Maximum steps per episode to prevent infinite loops
    const MAX_STEPS_PER_EPISODE = 10000;
    
    // Pre-cached random for faster action selection
    const mathRandom = Math.random;
    const mathFloor = Math.floor;
    
    /**
     * Select a random action (dummy policy)
     * Uses cached math functions for speed
     * @returns {number} Action index (0-9 for column actions)
     */
    function selectRandomAction() {
        return mathFloor(mathRandom() * NUM_ACTIONS);
    }
    
    /**
     * Run a single physics step without rendering - DETERMINISTIC
     * @param {Object} engine - Matter.js engine
     */
    function stepPhysics(engine) {
        MatterEngine.update(engine, DELTA_TIME);
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
        
        // Check if already running
        if (isRunning) {
            throw new Error('FastSim is already running. Call stop() first.');
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
            MatterRunner.stop(runner);
        }
        if (render) {
            MatterRender.stop(render);
        }
        
        // Cache RL method references for faster access in hot loops
        const rlResetEpisode = window.RL.resetEpisode;
        const rlGetState = window.RL.getState;
        const rlStep = window.RL.step;
        const rlIsTerminal = window.RL.isTerminal;
        const rlGetReward = window.RL.getReward;
        const rlTickCooldown = window.RL.tickCooldown;
        const rlSetHeadlessMode = window.RL.setHeadlessMode;
        const getEngine = gameContext.engine;
        const getScore = gameContext.getScore;
        
        try {
            for (let episodeIdx = 0; episodeIdx < numEpisodes; episodeIdx++) {
                if (shouldStop) {
                    console.log(`[FastSim] Stopped early at episode ${episodeIdx}`);
                    break;
                }
                
                // Reset for new episode using lightweight resetEpisode
                rlResetEpisode();
                
                // Get engine reference (should be stable in headless mode)
                const currentEngine = getEngine();
                
                let stepCount = 0;
                const episodeStartTime = performance.now();
                
                console.log(`[FastSim] Episode ${episodeIdx} started`);
                
                // Episode loop - optimized hot path
                // Avoid try-catch inside loop for performance
                while (!rlIsTerminal() && stepCount < MAX_STEPS_PER_EPISODE) {
                    if (shouldStop) {
                        break;
                    }
                    
                    // Select and execute action (random policy)
                    // Note: State validation moved outside hot path for performance
                    const action = selectRandomAction();
                    rlStep(action);
                    
                    // Advance physics simulation
                    stepPhysics(currentEngine);
                    
                    // Tick the step-based cooldown counter
                    rlTickCooldown();
                    
                    // Get reward (optional, for future use)
                    rlGetReward();
                    
                    stepCount++;
                    
                    // Yield to event loop periodically to prevent blocking
                    if (stepCount % 500 === 0) {
                        await new Promise(resolve => setTimeout(resolve, 0));
                    }
                }
                
                // Render the game state when game over occurs if option is enabled
                if (renderOnGameOver && rlIsTerminal()) {
                    const currentRender = gameContext.render();
                    if (currentRender) {
                        // Temporarily enable rendering for a single frame
                        rlSetHeadlessMode(false);
                        MatterRender.world(currentRender);
                        rlSetHeadlessMode(true);
                        console.log(`[FastSim] Episode ${episodeIdx}: Rendered fruits at game over state`);
                    }
                }
                
                // Record episode result
                const finalScore = getScore();
                const episodeTime = performance.now() - episodeStartTime;
                
                // Reuse object structure - avoid creating new objects when possible
                results.push({
                    episode: episodeIdx,
                    steps: stepCount,
                    finalScore: finalScore
                });
                
                console.log(
                    `[FastSim] Episode ${episodeIdx} ended: ` +
                    `steps=${stepCount}, score=${finalScore}, time=${episodeTime.toFixed(2)}ms`
                );
            }
            
        } finally {
            // Restore normal mode
            isRunning = false;
            
            // Disable headless mode
            rlSetHeadlessMode(false);
            
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
        // Use manual loops instead of spread operators for performance
        const endTime = performance.now();
        const totalTime = endTime - startTime;
        
        if (results.length > 0) {
            let totalScore = 0;
            let totalSteps = 0;
            let maxScore = results[0].finalScore;
            let minScore = results[0].finalScore;
            
            for (let i = 0; i < results.length; i++) {
                const r = results[i];
                totalScore += r.finalScore;
                totalSteps += r.steps;
                if (r.finalScore > maxScore) maxScore = r.finalScore;
                if (r.finalScore < minScore) minScore = r.finalScore;
            }
            
            const avgScore = totalScore / results.length;
            const avgSteps = totalSteps / results.length;
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
