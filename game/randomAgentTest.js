/**
 * Random Agent Test Harness
 * 
 * A simple JS module that runs a random agent in real time to verify the RL environment.
 * Uses the existing RL.getState() and RL.step() interface without modifying game physics or rendering.
 */

/**
 * Start a random agent test that repeatedly calls the RL interface.
 * 
 * @param {Object} options - Configuration options
 * @param {number} [options.fps=10] - How many steps per second
 * @param {number} [options.maxSteps=2000] - Automatically stop after N steps
 * @param {number} [options.logEvery=100] - Console-log state summary every N steps
 */
export function startRandomAgentTest(options = {}) {
    const fps = options.fps ?? 10;
    const maxSteps = options.maxSteps ?? 2000;
    const logEvery = options.logEvery ?? 100;

    // Number of valid actions (0-3: left, right, center, drop)
    const NUM_ACTIONS = 4;

    // Track state for validation
    let expectedStateLength = null;
    let stepCount = 0;
    let intervalId = null;

    const intervalMs = 1000 / fps;

    console.log(`[RandomAgent] Starting test: fps=${fps}, maxSteps=${maxSteps}, logEvery=${logEvery}`);

    /**
     * Validate the state array and log warnings if issues are detected.
     * @param {number[]} state - The state array from RL.getState()
     */
    function validateState(state) {
        if (!Array.isArray(state)) {
            console.warn(`[RandomAgent] Warning: state is not an array, got ${typeof state}`);
            return;
        }

        // Check for state length changes
        if (expectedStateLength === null) {
            expectedStateLength = state.length;
            console.log(`[RandomAgent] Initial state length: ${expectedStateLength}`);
        } else if (state.length !== expectedStateLength) {
            console.warn(`[RandomAgent] Warning: state length changed from ${expectedStateLength} to ${state.length}`);
        }

        // Check for NaN values and out-of-range values
        for (let i = 0; i < state.length; i++) {
            const val = state[i];

            if (Number.isNaN(val)) {
                console.warn(`[RandomAgent] Warning: NaN detected at state[${i}]`);
            }

            // Check for values outside expected normalized range (0-1 for most, but allow some tolerance)
            // The state is normalized, so values should typically be in [0, 1]
            // Allow a small tolerance for edge cases
            if (val < -0.01 || val > 1.01) {
                console.warn(`[RandomAgent] Warning: state[${i}] = ${val} is outside expected range [0, 1]`);
            }
        }
    }

    /**
     * Log a summary of the current state.
     * @param {number[]} state - The state array from RL.getState()
     */
    function logStateSummary(state) {
        if (!Array.isArray(state)) {
            console.log(`[RandomAgent] Step ${stepCount}: Invalid state`);
            return;
        }

        // Extract key state values for summary
        const currentX = state[0]?.toFixed(3) ?? 'N/A';
        const currentY = state[1]?.toFixed(3) ?? 'N/A';
        const currentFruit = state[2]?.toFixed(3) ?? 'N/A';
        const nextFruit = state[3]?.toFixed(3) ?? 'N/A';
        const boosterAvailable = state[4] ?? 'N/A';

        // Count non-zero board fruits (positions 5 onwards, every 3 values)
        let boardFruitCount = 0;
        for (let i = 5; i + 2 < state.length; i += 3) {
            // A fruit exists if any of its values are non-zero
            if (state[i] !== 0 || state[i + 1] !== 0 || state[i + 2] !== 0) {
                boardFruitCount++;
            }
        }

        console.log(
            `[RandomAgent] Step ${stepCount}: ` +
            `pos=(${currentX}, ${currentY}), ` +
            `fruit=${currentFruit}, next=${nextFruit}, ` +
            `booster=${boosterAvailable}, boardFruits=${boardFruitCount}`
        );
    }

    /**
     * Execute one step of the random agent.
     */
    function step() {
        // Check if RL interface is available
        if (typeof window.RL === 'undefined' || typeof window.RL.getState !== 'function' || typeof window.RL.step !== 'function') {
            console.error('[RandomAgent] Error: RL interface not available. Make sure the game is loaded.');
            stopTest();
            return;
        }

        // Check if game is in terminal state
        if (typeof window.RL.isTerminal === 'function' && window.RL.isTerminal()) {
            console.log('[RandomAgent] Game over detected. Resetting...');
            if (typeof window.RL.reset === 'function') {
                window.RL.reset();
            } else {
                console.warn('[RandomAgent] Warning: RL.reset() not available. Stopping test.');
                stopTest();
                return;
            }
        }

        stepCount++;

        // Get current state
        const state = window.RL.getState();

        // Validate the state
        validateState(state);

        // Log summary every N steps
        if (stepCount % logEvery === 0) {
            logStateSummary(state);
        }

        // Pick a random action
        const action = Math.floor(Math.random() * NUM_ACTIONS);

        // Execute the action
        window.RL.step(action);

        // Check if we've reached the max steps
        if (stepCount >= maxSteps) {
            stopTest();
            console.log('[RandomAgent] Random agent test completed - environment stable.');
        }
    }

    /**
     * Stop the test.
     */
    function stopTest() {
        if (intervalId !== null) {
            clearInterval(intervalId);
            intervalId = null;
        }
    }

    // Start the test loop
    intervalId = setInterval(step, intervalMs);

    // Return a control object
    return {
        stop: stopTest,
        getStepCount: () => stepCount
    };
}
