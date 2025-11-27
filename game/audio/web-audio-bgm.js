/**
 * Web Audio API wrapper for background music with reliable volume control.
 * 
 * This module provides a robust solution for background music playback that:
 * - Uses Web Audio API GainNode for precise, reliable volume control
 * - Handles iOS Safari autoplay restrictions via AudioContext unlock
 * - Falls back to HTMLAudioElement.volume when Web Audio API is unavailable
 * - Prevents creating multiple MediaElementSource nodes for the same element
 * 
 * @module web-audio-bgm
 */

/**
 * Convert decibels to linear gain value
 * @param {number} db - Decibels (e.g., -12)
 * @returns {number} Linear gain value (0 to 1+)
 */
export function dbToLinear(db) {
    return Math.pow(10, db / 20);
}

/**
 * Convert linear gain to decibels
 * @param {number} linear - Linear gain value (0 to 1+)
 * @returns {number} Decibels
 */
export function linearToDb(linear) {
    if (linear <= 0) return -Infinity;
    return 20 * Math.log10(linear);
}

/**
 * Creates a background music controller with Web Audio API integration
 * 
 * @param {Object} options - Configuration options
 * @param {HTMLAudioElement|string} options.audioElOrSrc - Audio element or source URL
 * @param {number} [options.defaultGain=0.25] - Default gain level (linear, ~-12dB)
 * @param {boolean} [options.loop=true] - Whether to loop the audio
 * @returns {Object} Controller object with play/pause/volume methods
 */
export function createBgmController({ audioElOrSrc, defaultGain = 0.25, loop = true }) {
    let audioContext = null;
    let audioElement = null;
    let sourceNode = null;
    let gainNode = null;
    let isWebAudioAvailable = false;
    let isPlaying = false;
    let hasCreatedMediaElementSource = false;
    let currentGain = defaultGain;

    // Initialize audio element
    if (typeof audioElOrSrc === 'string') {
        audioElement = new Audio(audioElOrSrc);
    } else if (audioElOrSrc instanceof HTMLAudioElement) {
        audioElement = audioElOrSrc;
    } else {
        throw new Error('audioElOrSrc must be a string URL or HTMLAudioElement');
    }

    audioElement.loop = loop;

    // Try to initialize Web Audio API
    try {
        const AudioContextClass = window.AudioContext || window.webkitAudioContext;
        if (AudioContextClass) {
            audioContext = new AudioContextClass();
            isWebAudioAvailable = true;
            //console.log('Web Audio API initialized for background music');
        }
    } catch (error) {
        console.warn('Web Audio API not available, falling back to HTMLAudioElement.volume:', error);
        isWebAudioAvailable = false;
    }

    // Setup Web Audio pipeline if available
    if (isWebAudioAvailable && audioContext) {
        try {
            // Create gain node
            gainNode = audioContext.createGain();
            gainNode.gain.value = currentGain;
            gainNode.connect(audioContext.destination);

            //console.log(`Web Audio BGM initialized with gain ${currentGain} (${linearToDb(currentGain).toFixed(2)}dB)`);
        } catch (error) {
            console.warn('Failed to setup Web Audio pipeline:', error);
            isWebAudioAvailable = false;
        }
    }

    // Fallback to HTMLAudioElement.volume if Web Audio is not available
    if (!isWebAudioAvailable) {
        audioElement.volume = currentGain;
        //console.log(`Using HTMLAudioElement.volume fallback with gain ${currentGain}`);
    }

    /**
     * Connect the audio element to the Web Audio graph
     * Note: createMediaElementSource can only be called once per element
     */
    function connectAudioElement() {
        if (!isWebAudioAvailable || !audioContext || !gainNode) return;
        if (hasCreatedMediaElementSource) return; // Already connected

        try {
            sourceNode = audioContext.createMediaElementSource(audioElement);
            sourceNode.connect(gainNode);
            hasCreatedMediaElementSource = true;
            //console.log('MediaElementSource connected to Web Audio graph');
        } catch (error) {
            console.warn('Failed to create MediaElementSource (may already exist):', error);
            hasCreatedMediaElementSource = true; // Mark as attempted to avoid retries
        }
    }

    /**
     * Resume/unlock the AudioContext after user gesture (required for iOS Safari)
     */
    async function unlockAudioContext() {
        if (!isWebAudioAvailable || !audioContext) return;

        if (audioContext.state === 'suspended') {
            try {
                await audioContext.resume();
                //console.log('AudioContext resumed/unlocked');
            } catch (error) {
                console.warn('Failed to resume AudioContext:', error);
            }
        }
    }

    /**
     * Play the background music
     * @returns {Promise<void>}
     */
    async function play() {
        try {
            // Unlock audio context if needed (iOS Safari requirement)
            await unlockAudioContext();

            // Connect audio element to Web Audio graph if not already done
            connectAudioElement();

            // Play the audio
            const playPromise = audioElement.play();
            if (playPromise !== undefined) {
                await playPromise;
                isPlaying = true;
                //console.log('Background music started');
            }
        } catch (error) {
            console.warn('Failed to play background music:', error);
            isPlaying = false;
        }
    }

    /**
     * Pause the background music
     */
    function pause() {
        try {
            audioElement.pause();
            isPlaying = false;
            //console.log('Background music paused');
        } catch (error) {
            console.warn('Failed to pause background music:', error);
        }
    }

    /**
     * Set the gain/volume using a linear value (0 to 1)
     * @param {number} value - Linear gain value
     */
    function setGainLinear(value) {
        currentGain = Math.max(0, Math.min(1, value)); // Clamp to [0, 1]

        if (isWebAudioAvailable && gainNode) {
            gainNode.gain.value = currentGain;
        } else {
            audioElement.volume = currentGain;
        }

        //console.log(`BGM gain set to ${currentGain} (${linearToDb(currentGain).toFixed(2)}dB)`);
    }

    /**
     * Set the gain/volume using decibels
     * @param {number} db - Gain in decibels
     */
    function setGainDb(db) {
        const linear = dbToLinear(db);
        setGainLinear(linear);
    }

    /**
     * Get the current gain as a linear value
     * @returns {number} Current linear gain
     */
    function getGainLinear() {
        return currentGain;
    }

    /**
     * Check if the audio is currently playing
     * @returns {boolean} True if playing
     */
    function isPlayingFn() {
        return isPlaying && !audioElement.paused;
    }

    /**
     * Install one-time event listeners to unlock audio on user gesture
     * This is crucial for iOS Safari autoplay policy compliance
     * @returns {Function} Cleanup function to remove listeners
     */
    function unlockOnUserGesture() {
        const events = ['touchstart', 'click'];
        let unlocked = false;

        const unlockHandler = async () => {
            if (unlocked) return;
            unlocked = true;

            //console.log('User gesture detected, unlocking audio...');
            await unlockAudioContext();
            connectAudioElement();

            // Remove listeners after first unlock
            events.forEach(event => {
                document.removeEventListener(event, unlockHandler, true);
            });
        };

        // Add listeners with capture phase to catch early
        events.forEach(event => {
            document.addEventListener(event, unlockHandler, { capture: true, once: false });
        });

        //console.log('Audio unlock listeners installed');

        // Return cleanup function
        return () => {
            events.forEach(event => {
                document.removeEventListener(event, unlockHandler, true);
            });
        };
    }

    // Return the controller API
    return {
        play,
        pause,
        setGainLinear,
        setGainDb,
        getGainLinear,
        isPlaying: isPlayingFn,
        unlockOnUserGesture,
        // Expose helpers for convenience
        dbToLinear,
        linearToDb
    };
}
