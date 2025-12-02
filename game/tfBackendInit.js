/**
 * tfBackendInit.js - TensorFlow.js Backend Initialization
 * 
 * This module initializes the TensorFlow.js backend (WebGPU preferred, with WASM fallback)
 * BEFORE any model or tensor creation to prevent backend initialization errors.
 * 
 * Must be imported and executed BEFORE any other TensorFlow.js operations.
 * 
 * Note: This module relies on the global `tf` object loaded via CDN scripts in index.html.
 * The tf.js core library must be loaded before this module is executed.
 */

let backendInitialized = false;
let currentBackend = null;

/**
 * Initialize TensorFlow.js backend with WebGPU preference and WASM fallback.
 * This function MUST be called before any model creation or tensor operations.
 * 
 * Requires: Global `tf` object from @tensorflow/tfjs CDN script
 * 
 * @returns {Promise<string>} The initialized backend name ('webgpu' or 'wasm')
 */
export async function initTFBackend() {
    if (backendInitialized) {
        console.log(`[TF Backend] Already initialized with backend: ${currentBackend}`);
        return currentBackend;
    }

    try {
        await tf.setBackend('webgpu');
        await tf.ready();
        currentBackend = 'webgpu';
        backendInitialized = true;
        console.log("[TF Backend] WebGPU backend initialized");
        return currentBackend;
    } catch (err) {
        console.warn("[TF Backend] WebGPU failed, falling back to WASM:", err);
        try {
            await tf.setBackend('wasm');
            await tf.ready();
            currentBackend = 'wasm';
            backendInitialized = true;
            console.log("[TF Backend] WASM backend initialized");
            return currentBackend;
        } catch (wasmErr) {
            console.error("[TF Backend] Both WebGPU and WASM backends failed:", wasmErr);
            throw new Error("Failed to initialize any TensorFlow.js backend");
        }
    }
}

/**
 * Get the currently initialized backend name.
 * 
 * @returns {string|null} Backend name or null if not initialized
 */
export function getCurrentBackend() {
    return currentBackend;
}

/**
 * Check if the backend has been initialized.
 * 
 * @returns {boolean} True if backend is initialized
 */
export function isBackendInitialized() {
    return backendInitialized;
}
