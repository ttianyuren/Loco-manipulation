 class RobotController {
            constructor() {
                this.socket = null;
                this.currentMocapIndex = 0;
                this.movementInterval = null;
                this.targetNames = ['Base', 'Left Arm', 'Right Arm'];
                // Latency monitoring
                this.latencies = [];
                this.videoLatencies = [];
                this.commandLatencies = [];
                this.domReady = false;
                
                // Initialize immediately - DOM should be ready by the time constructor is called
                console.log('RobotController constructor - checking DOM...');
                this.initializeUI();
                
                // Ensure latency elements exist with delay to handle any DOM timing issues
                setTimeout(() => {
                    console.log('Delayed latency elements check...');
                    this.ensureLatencyElementsExist();
                }, 100);
                
                this.connect();
                this.domReady = true; // Set ready after initialization
                
                // Make this instance globally available for WebRTC stats
                window.robotController = this;
            }

            initializeUI() {
                // Step size slider
                const stepSizeSlider = document.getElementById('stepSize');
                const stepSizeDisplay = document.getElementById('stepSizeDisplay');
                stepSizeSlider.addEventListener('input', (e) => {
                    const value = e.target.value;
                    stepSizeDisplay.textContent = value;
                    document.getElementById('stepSizeValue').textContent = value;
                    if (this.socket && this.socket.readyState === WebSocket.OPEN) {
                        this.socket.send(JSON.stringify({ type: 'set_step_size', value: value }));
                    }
                });
                // Mocap target buttons - send set_mocap and update color immediately
                document.querySelectorAll('.mocap-btn').forEach((btn, i) => {
                    btn.addEventListener('click', () => {
                        if (this.socket && this.socket.readyState === WebSocket.OPEN) {
                            this.socket.send(JSON.stringify({ type: 'set_mocap', index: i }));
                        }
                        this.updateMocapIndex(i); // Optimistic UI update
                    });
                });
            }
            connect() {
                const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${wsProtocol}//${window.location.host}/ws`;
                console.log('Connecting to WebSocket:', wsUrl);
                this.socket = new WebSocket(wsUrl);
                this.socket.onopen = () => {
                    console.log('Connected to robot');
                    this.updateConnectionStatus(true);
                    console.log('Starting latency test...');
                    this.startLatencyTest();
                };
                this.socket.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    console.log('Received WebSocket message:', data.type);
                    if (data.type === 'pong') {
                        this.handlePong(data);
                    } else if (data.type === 'mocap_index') {
                        this.updateMocapIndex(data.index);
                    } else if (data.type === 'command_latency') {
                        // Receive real server-side command latency
                        this.handleServerCommandLatency(data.latency);
                    }
                    // Note: Video is transmitted via WebRTC, not WebSocket
                };
                this.socket.onclose = (event) => {
                    console.log('WebSocket closed:', event.code, event.reason);
                    this.updateConnectionStatus(false);
                };
                this.socket.onerror = (error) => {
                    console.error('WebSocket error:', error);
                };
            }
            updateConnectionStatus(connected) {
                const statusElement = document.getElementById('connectionStatus');
                if (connected) {
                    statusElement.textContent = 'Connected';
                    statusElement.className = 'connection-status connected';
                } else {
                    statusElement.textContent = 'Disconnected';
                    statusElement.className = 'connection-status disconnected';
                }
            }
            updateMocapIndex(index) {
                console.log('[UI] updateMocapIndex called with index:', index);
                this.currentMocapIndex = index;
                document.getElementById('currentTarget').textContent = this.targetNames[index];
                document.querySelectorAll('.mocap-btn').forEach((btn, i) => {
                    if (i == index) { // Use == to match string/number
                        btn.classList.add('active');
                        console.log('[UI] Activating button', i, btn.textContent);
                    } else {
                        btn.classList.remove('active');
                    }
                });
            }
            sendControlCommand(command, direction) {
                if (this.socket && this.socket.readyState === WebSocket.OPEN) {
                    const commandStartTime = performance.now();
                    this.socket.send(JSON.stringify({ 
                        type: 'control', 
                        command, 
                        direction,
                        client_timestamp: Date.now() // Use absolute timestamp for server comparison
                    }));
                    // Measure local command processing latency (client-side only)
                    setTimeout(() => this.measureCommandLatency(commandStartTime), 1);
                }
            }

            // Latency monitoring methods
            startLatencyTest() {
                console.log('Starting latency test...');
                let pingCount = 0;
                const pingInterval = setInterval(() => {
                    if (this.socket && this.socket.readyState === WebSocket.OPEN) {
                        const timestamp = performance.now();
                        pingCount++;
                        // Log every ping to debug the issue
                        console.log(`Sending ping #${pingCount} with timestamp:`, timestamp);
                        try {
                            this.socket.send(JSON.stringify({ type: 'ping', timestamp: timestamp }));
                        } catch (error) {
                            console.error('Error sending ping:', error);
                        }
                    } else {
                        console.log(`Ping #${pingCount+1} skipped - WebSocket not ready, state:`, this.socket ? this.socket.readyState : 'null');
                    }
                }, 500); // Reduced to 500ms for faster feedback during debugging
            }

            handlePong(data) {
                console.log('Received pong:', data);
                if (data.timestamp) {
                    const roundTripLatency = performance.now() - data.timestamp;
                    console.log('Calculated ping latency:', roundTripLatency);
                    this.latencies.push(roundTripLatency);
                    if (this.latencies.length > 50) this.latencies.shift();
                    console.log(`Ping latencies array length: ${this.latencies.length}`);
                    
                    // Update display immediately, even with just one measurement
                    this.updateLatencyDisplay();
                } else {
                    console.warn('Pong message missing timestamp');
                }
            }

            handleServerCommandLatency(serverLatency) {
                // Replace client-side measurement with real server-side latency
                console.log('Received server command latency:', serverLatency);
                this.commandLatencies.push(serverLatency);
                if (this.commandLatencies.length > 20) this.commandLatencies.shift();
                
                // Update display immediately
                this.updateLatencyDisplay();
            }

            measureCommandLatency(commandStartTime) {
                // This measures client-side command preparation latency (should be very small)
                // We now use server-side latency instead, so this is mainly for debugging
                const commandLatency = performance.now() - commandStartTime;
                // Only log occasionally to reduce console spam
                if (Math.random() < 0.05) { // Reduced to 5%
                    console.log('Client command preparation latency:', commandLatency.toFixed(2), 'ms (debug only)');
                }
                // Don't add to commandLatencies array - we use server-side values now
            }

            updateVideoLatency(videoLatency) {
                console.log('Received video latency:', videoLatency);
                this.videoLatencies.push(videoLatency);
                if (this.videoLatencies.length > 30) this.videoLatencies.shift();
                console.log(`Video latencies array length: ${this.videoLatencies.length}`);
                this.updateLatencyDisplay();
            }

            updateLatencyDisplay() {
                // Force element creation on every update
                this.ensureLatencyElementsExist();
                
                // Calculate averages, but show even single measurements
                const avgPing = this.latencies.length > 0 ? this.latencies.reduce((a, b) => a + b) / this.latencies.length : 0;
                const avgVideo = this.videoLatencies.length > 0 ? this.videoLatencies.reduce((a, b) => a + b) / this.videoLatencies.length : 0;
                const avgCommand = this.commandLatencies.length > 0 ? this.commandLatencies.reduce((a, b) => a + b) / this.commandLatencies.length : 0;
                
                console.log(`*** LATENCY UPDATE *** Ping=${avgPing.toFixed(1)}ms (${this.latencies.length} samples), Video=${avgVideo.toFixed(1)}ms (${this.videoLatencies.length} samples), Command=${avgCommand.toFixed(1)}ms (${this.commandLatencies.length} samples)`);
                
                // Force immediate DOM update
                this.forceUpdateLatencyDOM('ping-latency', avgPing, [50, 100]);
                this.forceUpdateLatencyDOM('video-latency', avgVideo, [30, 60]);
                this.forceUpdateLatencyDOM('command-latency', avgCommand, [10, 25]);
            }

            forceUpdateLatencyDOM(elementId, value, thresholds) {
                // Brute force DOM update
                let element = document.getElementById(elementId);
                
                if (!element) {
                    console.warn(`FORCE UPDATE: ${elementId} not found, creating...`);
                    this.ensureLatencyElementsExist();
                    element = document.getElementById(elementId);
                }
                
                if (!element) {
                    console.error(`FORCE UPDATE: Failed to create ${elementId}`);
                    return;
                }
                
                const displayValue = value > 0 ? `${value.toFixed(1)}ms` : 'N/A';
                element.textContent = displayValue;
                
                // Apply color coding
                element.className = 'metric-value';
                if (value > 0) {
                    if (value < thresholds[0]) {
                        element.classList.add('good');
                    } else if (value < thresholds[1]) {
                        element.classList.add('warning');
                    } else {
                        element.classList.add('bad');
                    }
                }
                
                console.log(`FORCE UPDATE: Set ${elementId} to "${displayValue}" with class "${element.className}"`);
            }

            updateLatencyElementWithRetry(elementId, value, thresholds, retryCount = 0) {
                const element = document.getElementById(elementId);
                console.log(`Updating element ${elementId}: found=${!!element}, value=${value}, retry=${retryCount}`);
                
                if (!element) {
                    if (retryCount < 3) {
                        console.log(`Element ${elementId} not found, retrying in 100ms (attempt ${retryCount + 1})`);
                        setTimeout(() => {
                            this.ensureLatencyElementsExist();
                            this.updateLatencyElementWithRetry(elementId, value, thresholds, retryCount + 1);
                        }, 100);
                        return;
                    }
                    console.error(`Element ${elementId} not found after ${retryCount} retries!`);
                    return;
                }
                
                // Format the value
                const displayValue = value > 0 ? `${value.toFixed(1)}ms` : 'N/A';
                element.textContent = displayValue;
                console.log(`Set ${elementId} text to: ${displayValue}`);
                
                // Apply color coding based on thresholds
                element.className = 'metric-value';
                if (value > 0) {
                    if (value < thresholds[0]) {
                        element.classList.add('good');
                    } else if (value < thresholds[1]) {
                        element.classList.add('warning');
                    } else {
                        element.classList.add('bad');
                    }
                }
            }

            ensureLatencyElementsExist() {
                // Check if latency-display container exists
                let latencyDisplay = document.getElementById('latency-display');
                if (!latencyDisplay) {
                    console.warn('latency-display container not found! This should exist in HTML.');
                    return;
                }

                // Check if .latency-metrics container exists
                let metricsContainer = document.querySelector('.latency-metrics');
                if (!metricsContainer) {
                    console.log('Creating missing .latency-metrics container');
                    metricsContainer = document.createElement('div');
                    metricsContainer.className = 'latency-metrics';
                    
                    // Add a title
                    const title = document.createElement('div');
                    title.style.fontWeight = 'bold';
                    title.style.marginBottom = '10px';
                    title.textContent = 'Latency Monitor:';
                    latencyDisplay.appendChild(title);
                    latencyDisplay.appendChild(metricsContainer);
                }

                // Ensure each latency metric exists
                const metrics = [
                    { id: 'ping-latency', label: 'Ping' },
                    { id: 'video-latency', label: 'Video' },
                    { id: 'command-latency', label: 'Command' }
                ];

                metrics.forEach(metric => {
                    if (!document.getElementById(metric.id)) {
                        console.log(`Creating missing latency element: ${metric.id}`);
                        const metricDiv = document.createElement('div');
                        metricDiv.className = 'latency-metric';
                        metricDiv.innerHTML = `
                            <span class="metric-label">${metric.label}:</span>
                            <span id="${metric.id}" class="metric-value">N/A</span>
                        `;
                        metricsContainer.appendChild(metricDiv);
                    }
                });
            }

            updateLatencyElement(elementId, value, thresholds) {
                // This function is now legacy, use updateLatencyElementWithRetry instead
                this.updateLatencyElementWithRetry(elementId, value, thresholds);
            }
        }
        let robotController;
        
        // Multiple ways to ensure DOM is ready
        function initializeRobotController() {
            console.log('Initializing robot controller...');
            console.log('DOM ready state:', document.readyState);
            
            // Debug: Check if key elements exist
            console.log('latency-display exists:', !!document.getElementById('latency-display'));
            console.log('.latency-metrics exists:', !!document.querySelector('.latency-metrics'));
            
            robotController = new RobotController();
            
            // ICE toggle event
            const iceToggle = document.getElementById('iceToggle');
            if (iceToggle) {
                iceToggle.addEventListener('change', function() {
                    if (window.setIceConfig) window.setIceConfig(this.value);
                });
            }
        }
        
        // Ensure DOM is fully loaded before initializing
        function initializeRobotController() {
            console.log('Initializing robot controller...');
            robotController = new RobotController();
            
            // ICE toggle event
            const iceToggle = document.getElementById('iceToggle');
            if (iceToggle) {
                iceToggle.addEventListener('change', function() {
                    if (window.setIceConfig) window.setIceConfig(this.value);
                });
            }
        }
        
        // Multiple ways to ensure DOM is ready
        function initializeRobotController() {
            console.log('Initializing robot controller...');
            console.log('DOM ready state:', document.readyState);
            
            // Debug: Check if key elements exist
            console.log('latency-display exists:', !!document.getElementById('latency-display'));
            console.log('.latency-metrics exists:', !!document.querySelector('.latency-metrics'));
            
            robotController = new RobotController();
            
            // ICE toggle event
            const iceToggle = document.getElementById('iceToggle');
            if (iceToggle) {
                iceToggle.addEventListener('change', function() {
                    if (window.setIceConfig) window.setIceConfig(this.value);
                });
            }
        }
        
        // More robust DOM ready detection
        function waitForDOM(callback, maxAttempts = 10) {
            let attempts = 0;
            
            function checkDOM() {
                attempts++;
                console.log(`DOM check attempt ${attempts}`);
                
                const latencyDisplay = document.getElementById('latency-display');
                const latencyMetrics = document.querySelector('.latency-metrics');
                
                if (latencyDisplay && latencyMetrics) {
                    console.log('DOM elements found, initializing...');
                    callback();
                    return;
                }
                
                if (attempts < maxAttempts) {
                    console.log('DOM not ready, retrying in 200ms...');
                    setTimeout(checkDOM, 200);
                } else {
                    console.warn('DOM not ready after max attempts, initializing anyway...');
                    callback();
                }
            }
            
            checkDOM();
        }
        
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => waitForDOM(initializeRobotController));
        } else {
            waitForDOM(initializeRobotController);
        }
        
        // Fallback: also listen to window load event
        window.addEventListener('load', () => {
            if (!robotController) {
                console.log('Fallback initialization triggered');
                waitForDOM(initializeRobotController);
            }
        });
        // Single step movement function (one click = one step)
        function singleStep(command, direction) {
            if (!robotController) return;
            console.log(`Single step: ${command} ${direction > 0 ? '+' : '-'}`);
            robotController.sendControlCommand(command, direction);
        }
        
        // Legacy continuous movement functions (kept for compatibility)
        function startMovement(command, direction) {
            if (!robotController) return;
            robotController.sendControlCommand(command, direction);
            robotController.movementInterval = setInterval(() => {
                robotController.sendControlCommand(command, direction);
            }, 50);
        }
        function stopMovement() {
            if (robotController && robotController.movementInterval) {
                clearInterval(robotController.movementInterval);
                robotController.movementInterval = null;
            }
        }