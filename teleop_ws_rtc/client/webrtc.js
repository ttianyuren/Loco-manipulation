const video = document.getElementById('video');

// ICE server selection for local/remote
let useStun = false;
if (window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1') {
    useStun = true;
}
// Optionally, add a UI toggle for advanced users
// const stunToggle = document.getElementById('stun-toggle');
// stunToggle && stunToggle.addEventListener('change', e => { useStun = e.target.checked; });

const iceServers = useStun ? [{ urls: 'stun:stun.l.google.com:19302' }] : [];
const pc = new RTCPeerConnection({ iceServers });

pc.ontrack = (event) => {
    console.log('[WebRTC] ontrack event:', event);
    video.srcObject = event.streams[0];
};
pc.onicecandidate = (event) => {
    console.log('[WebRTC] ICE candidate:', event.candidate);
};
pc.onconnectionstatechange = () => {
    console.log('[WebRTC] Connection state:', pc.connectionState);
};

async function startWebRTC() {
    try {
        // Add only video transceiver before creating offer
        pc.addTransceiver('video', { direction: 'recvonly' });
        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);
        console.log('[WebRTC] Created offer:', offer);
        console.log('[WebRTC] Offer SDP:', offer.sdp);
        if (!offer.sdp.includes('m=video')) {
            console.warn('[WebRTC] Offer SDP does NOT contain video!');
        }
        // Send offer to server
        const resp = await fetch('/offer', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({sdp: offer.sdp, type: offer.type})
        });
        if (!resp.ok) {
            throw new Error('Failed to fetch /offer: ' + resp.status);
        }
        const answer = await resp.json();
        console.log('[WebRTC] Received answer:', answer);
        await pc.setRemoteDescription(new RTCSessionDescription(answer));
    } catch (err) {
        console.error('[WebRTC] Error during offer/answer:', err);
    }
}

// ICE config logic
function getIceServers() {
    const mode = document.getElementById('iceToggle')?.value || 'stun';
    if (mode === 'none' || mode === 'local') {
        return [];
    } else {
        return [{ urls: 'stun:stun.l.google.com:19302' }];
    }
}

// WebRTC connection logic (update to use getIceServers)
let peerConnection;
function startWebRTC() {
    const iceServers = getIceServers();
    peerConnection = new RTCPeerConnection({ iceServers });

    peerConnection.ontrack = (event) => {
        console.log('[WebRTC] ontrack event:', event);
        video.srcObject = event.streams[0];
    };
    peerConnection.onicecandidate = (event) => {
        console.log('[WebRTC] ICE candidate:', event.candidate);
    };
    peerConnection.onconnectionstatechange = () => {
        console.log('[WebRTC] Connection state:', peerConnection.connectionState);
    };

    (async () => {
        try {
            // Add only video transceiver before creating offer
            peerConnection.addTransceiver('video', {
                direction: 'recvonly',
                sendEncodings: [{ maxBitrate: 2000000 }] // 2 Mbps
            });
            const offer = await peerConnection.createOffer();
            await peerConnection.setLocalDescription(offer);
            console.log('[WebRTC] Created offer:', offer);
            console.log('[WebRTC] Offer SDP:', offer.sdp);
            if (!offer.sdp.includes('m=video')) {
                console.warn('[WebRTC] Offer SDP does NOT contain video!');
            }
            // Send offer to server
            const resp = await fetch('/offer', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({sdp: offer.sdp, type: offer.type})
            });
            if (!resp.ok) {
                throw new Error('Failed to fetch /offer: ' + resp.status);
            }
            const answer = await resp.json();
            console.log('[WebRTC] Received answer:', answer);
            await peerConnection.setRemoteDescription(new RTCSessionDescription(answer));
        } catch (err) {
            console.error('[WebRTC] Error during offer/answer:', err);
        }
    })();
}

// Reconnect on ICE mode change
window.addEventListener('DOMContentLoaded', () => {
    const iceToggle = document.getElementById('iceToggle');
    if (iceToggle) {
        iceToggle.addEventListener('change', () => {
            if (peerConnection) peerConnection.close();
            startWebRTC();
        });
    }
});

// --- Modern Control Panel Integration ---
let ws;
let movementInterval = null;
let currentMocapIndex = 0;
const targetNames = ['Base', 'Left Arm', 'Right Arm'];

function connectWebSocket() {
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${wsProtocol}//${window.location.host}/ws`;
    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        console.log('[WS] WebSocket opened');
        updateConnectionStatus(true);
        startLatencyTest();
    };
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'pong') {
            handlePong(data);
        } else if (data.type === 'frame') {
            // If you ever use frame fallback
        } else if (data.type === 'mocap_index') {
            updateMocapIndex(data.index);
        }
    };
    ws.onerror = (err) => {
        console.error('[WS] WebSocket error:', err);
    };
    ws.onclose = (event) => {
        console.log('[WS] WebSocket closed:', event);
        updateConnectionStatus(false);
    };
}
connectWebSocket();

function updateConnectionStatus(connected) {
    const statusElement = document.getElementById('connectionStatus');
    if (connected) {
        statusElement.textContent = 'Connected';
        statusElement.className = 'connection-status connected';
    } else {
        statusElement.textContent = 'Disconnected';
        statusElement.className = 'connection-status disconnected';
    }
}

function updateMocapIndex(index) {
    currentMocapIndex = index;
    document.getElementById('currentTarget').textContent = targetNames[index];
    document.querySelectorAll('.mocap-btn').forEach((btn) => {
        btn.classList.remove('active');
    });
    const activeBtn = document.querySelector('.mocap-btn[data-target="' + String(index) + '"]');
    if (activeBtn) {
        activeBtn.classList.add('active');
    }
}

function switchMocapTarget() {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'switch_mocap' }));
    }
}

document.querySelectorAll('.mocap-btn').forEach(btn => {
    btn.addEventListener('click', switchMocapTarget);
});

function sendControlCommand(command, direction) {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'control', command, direction }));
    }
}

function startMovement(command, direction) {
    sendControlCommand(command, direction);
    movementInterval = setInterval(() => {
        sendControlCommand(command, direction);
    }, 50);
}
function stopMovement() {
    if (movementInterval) {
        clearInterval(movementInterval);
        movementInterval = null;
    }
}

// Step size slider
const stepSizeSlider = document.getElementById('stepSize');
const stepSizeDisplay = document.getElementById('stepSizeDisplay');
stepSizeSlider.addEventListener('input', (e) => {
    const value = e.target.value;
    stepSizeDisplay.textContent = value;
    document.getElementById('stepSizeValue').textContent = value;
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'set_step_size', value }));
    }
});

// Latency measurement
let latencyStart = null;
let latencies = [];
function startLatencyTest() {
    setInterval(() => {
        latencyStart = performance.now();
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'ping', timestamp: latencyStart }));
        }
    }, 1000);
}
function handlePong(data) {
    if (latencyStart) {
        const latency = performance.now() - data.timestamp;
        latencies.push(latency);
        if (latencies.length > 50) latencies.shift();
        const avgLatency = latencies.reduce((a, b) => a + b) / latencies.length;
        document.getElementById('latency-display').textContent = `Avg Latency: ${avgLatency.toFixed(1)}ms`;
    }
}

// Keyboard controls
window.addEventListener('keydown', (e) => {
    switch(e.key) {
        case 'ArrowUp': e.preventDefault(); startMovement('move_x', 1); break;
        case 'ArrowDown': e.preventDefault(); startMovement('move_x', -1); break;
        case 'ArrowLeft': e.preventDefault(); startMovement('move_y', -1); break;
        case 'ArrowRight': e.preventDefault(); startMovement('move_y', 1); break;
        case 'q': case 'Q': startMovement('move_z', 1); break;
        case 'e': case 'E': startMovement('move_z', -1); break;
        case ' ': e.preventDefault(); switchMocapTarget(); break;
    }
});
window.addEventListener('keyup', (e) => {
    switch(e.key) {
        case 'ArrowUp': case 'ArrowDown': case 'ArrowLeft': case 'ArrowRight':
        case 'q': case 'Q': case 'e': case 'E': stopMovement(); break;
    }
});

startWebRTC();
