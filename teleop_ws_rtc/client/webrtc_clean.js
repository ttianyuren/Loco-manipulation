const video = document.getElementById('video');

// ICE server configuration
function getIceServers() {
    const mode = document.getElementById('iceToggle')?.value || 'none';
    if (mode === 'none') {
        return [];
    } else {
        return [{ urls: 'stun:stun.l.google.com:19302' }];
    }
}

// Single WebRTC connection
let pc = null;
let webrtcStatsInterval = null;

async function startWebRTC() {
    // Close existing connection if any
    if (pc) {
        pc.close();
    }
    
    const iceServers = getIceServers();
    pc = new RTCPeerConnection({ iceServers });

    pc.ontrack = (event) => {
        console.log('[WebRTC] ontrack event:', event);
        video.srcObject = event.streams[0];
        // Start stats collection when video track is received
        startWebRTCStatsCollection();
    };
    
    pc.onicecandidate = (event) => {
        console.log('[WebRTC] ICE candidate:', event.candidate);
    };
    
    pc.onconnectionstatechange = () => {
        console.log('[WebRTC] Connection state:', pc.connectionState);
        if (pc.connectionState === 'disconnected' || pc.connectionState === 'failed') {
            stopWebRTCStatsCollection();
        }
    };

    try {
        // Add only video transceiver before creating offer
        pc.addTransceiver('video', { 
            direction: 'recvonly',
            sendEncodings: [{ maxBitrate: 2000000 }] // 2 Mbps
        });
        
        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);
        console.log('[WebRTC] Created offer:', offer);
        
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

// WebRTC Statistics Collection for Latency Measurement
async function collectWebRTCStats() {
    if (!pc || pc.connectionState !== 'connected') {
        return;
    }

    try {
        const stats = await pc.getStats();
        let inboundVideoStats = null;
        
        // Find inbound video statistics
        stats.forEach((report) => {
            if (report.type === 'inbound-rtp' && report.mediaType === 'video') {
                inboundVideoStats = report;
            }
        });

        if (inboundVideoStats) {
            // Calculate video latency using jitter and other metrics
            const jitter = inboundVideoStats.jitter || 0;
            const packetsLost = inboundVideoStats.packetsLost || 0;
            const framesDecoded = inboundVideoStats.framesDecoded || 0;
            const framesDropped = inboundVideoStats.framesDropped || 0;
            
            // Estimate video latency (jitter + processing delays)
            const estimatedLatency = Math.round((jitter * 1000) + 16.67); // jitter in ms + frame time
            
            // Update video latency in the UI
            if (window.robotController) {
                window.robotController.updateVideoLatency(estimatedLatency);
            }
            
            console.log(`[WebRTC Stats] Jitter: ${jitter}ms, Packets Lost: ${packetsLost}, Frames: ${framesDecoded}/${framesDropped}, Est. Latency: ${estimatedLatency}ms`);
        }
    } catch (error) {
        console.error('[WebRTC] Stats collection error:', error);
    }
}

function startWebRTCStatsCollection() {
    // Collect stats every 2 seconds
    if (webrtcStatsInterval) {
        clearInterval(webrtcStatsInterval);
    }
    webrtcStatsInterval = setInterval(collectWebRTCStats, 2000);
    console.log('[WebRTC] Started statistics collection');
}

function stopWebRTCStatsCollection() {
    if (webrtcStatsInterval) {
        clearInterval(webrtcStatsInterval);
        webrtcStatsInterval = null;
        console.log('[WebRTC] Stopped statistics collection');
    }
}

// Handle ICE mode change
window.addEventListener('DOMContentLoaded', () => {
    const iceToggle = document.getElementById('iceToggle');
    if (iceToggle) {
        iceToggle.addEventListener('change', () => {
            console.log('[WebRTC] ICE mode changed, restarting connection');
            startWebRTC();
        });
    }
    
    // Start WebRTC automatically
    startWebRTC();
});
