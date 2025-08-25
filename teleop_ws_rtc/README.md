# TIAGo WebRTC Teleoperation

Real-time teleoperation interface for the TIAGo dual-arm robot using WebRTC for video streaming and WebSocket for control commands. Features low-latency video transmission, responsive controls, and comprehensive latency monitoring.

## Features

- **Real-time video streaming** via WebRTC with adaptive bitrate
- **Responsive robot control** with keyboard and mouse support
- **Target selection** between Base, Left Arm, and Right Arm
- **Latency monitoring** for ping and command response times
- **Step size adjustment** for precise movement control
- **Remote access** via ngrok tunneling

## Project Structure

```
teleop_ws_rtc/
├── server/
│   ├── main.py                    # Main server entry point
│   ├── tiago_sim.py              # TIAGo simulation with MuJoCo
│   ├── mujoco_video_stream.py    # Video streaming handler
│   ├── webrtc_server.py          # WebRTC server implementation
│   └── requirements.txt          # Python dependencies
└── client/
    ├── index.html                # Web interface
    └── webrtc.js                # Client-side WebRTC and controls
```

## Setup and Installation

### Prerequisites

- Python 3.8+
- MuJoCo physics engine
- ngrok (for remote access)

### 1. Install Dependencies

```bash
cd teleop_ws_rtc/server
pip install -r requirements.txt
```

### 2. Local Development

Start the teleoperation server:

```bash
cd server
python3 main.py
```

The server will start on `http://localhost:8080`

Open your web browser and navigate to `http://localhost:8080` to access the teleoperation interface.

## Remote Deployment with ngrok

### 1. Install ngrok

If ngrok is not already installed:

```bash
# Ubuntu/Debian
curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null
echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | sudo tee /etc/apt/sources.list.d/ngrok.list
sudo apt update && sudo apt install ngrok

# macOS
brew install ngrok

# Or download directly from https://ngrok.com/download
```

### 2. Start the Server

First, start the TIAGo teleoperation server:

```bash
cd teleop_ws_rtc/server
python3 main.py
```

Keep this terminal running.

### 3. Create ngrok Tunnel

In a **new terminal**, start ngrok to expose port 8080:

```bash
ngrok http 8080
```

You'll see output like:
```
Session Status                online
Account                       YourAccount (Plan: Free)
Version                       3.x.x
Region                        United States (us)
Web Interface                 http://127.0.0.1:4040
Forwarding                    https://abc123.ngrok-free.app -> http://localhost:8080

Connections                   ttl     opn     rt1     rt5     p50     p90
                              0       0       0.00    0.00    0.00    0.00
```

### 4. Access Remotely

Use the provided ngrok URL (e.g., `https://abc123.ngrok-free.app`) to access your TIAGo teleoperation interface from anywhere on the internet.

**Note**: Free ngrok URLs change each time you restart ngrok. For persistent URLs, consider upgrading to a paid ngrok plan.

## Usage Instructions

### Web Interface Controls

1. **Target Selection**: Click on Base, Left Arm, or Right Arm to select which part to control
2. **Movement Controls**: 
   - Use arrow buttons or keyboard arrows for movement
   - Q/E keys for up/down movement
   - Spacebar to switch between targets
3. **Step Size**: Adjust the slider for movement precision
4. **Latency Monitor**: View real-time ping and command response times

### Keyboard Shortcuts

- **Arrow Keys**: Move forward/backward/left/right
- **Q/E**: Move up/down
- **Space**: Switch between control targets
- **Mouse**: Click and hold movement buttons for continuous motion

## Performance and Latency

The interface includes comprehensive latency monitoring:

- **Ping Latency**: Round-trip time for WebSocket communication
- **Command Latency**: Time for control commands to be processed
- **Video Stream**: Real-time via WebRTC (separate from WebSocket latency)

Typical performance metrics:
- Local network: 10-50ms ping latency
- Internet (via ngrok): 50-200ms ping latency
- Command processing: 5-15ms

## Troubleshooting

### Common Issues

1. **Server won't start**:
   ```bash
   # Check if port 8080 is already in use
   lsof -i :8080
   # Kill any existing processes if needed
   ```

2. **Video not displaying**:
   - Check browser console for WebRTC errors
   - Ensure WebRTC is supported in your browser
   - Try refreshing the page

3. **High latency over ngrok**:
   - This is normal for free ngrok tunnels
   - Consider using a local network setup for better performance
   - Check your internet connection speed

4. **Controls not responsive**:
   - Check WebSocket connection status (should show "Connected")
   - Verify server logs for error messages
   - Try refreshing the browser

### Browser Compatibility

Tested browsers:
- ✅ Chrome/Chromium (recommended)
- ✅ Firefox
- ✅ Safari
- ✅ Edge

## Security Considerations

When using ngrok for remote access:

1. **Authentication**: Consider adding authentication to your server
2. **HTTPS**: ngrok provides HTTPS by default for security
3. **Firewall**: Ensure your local firewall allows the connections
4. **Access Control**: Monitor who has access to your ngrok URL

## Development

### Adding New Features

1. **Server-side**: Modify `main.py` for new WebSocket message types
2. **Client-side**: Update `index.html` and `webrtc.js` for new UI elements
3. **Simulation**: Extend `tiago_sim.py` for new robot behaviors

### Configuration

Key configuration options in `tiago_sim.py`:
- `frequency`: Simulation update rate (default: 200Hz)
- `step_size`: Default movement step size
- Render resolution: Currently set to 640x480 for performance

## License

This project is part of the larger TIAGo teleoperation framework. Please refer to the main project license.

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review browser console for error messages
3. Check server logs for detailed error information
4. Ensure all dependencies are properly installed
