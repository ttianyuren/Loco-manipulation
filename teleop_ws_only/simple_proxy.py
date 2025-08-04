import asyncio
import websockets
import json
from aiohttp import web, web_ws, WSMsgType
from pathlib import Path

async def websocket_proxy(request):
    """Proxy WebSocket connections to the robot server"""
    ws = web_ws.WebSocketResponse()
    await ws.prepare(request)
    
    try:
        # Connect to your robot server
        robot_ws = await websockets.connect('ws://localhost:8765')
        
        # Forward messages both ways
        async def forward_to_robot():
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    await robot_ws.send(msg.data)
        
        async def forward_from_robot():
            async for msg in robot_ws:
                await ws.send_str(msg)
        
        await asyncio.gather(forward_to_robot(), forward_from_robot())
        
    except Exception as e:
        print(f"Proxy error: {e}")
    finally:
        await ws.close()
    
    return ws

async def serve_html(request):
    """Serve the web interface"""
    html_path = Path(__file__).parent / "web_interface.html"
    with open(html_path, 'r') as f:
        content = f.read()
    return web.Response(text=content, content_type='text/html')

app = web.Application()
app.router.add_get('/', serve_html)
app.router.add_get('/ws', websocket_proxy)

if __name__ == '__main__':
    web.run_app(app, host='0.0.0.0', port=9000)