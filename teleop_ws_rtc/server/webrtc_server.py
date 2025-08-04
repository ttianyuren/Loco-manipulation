import asyncio
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription
from mujoco_video_stream import MuJoCoVideoStream

class WebRTCServer:
    def __init__(self, sim):
        self.sim = sim
        self.pcs = set()
    async def offer(self, request):
        params = await request.json()
        print(f'[WebRTC] Received offer SDP: {params.get("sdp")[:100]}...')  # Debug print
        offer = RTCSessionDescription(sdp=params['sdp'], type=params['type'])
        pc = RTCPeerConnection()
        self.pcs.add(pc)
        video = MuJoCoVideoStream(self.sim)
        pc.addTrack(video)  # Only video track
        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        print(f'[WebRTC] Sending answer SDP: {pc.localDescription.sdp[:100]}...')  # Debug print
        return web.json_response({'sdp': pc.localDescription.sdp, 'type': pc.localDescription.type})
    async def shutdown(self):
        coros = [pc.close() for pc in self.pcs]
        await asyncio.gather(*coros)
