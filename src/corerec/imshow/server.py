"""
Server components for the IMShow connector system.
"""

import json
import threading
import time
import tempfile
import webbrowser
from pathlib import Path
from typing import Dict, Any
import http.server
import socketserver
from urllib.parse import urlparse, parse_qs

from .frontends import get_frontend_template

def start_api_server(connector, port: int = 8000, debug: bool = False):
    """
    Start the API server for handling recommendation requests.
    
    Args:
        connector: Connector instance
        port: Port to run the server on
        debug: Enable debug mode
    """
    try:
        import uvicorn
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel
        from typing import Optional, List
        
        # Create FastAPI app
        app = FastAPI(title="CoreRec IMShow API", version="1.0.0")
        
        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Models
        class InteractionRequest(BaseModel):
            user_id: str
            item_id: str
            action: str
            metadata: Optional[Dict[str, Any]] = {}
        
        # Routes
        @app.get("/")
        async def root():
            return {
                "message": "CoreRec IMShow API",
                "connector": connector.get_info()
            }
        
        @app.get("/recommendations/{user_id}")
        async def get_recommendations(user_id: str, num_items: int = 12):
            try:
                recommendations = connector.get_recommendations(
                    user_id=user_id,
                    num_items=num_items
                )
                return {
                    "user_id": user_id,
                    "recommendations": recommendations,
                    "total": len(recommendations)
                }
            except Exception as e:
                if debug:
                    import traceback
                    traceback.print_exc()
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/interactions")
        async def record_interaction(interaction: InteractionRequest):
            try:
                connector.record_interaction(
                    user_id=interaction.user_id,
                    item_id=interaction.item_id,
                    action=interaction.action,
                    **interaction.metadata
                )
                return {"status": "success"}
            except Exception as e:
                if debug:
                    import traceback
                    traceback.print_exc()
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/info")
        async def get_info():
            return connector.get_info()
        
        # Run the server
        print(f"🔌 Starting API server on port {port}...")
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="error")
        
    except ImportError:
        print("❌ FastAPI/uvicorn not installed. Falling back to simple HTTP server...")
        start_simple_api_server(connector, port, debug)

def start_simple_api_server(connector, port: int, debug: bool = False):
    """
    Fallback simple API server using built-in HTTP server.
    """
    class APIHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            try:
                path = urlparse(self.path).path
                query = parse_qs(urlparse(self.path).query)
                
                if path == "/":
                    response = {
                        "message": "CoreRec IMShow API",
                        "connector": connector.get_info()
                    }
                elif path.startswith("/recommendations/"):
                    user_id = path.split("/")[-1]
                    num_items = int(query.get("num_items", [12])[0])
                    
                    recommendations = connector.get_recommendations(
                        user_id=user_id,
                        num_items=num_items
                    )
                    response = {
                        "user_id": user_id,
                        "recommendations": recommendations,
                        "total": len(recommendations)
                    }
                elif path == "/info":
                    response = connector.get_info()
                else:
                    self.send_response(404)
                    self.end_headers()
                    return
                
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.end_headers()
                
                self.wfile.write(json.dumps(response).encode())
                
            except Exception as e:
                if debug:
                    import traceback
                    traceback.print_exc()
                
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
        
        def do_POST(self):
            try:
                if self.path == "/interactions":
                    content_length = int(self.headers['Content-Length'])
                    post_data = self.rfile.read(content_length)
                    interaction = json.loads(post_data.decode())
                    
                    connector.record_interaction(
                        user_id=interaction["user_id"],
                        item_id=interaction["item_id"],
                        action=interaction["action"],
                        **interaction.get("metadata", {})
                    )
                    
                    response = {"status": "success"}
                else:
                    self.send_response(404)
                    self.end_headers()
                    return
                
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.end_headers()
                
                self.wfile.write(json.dumps(response).encode())
                
            except Exception as e:
                if debug:
                    import traceback
                    traceback.print_exc()
                
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
        
        def do_OPTIONS(self):
            self.send_response(200)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()
        
        def log_message(self, format, *args):
            if debug:
                super().log_message(format, *args)
    
    with socketserver.TCPServer(("", port), APIHandler) as httpd:
        print(f"🔌 API server running on port {port}")
        httpd.serve_forever()

def start_frontend_server(frontend: str, port: int, api_port: int, title: str, description: str):
    """
    Start the frontend server to serve the HTML interface.
    
    Args:
        frontend: Frontend type
        port: Port to run frontend server on
        api_port: Port where API server is running
        title: Demo title
        description: Demo description
    """
    # Get the HTML template
    template = get_frontend_template(frontend)
    
    # Replace template variables
    html_content = template.replace("{{title}}", title)
    html_content = html_content.replace("{{description}}", description)
    html_content = html_content.replace("{{api_port}}", str(api_port))
    
    # Create a temporary HTML file
    temp_dir = Path(tempfile.gettempdir()) / "corerec_imshow"
    temp_dir.mkdir(exist_ok=True)
    
    html_file = temp_dir / f"{frontend}_demo.html"
    html_file.write_text(html_content, encoding='utf-8')
    
    class FrontendHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(temp_dir), **kwargs)
        
        def do_GET(self):
            if self.path == "/" or self.path == f"/{frontend}":
                self.path = f"/{frontend}_demo.html"
            
            return super().do_GET()
        
        def log_message(self, format, *args):
            # Suppress logging for cleaner output
            pass
    
    try:
        with socketserver.TCPServer(("", port), FrontendHandler) as httpd:
            print(f"🌐 Frontend server running on port {port}")
            print(f"✅ {title} is ready!")
            print(f"   Open: http://localhost:{port}")
            httpd.serve_forever()
    except KeyboardInterrupt:
        print(f"\n🛑 Frontend server stopped")
    except Exception as e:
        print(f"❌ Error starting frontend server: {e}")

def check_port_available(port: int) -> bool:
    """
    Check if a port is available.
    
    Args:
        port: Port number to check
        
    Returns:
        True if port is available
    """
    import socket
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('localhost', port))
            return True
        except OSError:
            return False

def find_available_port(start_port: int = 8000, max_tries: int = 100) -> int:
    """
    Find an available port starting from start_port.
    
    Args:
        start_port: Starting port number
        max_tries: Maximum number of ports to try
        
    Returns:
        Available port number
        
    Raises:
        RuntimeError: If no available port is found
    """
    for port in range(start_port, start_port + max_tries):
        if check_port_available(port):
            return port
    
    raise RuntimeError(f"No available port found in range {start_port}-{start_port + max_tries}")

class SimpleServer:
    """
    Simple server wrapper for easier management.
    """
    
    def __init__(self, connector, frontend_port: int = None, api_port: int = None):
        self.connector = connector
        self.frontend_port = frontend_port or find_available_port(8080)
        self.api_port = api_port or find_available_port(self.frontend_port + 1000)
        self.running = False
        self.threads = []
    
    def start(self, background: bool = False):
        """Start both API and frontend servers."""
        self.running = True
        
        # Start API server in background
        api_thread = threading.Thread(
            target=start_api_server,
            args=(self.connector, self.api_port, self.connector.debug),
            daemon=True
        )
        api_thread.start()
        self.threads.append(api_thread)
        
        # Give API server time to start
        time.sleep(2)
        
        if background:
            # Start frontend server in background
            frontend_thread = threading.Thread(
                target=start_frontend_server,
                args=(
                    self.connector.frontend,
                    self.frontend_port,
                    self.api_port,
                    self.connector.title,
                    self.connector.description
                ),
                daemon=True
            )
            frontend_thread.start()
            self.threads.append(frontend_thread)
            
            return f"http://localhost:{self.frontend_port}"
        else:
            # Start frontend server (blocking)
            start_frontend_server(
                self.connector.frontend,
                self.frontend_port,
                self.api_port,
                self.connector.title,
                self.connector.description
            )
    
    def stop(self):
        """Stop the servers."""
        self.running = False
        print("🛑 Stopping servers...")
    
    def get_url(self) -> str:
        """Get the frontend URL."""
        return f"http://localhost:{self.frontend_port}" 