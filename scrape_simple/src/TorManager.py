import re
import socket
import requests # type: ignore
import socks # type: ignore
import stem.process # type: ignore
import stem.control # type: ignore
import os
import sys
import subprocess


class TorManager:
    def __init__(self):
        self.tor_process = None
        self.socks_port = 9050
        self.control_port = 9051
        
    def start_tor(self, use_existing=False):
        """Start the Tor process and configure connection."""
        print("Setting up Tor connection...")
        
        # Configure requests to use the SOCKS proxy
        socks.set_default_proxy(socks.SOCKS5, "localhost", self.socks_port)
        socket.socket = socks.socksocket
        
        # Check if Tor is already running
        if use_existing or self._is_tor_running():
            print("Using existing Tor process")
        else:
            # Try to find Tor in PATH
            tor_path = self._find_tor_path()
            if not tor_path:
                raise OSError("Tor executable not found. Please install Tor: "
                              "sudo apt install tor (Ubuntu/Debian) or "
                              "brew install tor (macOS)")
            
            # Launch a new Tor process
            print(f"Starting new Tor process using {tor_path}...")
            try:
                self.tor_process = stem.process.launch_tor_with_config(
                    config={
                        'SocksPort': str(self.socks_port),
                        'ControlPort': str(self.control_port),
                    },
                    init_msg_handler=lambda line: print(f"Tor: {line}" if re.search("Bootstrapped", line) else ""),
                    take_ownership=True
                )
            except OSError as e:
                if "Failed to bind one of the listener ports" in str(e):
                    print("Tor ports already in use, attempting to use the existing Tor process")
                else:
                    raise OSError(f"Failed to start Tor: {e}. Make sure Tor is correctly installed.")
        
        # Test connection regardless of whether we started Tor or are using an existing instance
        self._test_tor_connection()
    
    def _find_tor_path(self):
        """Find the Tor executable path."""
        # Common locations for Tor
        common_locations = [
            "/usr/bin/tor",
            "/usr/local/bin/tor",
            "/opt/homebrew/bin/tor"
        ]
        
        # Check if tor is in PATH
        try:
            # Check if tor is in PATH
            result = subprocess.run(['which', 'tor'], 
                                    stdout=subprocess.PIPE, 
                                    stderr=subprocess.PIPE, 
                                    text=True)
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        
        # Check common locations
        for path in common_locations:
            if os.path.exists(path):
                return path
                
        return None
    
    def _is_tor_running(self):
        """Check if Tor is already running on the specified ports."""
        try:
            # Try to connect to the control port
            controller = stem.control.Controller.from_port(port=str(self.control_port))
            controller.close()
            return True
        except stem.SocketError:
            return False
            
    def _test_tor_connection(self):
        """Test the Tor connection to ensure it's working."""
        print("Testing Tor connection...")
        try:
            response = requests.get("https://check.torproject.org/", timeout=30)
            if "Congratulations" in response.text:
                print("Successfully connected to Tor network!")
            else:
                print("Connected to the internet, but not through Tor")
        except Exception as e:
            print(f"Error connecting to Tor: {e}")
            self.stop_tor()
            raise
            
    def stop_tor(self):
        """Stop the Tor process if we started it."""
        if self.tor_process:
            print("Stopping Tor...")
            self.tor_process.kill()
            self.tor_process = None