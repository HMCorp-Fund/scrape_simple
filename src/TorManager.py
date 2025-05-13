import re
import socket
import requests # type: ignore
import socks # type: ignore
import stem.process # type: ignore
import stem.control # type: ignore


class TorManager:
    def __init__(self):
        self.tor_process = None
        self.socks_port = 9050
        self.control_port = 9051
        
    def start_tor(self, use_existing=False):
        """Start the Tor process and configure connection.
        
        Args:
            use_existing (bool): If True, try to use an existing Tor instance 
                                instead of launching a new one.
        """
        print("Setting up Tor connection...")
        
        # Configure requests to use the SOCKS proxy
        socks.set_default_proxy(socks.SOCKS5, "localhost", self.socks_port)
        socket.socket = socks.socksocket
        
        # Check if Tor is already running
        if use_existing or self._is_tor_running():
            print("Using existing Tor process")
        else:
            # Launch a new Tor process
            print("Starting new Tor process...")
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
                    raise
        
        # Test connection regardless of whether we started Tor or are using an existing instance
        self._test_tor_connection()
    
    def _is_tor_running(self):
        """Check if Tor is already running on the specified ports."""
        try:
            # Try to connect to the control port
            controller = stem.control.Controller.from_port(port=self.control_port)
            controller.close()
            return True
        except stem.SocketError:
            return False
            
    def _test_tor_connection(self):
        """Test the Tor connection to ensure it's working."""
        print("Testing Tor connection...")
        try:
            response = requests.get("https://check.torproject.org/")
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