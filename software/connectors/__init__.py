"""The connectors MEDO ships. Add an app = add its class here (a new file), no
core edit. TemplateConnector is intentionally excluded (it's the copy-me example)."""

from software.connectors.browser import BrowserConnector
from software.connectors.media import MediaConnector
from software.connectors.window import WindowConnector

CONNECTOR_CLASSES = [MediaConnector, WindowConnector, BrowserConnector]
