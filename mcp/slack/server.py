from typing import Dict, Any
from mcp import StdioServerParameters, ClientSession, Tool
from contextlib import AsyncExitStack
from mcp.client.stdio import stdio_client

import json, shutil, os, asyncio, logging


class MCPStdioServer:
    def __init__(self, name: str, mcp_config_path: str):
        self._name = name
        self._mcp_config_path = mcp_config_path
        self._mcp_config = self.get_config()
        self._mcp_server_session: ClientSession = None
        self._mcp_server_params = self.make_server_params()
        self._stack = AsyncExitStack()
        self._cleanup_lock: asyncio.Lock = asyncio.Lock()
        self._stop_event: asyncio.Event = asyncio.Event()

    @property
    def session(self) -> ClientSession | None:
        if self._mcp_server_session is None:
            logging.warning("Server is not running")

        return self._mcp_server_session

    def stop(self):
        logging.info("Stopping MCP server")
        self._stop_event.set()

    def get_config(self) -> Dict[str, str]:
        with open(self._mcp_config_path, "r") as f:
            config: dict = json.load(f)
            config = config.get("mcpServers")

        config = config.get(self._name)

        if config is None:
            raise ValueError(f"Invalid config file: {self._mcp_config_path}")
        else:
            return config

    def make_server_params(self) -> StdioServerParameters:
        command = self._mcp_config.get("command")
        if command is None:
            raise ValueError("Missing command in config file")
        else:
            command = shutil.which(command)
            logging.info(f"Command : {command}")

        args = self._mcp_config.get("args")
        logging.info(f"Args : {args}")

        env: dict[str, str] = self._mcp_config.get("env")
        if env.get("SLACK_BOT_TOKEN") is None:
            env.update({"SLACK_BOT_TOKEN", os.getenv("SLACK_BOT_TOKEN")})
        if env.get("SLACK_TEAM_ID") is None:
            env.update({"SLACK_TEAM_ID", os.getenv("SLACK_TEAM_ID")})

        if not list(env.values()):
            raise ValueError("Missing environment variables: SLACK_BOT_TOKEN, SLACK_TEAM_ID")

        return StdioServerParameters(command=command, args=args, env=env)

    async def run(self) -> None:
        try:
            logging.info("Start running MCP Stdio Server")
            stdio_transport = await self._stack.enter_async_context(stdio_client(self._mcp_server_params))
            read, write = stdio_transport
            self._mcp_server_session = await self._stack.enter_async_context(ClientSession(read, write))
            await self._mcp_server_session.initialize()
            await self._stop_event.wait()

        except Exception as e:
            logging.error(f"Error: {e}")
            raise

        finally:
            logging.info("Cleaning up MCP server session")
            await self.cleanup()

    async def cleanup(self):
        async with self._cleanup_lock:
            try:
                await self._stack.aclose()
                self._mcp_server_session = None

            except Exception as e:
                logging.error(f"Error during cleanup of server : {e}")
                raise

    async def wait_until_ready(self, timeout: float = 60.0):
        start = asyncio.get_event_loop().time()
        while self._mcp_server_session is None:
            if asyncio.get_event_loop().time() - start > timeout:
                raise TimeoutError("Timed out waiting for MCP server to be ready")
            await asyncio.sleep(0.1)


class MCPInterfaceWithServer:
    def __init__(self, mcp_server: MCPStdioServer):
        self._mcp_server = mcp_server
        self._mcp_server_session = self._mcp_server.session

    async def init(self):
        if isinstance(self._mcp_server, MCPStdioServer):
            asyncio.create_task(self._mcp_server.run())

        await self._mcp_server.wait_until_ready()
        self._mcp_server_session = self._mcp_server.session

    def stop(self):
        self._mcp_server.stop()

    async def list_tools(self) -> list[tuple[str, list[Tool]]]:
        if not self._mcp_server_session:
            raise RuntimeError("Server is not running")
        else:
            logging.info("Run list_tools")

        response = await self._mcp_server_session.list_tools()

        return list(response)

    async def get_tools_description(self) -> list[str]:
        def format_for_llm(name: str, description: str, input_schema: dict[str, Any]) -> str:
            args_desc = []
            if "properties" in input_schema:
                properties: dict[str, dict[str, Any]] = input_schema["properties"]
                for param_name, param_info in properties.items():
                    arg_desc = f"- {param_name}: {param_info.get('description', 'No description')}"
                    if param_name in input_schema.get("required", []):
                        arg_desc += " (required)"
                    args_desc.append(arg_desc)

            return f"Tool: {name}\nDescription: {description}\nArguments:\n{chr(10).join(args_desc)}"

        tools_list = await self.list_tools()
        tools_description = []
        for item in tools_list:
            if isinstance(item, tuple) and item[0] == "tools":
                tools = item[1]
                for tool in tools:
                    tools_description.append(format_for_llm(tool.name, tool.description, tool.inputSchema))

        logging.info(f"Tools description :\n{tools_description}")
        return tools_description

    async def call_tool(self, tool_name: str, args: dict[str, Any]) -> str:
        if self._mcp_server_session is None:
            raise RuntimeError("Server is not running")

        try:
            result = await self._mcp_server_session.call_tool(tool_name, args)
            result = result.content[0].text
            logging.info(f"Result: {result}")
            return result

        except Exception as e:
            logging.error(f"Error calling tool {tool_name}: {e}")
            raise


async def main() -> None:
    mcp_server = MCPStdioServer(name="slack", mcp_config_path="mcp_config.json")
    asyncio.create_task(mcp_server.run())
    mcp_interface = MCPInterfaceWithServer(mcp_server=mcp_server)
    await mcp_interface.init()

    tools = await mcp_interface.list_tools()
    logging.info(f"Tools: {tools}")

    mcp_interface.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
