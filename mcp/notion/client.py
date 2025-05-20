from server import MCPStdioServer, MCPInterfaceWithServer
from openai import OpenAI
from dotenv import load_dotenv

import asyncio, logging, os


class OpenAIClient:
    def __init__(self, api_key: str, model_name: str = "gpt-4o-mini"):
        self._client = OpenAI(api_key=api_key)
        self._model_name = model_name
        self._messages: list[dict[str, str]] = []

    def add_message(self, role: str, message: str) -> None:
        self._messages.append({"role": role, "content": message})

    def get_response(self) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self._model_name,
                messages=self._messages,
            )
            logging.info(f"LLM Response :\n{[choice.message.content for choice in response.choices]}")

            return response.choices[0].message.content

        except Exception as e:
            logging.error(e)
            raise


class MCPHost:
    def __init__(self, llm_client: OpenAIClient, interface: MCPInterfaceWithServer):
        self._llm_client = llm_client
        self._interface = interface

    async def execute(self, query: str) -> None:
        try:
            await self._interface.init()
            tools_description = await self._interface.get_tools_description()
            system_message = (
                "You are a helpful assistant with access to these tools:\n\n"
                f"{tools_description}\n"
                "Choose the appropriate tool based on the user's question. "
                "If no tool is needed, reply directly.\n\n"
                "IMPORTANT: When you need to use a tool, you must ONLY respond with "
                "the exact JSON object format below, nothing else:\n"
                "{\n"
                '    "tool": "tool-name",\n'
                '    "arguments": {\n'
                '        "argument-name": "value"\n'
                "    }\n"
                "}\n\n"
                "After receiving a tool's response:\n"
                "1. Transform the raw data into a natural, conversational response\n"
                "2. Keep responses concise but informative\n"
                "3. Focus on the most relevant information\n"
                "4. Use appropriate context from the user's question\n"
                "5. Avoid simply repeating the raw data\n"
                "6. Only give me abvoe JSON format\n\n"
                "Please use only the tools that are explicitly defined above."
            )
            self._llm_client.add_message("system", system_message)
            self._llm_client.add_message("user", query)
            import json

            llm_response = self._llm_client.get_response()
            try:
                import re

                match = re.search(r"(\{.*\})", llm_response, re.DOTALL)
                result = match.group(1)
                access_tool_info: dict[str, str] = json.loads(result)
            except json.decoder.JSONDecodeError as e:
                logging.error(e)
                access_tool_info = llm_response
            logging.info(f"Selected tool :\n{access_tool_info}")

            self._llm_client.add_message("assistant", llm_response)
            tool_response = await self._interface.call_tool(
                access_tool_info.get("tool"), access_tool_info.get("arguments")
            )
            self._llm_client.add_message("system", tool_response)
            result = self._llm_client.get_response()
            logging.info(f"result : {result}")

        except Exception as e:
            logging.error(e)
            raise

        finally:
            self._interface.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    # run server
    load_dotenv()
    mcp_server = MCPStdioServer(name="notionApi", mcp_config_path="mcp_config.json")
    mcp_interface = MCPInterfaceWithServer(mcp_server=mcp_server)
    llm_client = OpenAIClient(os.getenv("OPENAI_API_KEY"))

    mcp_host = MCPHost(llm_client, mcp_interface)
    asyncio.run(mcp_host.execute("1cfb97e401d380369644d55d70639604 페이지 요약해줘"))
