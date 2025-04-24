from mcp import ClientSession
from server import Server
from openai import OpenAI


class OpenAIClient:
    def __init__(self, api_key: str, model_name: str = "gpt-4o-mini"):
        self._client = OpenAI(api_key=api_key)
        self._model_name = model_name
        self._system_prompt: dict[str, str] = {"role": "system", "content": ""}
        self._user_prompt: dict[str, str] = {"role": "user", "content": ""}

    @property
    def system_prompt(self) -> str:
        return self._system_prompt.get("content")

    @system_prompt.setter
    def system_prompt(self, prompt: str) -> None:
        self._system_prompt.update({"content": prompt})

    @property
    def user_prompt(self) -> str:
        return self._user_prompt.get("content")

    @user_prompt.setter
    def user_prompt(self, prompt: str) -> None:
        self._user_prompt.update({"content": prompt})

    def get_response(self) -> str:
        messages = []
        if not (self.system_prompt and self.user_prompt):
            raise ValueError(
                f"Check system prompt : {self.system_prompt}, user prompt : {self.user_prompt}. You must set these"
            )
        else:
            messages.append(self._system_prompt)
            messages.append(self._user_prompt)

        try:
            response = self._client.chat.completions.create(
                model=self._model_name,
                messages=messages,
            )

            return response.choices[0].message.content

        except Exception as e:
            logging.error(e)
            raise


class Client:
    def __init__(self, session: ClientSession):
        self._session = session


if __name__ == "__main__":
    import os, asyncio, logging

    api_key = os.getenv("OPENAI_API_KEY")
    mcp_server = Server(name="slack", mcp_config_path="mcp_config.json")
    asyncio.create_task(mcp_server.run())

    # while mcp_server.session is None:
    #     logging.info("Waiting for server to start...")
    #     await asyncio.sleep(1)

    # tools = await mcp_server.list_tools()
    # logging.info(f"Tools: {tools}")

    # await asyncio.sleep(15)
    # mcp_server.stop()
