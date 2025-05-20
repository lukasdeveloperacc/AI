from langserve import RemoteRunnable

import base64


def invoke(host: str, base64_file_path: str, language: str = "korean"):
    try:
        chain = RemoteRunnable(host)
        base64_file: str = None
        with open(base64_file_path, "rb") as f:
            base64_file = base64.b64encode(f.read()).decode("utf-8")

        result = chain.invoke({"base64_file": base64_file, "language": language})
        print(f"Result : \n{result}")

    except Exception as e:
        raise Exception(f"Client Error : \n{e}")


async def ainvoke(host: str, base64_file_path: str, language: str = "korean"):
    try:
        chain = RemoteRunnable(host)
        base64_file: str = None
        with open(base64_file_path, "rb") as f:
            base64_file = base64.b64encode(f.read()).decode("utf-8")

        result = await chain.ainvoke({"base64_file": base64_file, "language": language})
        print(f"Result : \n{result}")

    except Exception as e:
        raise Exception(f"Client Error : \n{e}")


def stream(host: str, base64_file_path: str, language: str = "korean"):
    try:
        chain = RemoteRunnable(host)
        base64_file: str = None
        with open(base64_file_path, "rb") as f:
            base64_file = base64.b64encode(f.read()).decode("utf-8")

        for msg in chain.stream({"base64_file": base64_file, "language": language}):
            print(msg, end="", flush=True)

    except Exception as e:
        raise Exception(f"Client Error : \n{e}")


async def astream(host: str, base64_file_path: str, language: str = "korean"):
    try:
        chain = RemoteRunnable(host)
        base64_file: str = None
        with open(base64_file_path, "rb") as f:
            base64_file = base64.b64encode(f.read()).decode("utf-8")

        async for msg in chain.astream({"base64_file": base64_file, "language": language}):
            print(msg, end="", flush=True)

    except Exception as e:
        raise Exception(f"Client Error : \n{e}")


if __name__ == "__main__":
    import asyncio

    # invoke("http://localhost:1234/summary/", "data/1706.03762v7.pdf")
    # asyncio.run(ainvoke("http://localhost:8000/summary/", "data/1706.03762v7.pdf"))
    stream("http://localhost:8000", "data/1706.03762v7.pdf")
    # asyncio.run(astream("http://localhost:8000/", "data/1706.03762v7.pdf"))
