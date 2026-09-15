from pathlib import Path

print("Hello world")
Path(__file__).resolve().with_name("hello.txt").write_text(
    "hello world\n", encoding="utf-8"
)
