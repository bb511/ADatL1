from pathlib import Path
import os

print("Hello world")
output_dir = Path(os.environ.get("ADL1T_OUTPUT_ROOT", Path(__file__).resolve().parent))
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / "hello.txt"
output_file.write_text("hello world\n", encoding="utf-8")
print(f"Wrote {output_file}")
