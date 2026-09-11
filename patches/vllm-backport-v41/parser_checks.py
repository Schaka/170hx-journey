"""Check the parser edits in fixups.py. Needs no GPU.

Run inside a built image, with the checkpoint mounted at /model:

    podman run --rm -v /models/DeepSeek-V4.1-Flash:/model:ro \
        -v $PWD/parser_checks.py:/parser_checks.py:ro \
        --entrypoint python3 localhost/vllm-backport-v41:sm80 /parser_checks.py

The base image fails four of these checks. A build from this kit passes all
of them.
"""
import json
from vllm.parser.parser_manager import ParserManager
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from transformers import AutoTokenizer

TOK = AutoTokenizer.from_pretrained("/model", trust_remote_code=True)
cls = ParserManager.get_parser(
    tool_parser_name="deepseek_v41",
    reasoning_parser_name="deepseek_v41",
    enable_auto_tools=True,
)
assert cls is not None

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                       "required": ["city"]},
    },
}]

def run(text, tools=TOOLS, tool_choice="auto", chunk=7):
    req = ChatCompletionRequest(
        model="m", messages=[{"role": "user", "content": "hi"}],
        tools=tools, tool_choice=tool_choice, stream=True,
    )
    p = cls(TOK, req.tools)
    if hasattr(p, "adjust_request"):
        p.adjust_request(req)
    reasoning, content, calls = "", "", []
    pieces = [text[i:i + chunk] for i in range(0, len(text), chunk)]
    for n, delta in enumerate(pieces):
        dm = p.parse_delta(
            delta_text=delta, delta_token_ids=[], request=req,
            finished=(n == len(pieces) - 1),
        )
        if dm is None:
            continue
        reasoning += dm.reasoning or ""
        content += dm.content or ""
        for tc in (dm.tool_calls or []):
            if tc.index is not None and tc.index >= len(calls):
                calls.append({"name": "", "args": ""})
            if tc.function is not None:
                if tc.function.name:
                    calls[tc.index]["name"] += tc.function.name
                if tc.function.arguments:
                    calls[tc.index]["args"] += tc.function.arguments
    return reasoning, content, calls

D = "｜DSML｜"
INV = f'<{D} invoke name="get_weather"><{D} parameter name="city" string="true">Berlin</{D} parameter></{D} invoke>'

cases = []

# 1. A bare invoke inside the thinking block, naming a declared tool.
r, c, t = run(f"Let me check.{INV}")
cases.append(("orphan invoke in think -> call", bool(t) and t[0]["name"] == "get_weather", (r, c, t)))

# 2. Prose inside the thinking block that merely quotes the marker.
r, c, t = run(f'I could write <{D} invoke name="get_weather"> here but I will not. The answer is 4.')
cases.append(("quoted marker stays reasoning", not t and "I could write" in r and c.strip() == "", (r, c, t)))

# 3. tool_choice none must not recover.
r, c, t = run(f"Let me check.{INV}", tool_choice="none")
cases.append(("tool_choice none -> no call", not t, (r, c, t)))

# 4. An undeclared name must not recover.
bad = INV.replace("get_weather", "rm_rf")
r, c, t = run(f"Let me check.{bad}")
cases.append(("undeclared name -> no call", not t, (r, c, t)))

# 5. A V3.2 foreign wrapper quoted inside the thinking block.
r, c, t = run(f"See <{D}function_calls>x</{D}function_calls> for the old form. Done.")
cases.append(("foreign wrapper stays reasoning", not t and c.strip() == "", (r, c, t)))

# 6. A proper wrapped call after the thinking block.
r, c, t = run(f"Thinking.</think>Sure.<{D} calls>{INV}</{D} calls>")
cases.append(("wrapped call after think", bool(t) and t[0]["name"] == "get_weather", (r, c, t)))

# 7. Two bare invokes in a row.
r, c, t = run(f"Check both.{INV}{INV}")
cases.append(("two orphan invokes", len(t) == 2, (r, c, t)))

# 8. The near-miss envelope.
r, c, t = run(f"Thinking.</think><{D}calls>{INV}</{D}calls>")
cases.append(("near-miss envelope", bool(t) and t[0]["name"] == "get_weather", (r, c, t)))

# 9. A declared name in an invoke that never closes must stay text.
r, c, t = run(f'Let me check.<{D} invoke name="get_weather"><{D} parameter name="city" string="true">Berlin')
cases.append(("unclosed invoke -> no call", not t, (r, c, t)))

# 10. Text after a recovered invoke is the answer, not padding.
r, c, t = run(f"Check.{INV}The weather is mild.")
cases.append(("text after recovered call", bool(t) and "The weather is mild." in (r + c), (r, c, t)))

# 11. The same unclosed invoke after the thinking block. The base commits
# this at the name; the hold must wait for the invoke to close.
r, c, t = run(f'Done.</think>I could write <{D} invoke name="get_weather"><{D} parameter name="city" string="true">Berlin')
cases.append(("unclosed invoke in content -> no call", not t, (r, c, t)))

ok = True
for name, passed, detail in cases:
    print(f"{'PASS' if passed else 'FAIL'}  {name}")
    if not passed:
        ok = False
        print(f"      reasoning={detail[0]!r}")
        print(f"      content  ={detail[1]!r}")
        print(f"      calls    ={detail[2]!r}")
print("ALL PASS" if ok else "SOME FAILED")
