import json

with open("requests.log") as f:
    rows = [json.loads(line) for line in f if line.startswith("{")]

total = len(rows)
server_errors = sum(1 for r in rows if r["status"] >= 500)
client_errors = sum(1 for r in rows if 400 <= r["status"] < 500)

# p95 is measured on successful predictions only
latencies = sorted(
    r["latency_ms"] for r in rows if r["path"] == "/predict" and r["status"] == 200
)
p95 = latencies[int(0.95 * len(latencies)) - 1]
median = latencies[len(latencies) // 2]

print(f"total requests:       {total}")
print(f"server errors (5xx):  {server_errors}  ({server_errors / total:.1%})")
print(f"rejected input (4xx): {client_errors}  ({client_errors / total:.1%})")
print(f"successful predicts:  {len(latencies)}")
print(f"median latency:       {median} ms")
print(f"p95 latency:          {p95} ms")