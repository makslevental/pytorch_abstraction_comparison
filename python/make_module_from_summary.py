import re

counters = {
    "conv": 0,
    "bn": 0,
    "relu": 0,
    "maxpool": 0,
    "avgpool": 0,
    "fc": 0
}

f = open("/Users/maksim/Library/Application Support/JetBrains/PyCharm2020.2/scratches/buffer1.txt", "r")
f_lines = f.readlines()
x_lines = []
for i, line in enumerate(f_lines):
    if not line.strip():
        x_lines.append("\n")
        continue
    op = line.split('=')[0].split('.')[1]
    op, old_count = re.match(r"([a-z]*)(\d?)", op).groups()
    counters[op] += 1
    new_count = counters[op]
    f_lines[i] = line.replace(f"{op}{old_count}", f"{op}{new_count}")

    x_lines.append(f"x = self.{op}{new_count}(x)")

print("".join(f_lines))
print("\n".join(x_lines))
