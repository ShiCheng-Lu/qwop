texts = [
    "mulberry, niece, merge, mute",
    "milk, errand, iceberg, amuse",
    "banana, easter, television",
    "balance, anteater, tuition",
    "broccoli, pumpkin, caboose",
    "bashful, boomerang, glossy"
]
x = "".join(texts[0].split(", "))

y = "".join(texts[1].split(", "))

print(x, y)

result = ""
for i in range(len(x)):
    if x[i] == y[i]:
        result += "1"
    else:
        result += "0"
print(result)
